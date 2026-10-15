//! Bounded shutdown.
//!
//! WayVR's GPU state is shared with the OpenXR runtime (Monado): the runtime's client compositor submits work on
//! our queue and signals semaphores it imported from the service. At exit two things went wrong:
//! * objects (swapchain images, the compositor's timeline semaphore) were freed while that work was still
//!   pending, so the GPU wrote into unmapped memory (NVIDIA Xid 31, PBDMA semaphore write);
//! * once the service is gone, a queue wait can never finish; vulkano's `Queue` drop calls `vkQueueWaitIdle`,
//!   and the NVIDIA driver waits by spinning, so the process hung at 100 % of a core and ignored signals.
//!
//! So the OpenXR backend first waits for the GPU with a time limit ([`quiesce_gpu`]), and only frees XR and
//! Vulkan objects once it is idle. If it never gets idle, freeing anything is unsafe and waiting is pointless:
//! leave at once ([`exit_now`]) and let the kernel tear the GPU context down.
//!
//! # The exit deadline (incident 2026-10-15, "wayvr-exit-spin")
//!
//! The whole exit is bounded by a deadline armed at its very start ([`begin_exit`]: the first SIGINT/SIGTERM, or
//! the main loop seeing `RUNNING == false`). The first version spawned a watchdog thread when the teardown began
//! and logged through `log::error!` before `_exit`. That can block forever when the main thread hangs inside the
//! dynamic loader: a library destructor run by `dlclose` (or a constructor run by `dlopen`) holds glibc's
//! `dl_load_lock` while it spins, and a thread's first use of a thread-local with a destructor (Rust's thread
//! start, `tracing`'s per-thread state on the first log line) registers it through `__cxa_thread_atexit_impl`,
//! which takes that very lock. That is what happened: main thread at 100 % in `R` state, the watchdog parked in
//! a futex, never firing; the second-signal path also logged first and would block the same way. So now:
//! * the watchdog thread is started with the program ([`start_watchdog`]), long before anything can hold the
//!   loader lock, and when it fires it uses nothing but `clock_gettime`, `write(2)`, `waitpid` and `_exit`;
//! * a kernel backstop (`setitimer` + `SIGALRM` at its default action) kills the process
//!   [`BACKSTOP_EXTRA`] after the deadline even if no thread of ours can run userspace code;
//! * SIGINT/SIGTERM are handled in signal-handler context ([`on_term_signal`], async-signal-safe), not by a
//!   thread that may be blocked: the first one starts the exit, a second one exits at once when the OpenXR
//!   objects are already released, otherwise right after they are ([`mark_xr_released`]).
//!
//! Nothing here exits before the OpenXR objects are released, except the deadline.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::Duration,
};

use vulkano::device::Device;

/// Time from the first exit request to the forced exit.
pub const EXIT_DEADLINE: Duration = Duration::from_secs(10);
/// Once the OpenXR objects are released, the rest (our Vulkan objects, the compositor, child processes, the XR
/// instance) gets at most this long (still never past [`EXIT_DEADLINE`]).
pub const AFTER_RELEASE_BUDGET: Duration = Duration::from_secs(5);
/// The kernel backstop fires this long after the deadline (the watchdog thread should have exited by then).
pub const BACKSTOP_EXTRA: Duration = Duration::from_secs(2);
/// How often the watchdog thread looks at the deadline.
const WATCHDOG_TICK: Duration = Duration::from_millis(50);

/// Exit code of a forced exit after the deadline.
pub const DEADLINE_EXIT_CODE: i32 = 0;
/// Exit code of an exit by a second SIGINT/SIGTERM.
pub const SECOND_SIGNAL_EXIT_CODE: i32 = 130;

/// `CLOCK_MONOTONIC` nanoseconds of the forced exit; `u64::MAX` = not armed.
static DEADLINE_NS: AtomicU64 = AtomicU64::new(u64::MAX);
static EXIT_STARTED: AtomicBool = AtomicBool::new(false);
static XR_RELEASED: AtomicBool = AtomicBool::new(false);
static EXIT_ASAP: AtomicBool = AtomicBool::new(false);

/// Monotonic clock in nanoseconds (async-signal-safe, no locks).
fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: plain clock read into a local.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
    (ts.tv_sec as u64).saturating_mul(1_000_000_000) + ts.tv_nsec as u64
}

fn dur_ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// Write to stderr without any lock, allocation or thread-local (async-signal-safe).
pub fn raw_eprint(msg: &[u8]) {
    let mut rest = msg;
    while !rest.is_empty() {
        // SAFETY: writing a valid buffer to fd 2.
        let n = unsafe { libc::write(2, rest.as_ptr().cast(), rest.len()) };
        if n <= 0 {
            if n < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return;
        }
        rest = &rest[n as usize..];
    }
}

/// Reap every exited child (non-blocking). Children still running are left alone. Async-signal-safe.
pub fn reap_children() {
    loop {
        let mut status = 0;
        // SAFETY: plain waitpid on any child, non-blocking.
        let pid = unsafe { libc::waitpid(-1, &raw mut status, libc::WNOHANG) };
        if pid <= 0 {
            break;
        }
    }
}

/// Leave immediately: no destructors, no atexit handlers (they are what can hang). Async-signal-safe.
pub fn exit_now(code: i32) -> ! {
    reap_children();
    // SAFETY: terminating the process.
    unsafe { libc::_exit(code) }
}

/// Make sure `SIGALRM` kills the process (default action, not blocked in the calling thread). Async-signal-safe.
fn sigalrm_default_unblocked() {
    // SAFETY: plain signal disposition/mask changes.
    unsafe {
        libc::signal(libc::SIGALRM, libc::SIG_DFL);
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&raw mut set);
        libc::sigaddset(&raw mut set, libc::SIGALRM);
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &raw const set, std::ptr::null_mut());
    }
}

/// (Re)arm the kernel backstop for the current deadline + [`BACKSTOP_EXTRA`]. Async-signal-safe.
fn arm_backstop() {
    let deadline = DEADLINE_NS.load(Ordering::SeqCst);
    if deadline == u64::MAX {
        return;
    }
    let ns = deadline.saturating_sub(now_ns()) + dur_ns(BACKSTOP_EXTRA);
    let ns = ns.max(1_000); // 0 would disarm the timer
    let it = libc::itimerval {
        it_interval: libc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
        it_value: libc::timeval {
            tv_sec: (ns / 1_000_000_000) as libc::time_t,
            tv_usec: ((ns % 1_000_000_000) / 1_000) as libc::suseconds_t,
        },
    };
    sigalrm_default_unblocked();
    // SAFETY: arming the process's real-time interval timer.
    unsafe { libc::setitimer(libc::ITIMER_REAL, &raw const it, std::ptr::null_mut()) };
}

/// Move the deadline to `now + limit` if that is earlier than the current one. Async-signal-safe.
fn deadline_within(limit: Duration) {
    let at = now_ns().saturating_add(dur_ns(limit));
    let old = DEADLINE_NS.fetch_min(at, Ordering::SeqCst);
    if at < old {
        arm_backstop();
    }
}

/// Time left until the forced exit (None = not armed).
#[cfg(test)]
fn time_left() -> Option<Duration> {
    let d = DEADLINE_NS.load(Ordering::SeqCst);
    (d != u64::MAX).then(|| Duration::from_nanos(d.saturating_sub(now_ns())))
}

/// The exit has begun: from now on the process is gone within `limit` at the latest. The first call arms the
/// deadline (later calls can only bring it closer). Returns true on the first call. Async-signal-safe.
pub fn begin_exit(limit: Duration) -> bool {
    let first = !EXIT_STARTED.swap(true, Ordering::SeqCst);
    deadline_within(limit);
    first
}

/// True once a second SIGINT/SIGTERM asked to leave as soon as it is safe (the main loop should stop waiting
/// for the runtime's session state changes).
pub fn exit_asap() -> bool {
    EXIT_ASAP.load(Ordering::SeqCst)
}

/// SIGINT/SIGTERM action. Runs in signal-handler context: async-signal-safe calls only (atomics, write, waitpid,
/// setitimer, _exit).
pub fn on_term_signal() {
    crate::RUNNING.store(false, Ordering::SeqCst);
    if begin_exit(EXIT_DEADLINE) {
        return;
    }
    // A second SIGINT/SIGTERM while the first one's teardown is still running: the user (or the launcher) wants
    // out now. Leave at once if the OpenXR objects are already released (nothing GPU-side can be pending then);
    // otherwise as soon as they are (mark_xr_released), and the deadline still bounds it.
    if XR_RELEASED.load(Ordering::SeqCst) {
        raw_eprint(b"wayvr: second termination signal during shutdown: exiting immediately.\n");
        exit_now(SECOND_SIGNAL_EXIT_CODE);
    }
    if !EXIT_ASAP.swap(true, Ordering::SeqCst) {
        raw_eprint(
            b"wayvr: second termination signal during shutdown: exiting as soon as the OpenXR objects are released.\n",
        );
    }
}

/// The GPU was idle and the OpenXR objects (swapchains, spaces, the session) are released: a pending second
/// signal exits now, and the rest of the teardown gets [`AFTER_RELEASE_BUDGET`].
pub fn mark_xr_released() {
    XR_RELEASED.store(true, Ordering::SeqCst);
    if EXIT_ASAP.load(Ordering::SeqCst) {
        raw_eprint(b"wayvr: OpenXR objects released; exiting now (second termination signal).\n");
        exit_now(SECOND_SIGNAL_EXIT_CODE);
    }
    deadline_within(AFTER_RELEASE_BUDGET);
}

/// Start the exit watchdog. Call it early in `main`, while no thread can hold the dynamic loader's lock: the
/// thread is fully set up now, and when it fires it touches no lock, no allocator and no thread-local.
pub fn start_watchdog() -> std::io::Result<()> {
    sigalrm_default_unblocked();
    let (ready_tx, ready_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("wayvr-exit-watchdog".into())
        .spawn(move || {
            sigalrm_default_unblocked();
            // Thread start-up (which registers thread-local destructors, under the loader's lock) is over.
            let _ = ready_tx.send(());
            drop(ready_tx);
            loop {
                // nanosleep: no lock, no thread-local.
                std::thread::sleep(WATCHDOG_TICK);
                if now_ns() >= DEADLINE_NS.load(Ordering::SeqCst) {
                    raw_eprint(
                        b"wayvr: shutdown did not finish before its deadline: exiting now without further cleanup.\n",
                    );
                    exit_now(DEADLINE_EXIT_CODE);
                }
            }
        })?;
    // Return only once the thread runs: a thread still starting up when the loader's lock is taken would block.
    let _ = ready_rx.recv_timeout(Duration::from_secs(2));
    Ok(())
}

/// Install SIGINT/SIGTERM handling: [`on_term_signal`] in signal-handler context.
pub fn install_term_handlers() -> std::io::Result<()> {
    for sig in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: on_term_signal only does async-signal-safe things.
        unsafe { signal_hook::low_level::register(sig, on_term_signal) }?;
    }
    Ok(())
}

/// Arms the exit deadline when dropped: declared after the GPU/XR locals of a backend's run function, it is dropped
/// before them on any early return (`?`), so an error path is bounded too.
pub struct ArmExitOnDrop;

impl Drop for ArmExitOnDrop {
    fn drop(&mut self) {
        begin_exit(EXIT_DEADLINE);
    }
}

/// Wait until the device is idle, for at most `limit`. Returns false when it did not get idle in time (the wait
/// keeps running on a detached thread; the caller must not free GPU objects then).
pub fn quiesce_gpu(device: &Arc<Device>, limit: Duration) -> bool {
    let (tx, rx) = mpsc::channel();
    let device = device.clone();
    let spawned = std::thread::Builder::new()
        .name("wayvr-gpu-idle".into())
        .spawn(move || {
            // SAFETY: called after the main loop: no other thread of ours records or submits work any more.
            let r = unsafe { device.wait_idle() };
            let _ = tx.send(r);
        });
    if spawned.is_err() {
        return false;
    }
    match rx.recv_timeout(limit) {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            // Device lost and the like: nothing is pending any more.
            log::warn!("vkDeviceWaitIdle at shutdown: {e:?}");
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    //! The forced exits are tested in child processes (this test binary re-run with `WAYVR_TEARDOWN_CHILD` set),
    //! as they end the process.
    use super::*;
    use std::process::{Command, Stdio};
    use std::time::Instant;

    const CHILD_ENV: &str = "WAYVR_TEARDOWN_CHILD";
    const ME: &str = "teardown::tests::child_entry";

    /// Run [`child_entry`] as a child in `mode`; returns (exit status, stderr, elapsed). Panics (after killing it)
    /// if it is still alive after `kill_after`.
    fn run_child(
        mode: &str,
        env: &[(&str, &str)],
        kill_after: Duration,
    ) -> (std::process::ExitStatus, String, Duration) {
        let exe = std::env::current_exe().unwrap();
        let t0 = Instant::now();
        let mut cmd = Command::new(exe);
        cmd.args(["--exact", ME, "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, mode)
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().unwrap();
        // Drain stderr on a thread so a chatty child cannot block on a full pipe.
        let mut stderr = child.stderr.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = std::io::Read::read_to_string(&mut stderr, &mut s);
            s
        });
        loop {
            if let Some(st) = child.try_wait().unwrap() {
                let t = t0.elapsed();
                return (st, reader.join().unwrap(), t);
            }
            if t0.elapsed() > kill_after {
                let _ = child.kill();
                let _ = child.wait();
                panic!("child {mode} still alive after {kill_after:?}: the exit is not bounded");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Spin in userspace forever (the incident's main thread).
    fn spin_forever() -> ! {
        let mut i = 0u64;
        loop {
            i = std::hint::black_box(i.wrapping_add(1));
        }
    }

    /// A shared object whose destructor spins forever: `dlclose` then holds glibc's `dl_load_lock` while it
    /// spins, as in the incident. None if no C compiler is around.
    fn build_spinning_lib(dir: &std::path::Path) -> Option<std::path::PathBuf> {
        std::fs::create_dir_all(dir).ok()?;
        let src = dir.join("spin.c");
        let lib = dir.join("libspin.so");
        std::fs::write(
            &src,
            "__attribute__((destructor)) static void fini(void) { volatile unsigned long i = 0; for (;;) i++; }\n",
        )
        .ok()?;
        let ok = Command::new("cc")
            .args(["-shared", "-fPIC", "-o"])
            .arg(&lib)
            .arg(&src)
            .status()
            .ok()?
            .success();
        ok.then_some(lib)
    }

    /// Child side of the forced-exit tests (a no-op in a normal test run).
    #[test]
    fn child_entry() {
        let Ok(mode) = std::env::var(CHILD_ENV) else {
            return;
        };
        match mode.as_str() {
            // The watchdog fires while the main thread spins holding the stderr lock.
            "spin" => {
                start_watchdog().unwrap();
                begin_exit(Duration::from_millis(300));
                let _held = std::io::stderr().lock();
                spin_forever();
            }
            // The incident: the main thread spins inside dlclose, holding the dynamic loader's lock. The old
            // watchdog (spawned late, logging through tracing) blocked in a futex here.
            "dlclose" => {
                let lib = std::env::var("WAYVR_TEARDOWN_LIB").unwrap();
                start_watchdog().unwrap();
                begin_exit(Duration::from_millis(300));
                let c = std::ffi::CString::new(lib).unwrap();
                // SAFETY: test library; dlclose runs its spinning destructor.
                unsafe {
                    let h = libc::dlopen(c.as_ptr(), libc::RTLD_NOW);
                    assert!(!h.is_null());
                    libc::dlclose(h);
                }
                unreachable!("the destructor never returns");
            }
            // No watchdog thread at all: the kernel backstop alone ends it.
            "backstop" => {
                begin_exit(Duration::from_millis(100));
                spin_forever();
            }
            // Second SIGTERM before the OpenXR objects are released: exit right after they are, not before.
            "second-signal" => {
                start_watchdog().unwrap();
                install_term_handlers().unwrap();
                // SAFETY: signalling ourselves (the handler runs synchronously in raise).
                unsafe { libc::raise(libc::SIGTERM) };
                assert!(!crate::RUNNING.load(Ordering::SeqCst));
                assert!(time_left().is_some(), "the first signal arms the deadline");
                unsafe { libc::raise(libc::SIGTERM) };
                assert!(
                    exit_asap(),
                    "the second signal must not exit before the release"
                );
                std::thread::sleep(Duration::from_millis(200));
                mark_xr_released();
                unreachable!("mark_xr_released exits after a second signal");
            }
            // Second SIGTERM after the release: immediate exit.
            "second-signal-after-release" => {
                install_term_handlers().unwrap();
                unsafe { libc::raise(libc::SIGTERM) };
                mark_xr_released();
                unsafe { libc::raise(libc::SIGTERM) };
                unreachable!("the second signal exits");
            }
            m => panic!("unknown mode {m}"),
        }
    }

    #[test]
    fn watchdog_fires_while_main_thread_spins() {
        let (st, err, t) = run_child("spin", &[], Duration::from_secs(10));
        assert_eq!(st.code(), Some(DEADLINE_EXIT_CODE), "{st:?} {err}");
        assert!(err.contains("deadline"), "{err}");
        assert!(
            t >= Duration::from_millis(300) && t < Duration::from_secs(2),
            "{t:?}"
        );
    }

    #[test]
    fn watchdog_fires_while_main_thread_spins_inside_dlclose() {
        let dir = std::env::temp_dir().join(format!("wayvr-teardown-test-{}", std::process::id()));
        let Some(lib) = build_spinning_lib(&dir) else {
            eprintln!("no C compiler: skipping the dlclose case");
            return;
        };
        let lib = lib.to_string_lossy().into_owned();
        let (st, err, t) = run_child(
            "dlclose",
            &[("WAYVR_TEARDOWN_LIB", &lib)],
            Duration::from_secs(10),
        );
        let _ = std::fs::remove_dir_all(&dir);
        // The watchdog (not the kernel backstop, which would be SIGALRM after 2 s more) must end it.
        assert_eq!(st.code(), Some(DEADLINE_EXIT_CODE), "{st:?} {err}");
        assert!(err.contains("deadline"), "{err}");
        assert!(t < Duration::from_secs(2), "{t:?}");
    }

    #[test]
    fn kernel_backstop_kills_without_the_watchdog() {
        use std::os::unix::process::ExitStatusExt;
        let (st, _, t) = run_child("backstop", &[], Duration::from_secs(10));
        assert_eq!(st.signal(), Some(libc::SIGALRM), "{st:?}");
        assert!(
            t >= BACKSTOP_EXTRA && t < BACKSTOP_EXTRA + Duration::from_secs(2),
            "{t:?}"
        );
    }

    #[test]
    fn second_signal_waits_for_the_xr_release_then_exits() {
        let (st, err, t) = run_child("second-signal", &[], Duration::from_secs(10));
        assert_eq!(st.code(), Some(SECOND_SIGNAL_EXIT_CODE), "{st:?} {err}");
        assert!(
            err.contains("as soon as the OpenXR objects are released"),
            "{err}"
        );
        assert!(err.contains("exiting now"), "{err}");
        assert!(t >= Duration::from_millis(200), "{t:?}");
    }

    #[test]
    fn second_signal_after_the_release_exits_at_once() {
        let (st, err, _) = run_child("second-signal-after-release", &[], Duration::from_secs(10));
        assert_eq!(st.code(), Some(SECOND_SIGNAL_EXIT_CODE), "{st:?} {err}");
        assert!(err.contains("exiting immediately"), "{err}");
    }
}
