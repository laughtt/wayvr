//! Bounded shutdown helpers.
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
//! leave at once ([`exit_now`]) and let the kernel tear the GPU context down. [`arm_watchdog`] bounds the whole
//! teardown in case anything else blocks.

use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use vulkano::device::Device;

/// Reap every exited child (non-blocking). Children still running are left alone.
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

/// Leave immediately: no destructors, no atexit handlers (they are what can hang).
pub fn exit_now(code: i32) -> ! {
    reap_children();
    // SAFETY: terminating the process.
    unsafe { libc::_exit(code) }
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

/// From now on, exit by force if the process is still alive after `limit`.
pub fn arm_watchdog(limit: Duration) {
    let _ = std::thread::Builder::new()
        .name("wayvr-exit-watchdog".into())
        .spawn(move || {
            std::thread::sleep(limit);
            log::error!(
                "Shutdown did not finish within {} s: exiting now without further cleanup.",
                limit.as_secs()
            );
            exit_now(0);
        });
}
