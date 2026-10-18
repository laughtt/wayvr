use std::{
    collections::HashMap,
    process::{self, ExitCode},
    time::Duration,
};

use anyhow::Context;
use clap::Parser;
use env_logger::Env;
use wayvr_ipc::{
    client::WayVRClient,
    ipc,
    packet_client::{self, PositionMode},
    packet_server::{WvrProcessHandle, WvrWindowHandle},
};

use crate::helper::{
    WayVRClientState, wlr_input_capture, wlx_device_haptics, wlx_handsfree, wlx_input_state,
    wlx_desk_layout, wlx_desk_screens, wlx_screen_nudge, wlx_screen_place, wlx_screen_numbers, wlx_gaze, wlx_gaze_check, wlx_gaze_laser, wlx_gaze_learn, wlx_prefer_hands, wlx_panel_modify, wlx_recenter, wlx_reset_screens, wlx_set_skybox, wlx_show_hide, wlx_switch_set, wvr_process_get, wvr_process_launch,
    wvr_process_list, wvr_process_terminate, wvr_window_list, wvr_window_set_visible,
};

mod helper;
mod types;

fn main() -> ExitCode {
    env_logger::init_from_env(Env::default().default_filter_or("info"));
    let args = Args::parse();

    smol::block_on(async move {
        let mut state = WayVRClientState {
            wayvr_client: WayVRClient::new(&format!("wayvrctl-{}", process::id()))
                .await
                .inspect_err(|e| {
                    log::error!("Failed to initialize WayVR connection: {e:?}");
                    process::exit(1);
                })
                .unwrap(),
            serial_generator: ipc::SerialGenerator::new(),
            pretty_print: args.pretty,
        };

        let maybe_err = if let Subcommands::Batch { fail_fast } = args.command {
            run_batch(&mut state, fail_fast).await
        } else {
            run_once(&mut state, args).await
        };

        if let Err(e) = maybe_err {
            log::error!("{e:?}");
            return ExitCode::FAILURE;
        } else {
            std::thread::sleep(Duration::from_millis(20));
        }

        ExitCode::SUCCESS
    })
}

async fn run_batch(state: &mut WayVRClientState, fail_fast: bool) -> anyhow::Result<()> {
    let stdin = std::io::stdin();

    for (line_no, line) in stdin.lines().enumerate() {
        let line = line.context("error reading stdin")?;

        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }

        if let Err(e) = parse_run_line(state, &line)
            .await
            .with_context(|| format!("error on line {}", line_no + 1))
        {
            if fail_fast {
                return Err(e);
            } else {
                log::error!("{e:?}");
            }
        }
    }
    Ok(())
}

async fn parse_run_line(state: &mut WayVRClientState, line: &str) -> anyhow::Result<()> {
    let mut argv = shell_words::split(line).context("parse error")?;

    // clap expects argv[0] to be the binary name
    argv.insert(0, env!("CARGO_PKG_NAME").to_string());

    let args = Args::try_parse_from(argv).context("invalid arguments")?;
    run_once(state, args).await?;

    Ok(())
}

async fn run_once(state: &mut WayVRClientState, args: Args) -> anyhow::Result<()> {
    match args.command {
        Subcommands::Batch { .. } => {
            log::warn!("Ignoring recursive batch command");
        }
        Subcommands::InputState => {
            wlx_input_state(state).await;
        }
        Subcommands::WindowList => {
            wvr_window_list(state).await;
        }
        Subcommands::WindowSetVisible {
            handle,
            visible_0_or_1,
        } => {
            let handle =
                serde_json::from_str::<WvrWindowHandle>(&handle).context("Invalid handle")?;
            wvr_window_set_visible(state, handle, visible_0_or_1 != 0).await;
        }
        Subcommands::ProcessGet { handle } => {
            let handle =
                serde_json::from_str::<WvrProcessHandle>(&handle).context("Invalid handle")?;
            wvr_process_get(state, handle).await;
        }
        Subcommands::ProcessList => {
            wvr_process_list(state).await;
        }
        Subcommands::ProcessTerminate { handle } => {
            let handle = serde_json::from_str(&handle).context("Invalid handle")?;
            wvr_process_terminate(state, handle).await;
        }
        Subcommands::ProcessLaunch {
            exec,
            name,
            env,
            resolution,
            pos,
            icon,
            args,
        } => {
            let env = env.split(",").map(|s| s.to_string()).collect::<Vec<_>>();
            let resolution = resolution
                .split_once('x')
                .and_then(|(x, y)| Some([x.parse::<u32>().ok()?, y.parse::<u32>().ok()?]))
                .context(
                    "Invalid resolution format. Expecting <width>x<height>, for example: 1920x1080, 1280x720",
                )?;

            let pos_mode = match pos {
                PosModeEnum::Floating => PositionMode::Float,
                PosModeEnum::Static => PositionMode::Static,
                PosModeEnum::Anchored => PositionMode::Anchor,
            };

            wvr_process_launch(
                state,
                exec,
                name,
                env,
                resolution,
                pos_mode,
                icon.filter(|s| !s.is_empty()), // "" = no icon (lets scripts pass `args` positionally)
                args,
                HashMap::new(),
            )
            .await;
        }
        Subcommands::Haptics {
            device,
            intensity,
            duration,
            frequency,
        } => {
            wlx_device_haptics(state, device, intensity, duration, frequency).await;
        }
        Subcommands::ResetScreens => {
            wlx_reset_screens(state).await;
        }
        Subcommands::Recenter => {
            wlx_recenter(state).await;
        }
        Subcommands::DeskLayout {
            screens,
            curvature,
            width,
            distance,
        } => {
            wlx_desk_layout(
                state,
                packet_client::WlxDeskLayoutParams {
                    screens,
                    curvature,
                    width,
                    distance,
                },
            )
            .await;
        }
        Subcommands::DeskScreens {
            place,
            hide_others,
            width,
            distance,
            curvature,
            gap,
            specs,
        } => {
            let mut screens = Vec::new();
            for spec in &specs {
                screens.push(parse_desk_spec(spec)?);
            }
            wlx_desk_screens(
                state,
                packet_client::WlxDeskScreensParams {
                    screens,
                    place,
                    hide_others,
                    width,
                    distance,
                    curvature,
                    gap,
                },
            )
            .await;
        }
        Subcommands::ScreenNumbers { action, n } => {
            use packet_client::WlxScreenNumbersAction as A;
            if let Some(n) = n
                && !(1..=6).contains(&n)
            {
                anyhow::bail!("screen number must be 1 to 6, not {n}");
            }
            let action = match action {
                ScreenNumbersArg::Off => A::Off,
                ScreenNumbersArg::Ondemand => A::OnDemand,
                ScreenNumbersArg::Always => A::Always,
                ScreenNumbersArg::Show => A::Show,
                ScreenNumbersArg::Hide => A::Hide,
                ScreenNumbersArg::Flash => A::Flash,
            };
            wlx_screen_numbers(state, packet_client::WlxScreenNumbersParams { action, n }).await;
        }
        Subcommands::ScreenNudge { name, dir, step } => {
            use packet_client::WlxNudgeDir;
            let dir = match dir.as_str() {
                "left" => WlxNudgeDir::Left,
                "right" => WlxNudgeDir::Right,
                "up" => WlxNudgeDir::Up,
                "down" => WlxNudgeDir::Down,
                "closer" => WlxNudgeDir::Closer,
                "farther" => WlxNudgeDir::Farther,
                "bigger" => WlxNudgeDir::Bigger,
                "smaller" => WlxNudgeDir::Smaller,
                other => anyhow::bail!(
                    "bad direction '{other}': must be left, right, up, down, closer, farther, bigger or smaller"
                ),
            };
            wlx_screen_nudge(state, packet_client::WlxScreenNudgeParams { name, dir, step }).await;
        }
        Subcommands::ScreenPlace { name, rel, of } => {
            use packet_client::WlxPlaceRel;
            let rel = match rel.as_str() {
                "left" => WlxPlaceRel::Left,
                "right" => WlxPlaceRel::Right,
                "up" | "above" => WlxPlaceRel::Up,
                "down" | "below" => WlxPlaceRel::Down,
                "front" | "here" => WlxPlaceRel::Front,
                "swap" => WlxPlaceRel::Swap,
                other => anyhow::bail!("bad place '{other}': must be left, right, up, down, front or swap"),
            };
            if rel == WlxPlaceRel::Swap && of.is_none() {
                anyhow::bail!("swap needs --of OTHER");
            }
            wlx_screen_place(state, packet_client::WlxScreenPlaceParams { name, rel, of }).await;
        }
        Subcommands::Skybox { name } => {
            let s = match name.as_str() {
                "off" => None,
                "default" | "sky" => Some(String::new()),
                file => Some(file.to_string()),
            };
            wlx_set_skybox(state, s).await;
        }
        Subcommands::ShowHide => {
            wlx_show_hide(state).await;
        }
        Subcommands::Gaze => {
            wlx_gaze(state).await?;
        }
        Subcommands::GazeLearn { action, json } => {
            wlx_gaze_learn(state, matches!(action, GazeLearnAction::Forget), json).await?;
        }
        Subcommands::GazeCheck { json } => {
            wlx_gaze_check(state, json).await?;
        }
        Subcommands::GazeLaser { state: on } => {
            let on = match on {
                OnOffToggle::On => Some(true),
                OnOffToggle::Off => Some(false),
                OnOffToggle::Toggle => None,
            };
            wlx_gaze_laser(state, on).await;
        }
        Subcommands::Controllers { state: on } => {
            // the wire packet says "hands only": controllers on = hands only off
            let hands_only = match on {
                OnOffToggle::On => Some(false),
                OnOffToggle::Off => Some(true),
                OnOffToggle::Toggle => None,
            };
            wlx_prefer_hands(state, hands_only).await;
        }
        Subcommands::PreferHands { state: on } => {
            let on = match on {
                OnOffToggle::On => Some(true),
                OnOffToggle::Off => Some(false),
                OnOffToggle::Toggle => None,
            };
            wlx_prefer_hands(state, on).await;
        }
        Subcommands::PanelModify {
            overlay,
            element,
            command,
        } => {
            let command = match command {
                SubcommandPanelModify::SetText { text } => {
                    packet_client::WlxModifyPanelCommand::SetText(text.join(" "))
                }
                SubcommandPanelModify::SetColor { hex_color } => {
                    packet_client::WlxModifyPanelCommand::SetColor(hex_color)
                }
                SubcommandPanelModify::SetImage { absolute_path } => {
                    packet_client::WlxModifyPanelCommand::SetImage(absolute_path)
                }
                SubcommandPanelModify::SetVisible { visible_0_or_1 } => {
                    packet_client::WlxModifyPanelCommand::SetVisible(visible_0_or_1 != 0)
                }
                SubcommandPanelModify::SetStickyState {
                    sticky_state_0_or_1,
                } => packet_client::WlxModifyPanelCommand::SetStickyState(sticky_state_0_or_1 != 0),
            };

            wlx_panel_modify(state, overlay, element, command).await;
        }
        Subcommands::SwitchSet { set_or_0: set } => {
            let set = if set == 0 { None } else { Some((set - 1) as _) };
            wlx_switch_set(state, set).await;
        }
        Subcommands::Handsfree { command } => {
            wlx_handsfree(state, command.into()).await;
        }
        Subcommands::InputCapture { command } => {
            wlr_input_capture(state, matches!(command, GrabRelease::Grab)).await;
        }
    }
    Ok(())
}

/// A command-line interface for WayVR IPC
#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The command to run
    #[command(subcommand)]
    command: Subcommands,

    /// Pretty-print JSON output
    #[arg(short, long)]
    pretty: bool,
}

#[derive(clap::Parser, Debug)]
enum Subcommands {
    /// Read commands from stdout, one per line.
    Batch {
        /// Stop on the first error
        #[arg(short, long)]
        fail_fast: bool,
    },
    /// Get the positions of HMD & controllers
    InputState,
    /// List WayVR windows
    WindowList,
    /// Delete a WayVR display
    // DisplaySetLayout skipped
    /// Change the visibility of a window on a WayVR display
    WindowSetVisible {
        /// A JSON window handle returned by DisplayWindowList
        handle: String,
        visible_0_or_1: u8,
    },
    /// Retrieve information about a WayVR-managed process
    ProcessGet {
        /// A JSON process handle returned by ProcessList or ProcessLaunch
        handle: String,
    },
    /// List all processes managed by WayVR
    ProcessList,
    /// Terminate a WayVR-managed process
    ProcessTerminate {
        /// A JSON process handle returned by ProcessList or ProcessLaunch
        handle: String,
    },
    /// Launch a new process inside WayVR
    ProcessLaunch {
        /// Name for the overlay
        #[arg(short, long, default_value = "")]
        name: String,
        /// Environment variables, separated by comma
        #[arg(short, long, default_value = "")]
        env: String,
        /// Executable to run
        exec: String,
        #[arg(default_value = "1920x1080")]
        resolution: String,
        /// Default positioning
        pos: PosModeEnum,
        /// Absolute path to the app icon
        icon: Option<String>,
        /// Arguments to pass to executable
        #[arg(default_value = "")]
        args: String,
    },
    /// Trigger haptics on the user's controller
    Haptics {
        /// 0 for left, 1 for right controller
        device: usize,
        #[arg(short, long, default_value = "0.25")]
        intensity: f32,
        #[arg(short, long, default_value = "0.1")]
        duration: f32,
        #[arg(short, long, default_value = "0.1")]
        frequency: f32,
    },
    /// Toggle overlay show or hide
    ShowHide,
    /// Put all screens back in front of the head
    ResetScreens,
    /// Recenter the play space
    Recenter,
    /// Show the first N screens / app panels (0 = all) in an arc in front of the head, hide the rest
    DeskLayout {
        screens: u32,
        #[arg(short, long, default_value = "0")]
        curvature: f32,
        /// Width of each screen in metres
        #[arg(short, long, default_value = "2.0")]
        width: f32,
        /// Distance from the head in metres
        #[arg(short, long, default_value = "2.5")]
        distance: f32,
    },
    /// Place/show/hide screens by output name: SLOT=NAME (shown) or SLOT=NAME:hide, SLOT = main|left|right|up|up-left|up-right
    DeskScreens {
        /// Re-place the listed visible screens even if they are already shown
        #[arg(long)]
        place: bool,
        /// Hide every screen / app panel / mirror that is not listed
        #[arg(long)]
        hide_others: bool,
        /// Width of the main screen in metres
        #[arg(short, long, default_value = "2.0")]
        width: f32,
        /// Distance of the main screen from the head in metres
        #[arg(short, long, default_value = "2.5")]
        distance: f32,
        #[arg(short, long, default_value = "0")]
        curvature: f32,
        /// Gap between neighbouring screen edges in metres
        #[arg(long, default_value = "0.05")]
        gap: f32,
        /// SLOT=NAME or SLOT=NAME:hide (e.g. main=SUNSHINE left=PIMAX-LEFT right=PIMAX-RIGHT:hide)
        #[arg(required = true)]
        specs: Vec<String>,
    },
    /// Screen number badges (1-6, top centre of each desk screen). Mode: off | ondemand (default) | always.
    /// Now: show (flash all for ~5 s) | hide | flash [N] (flash all and highlight screen N)
    ScreenNumbers {
        action: ScreenNumbersArg,
        /// Screen to highlight (flash only), 1 = main, 2 = left, 3 = right, 4 = up, 5 = up-left, 6 = up-right
        n: Option<u8>,
    },
    /// Move/scale a screen in space: left|right|up|down|closer|farther|bigger|smaller
    ScreenNudge {
        /// Screen (overlay) name, e.g. SUNSHINE
        name: String,
        /// left|right|up|down|closer|farther|bigger|smaller
        dir: String,
        /// Fraction of the screen width (left/right), height (up/down), distance (closer/farther) or size (bigger/smaller)
        #[arg(short, long, default_value = "0.25")]
        step: f32,
    },
    /// Place a screen once, from where your head is now, so it overlaps no other screen; the others make room.
    /// left|right|up|down: beside --of OTHER (default: its nearest neighbour on that side, i.e. one place over);
    /// front: straight in front of you; swap: change places with --of OTHER
    ScreenPlace {
        /// Screen (overlay) name, e.g. PIMAX-LEFT
        name: String,
        /// left|right|up|down|front|swap
        rel: String,
        /// The reference screen (overlay name)
        #[arg(long)]
        of: Option<String>,
    },
    /// Backdrop over the main app: off, default (built-in sky) or a DDS file relative to the WayVR config root
    Skybox { name: String },
    /// Where the user is looking (silent eye tracking): overlay, UV, desktop pixel, fixations, as JSON
    Gaze,
    /// The subtle gaze laser (a dot or short ray where the eyes land; never clicks or moves the mouse)
    GazeLaser { state: OnOffToggle },
    /// Gaze click v2: the learned gaze correction (status) or forget it (forget)
    GazeLearn {
        action: GazeLearnAction,
        /// Print JSON {enabled, lessons, taken, refused, last_error_deg, offset_deg, key}
        #[arg(long)]
        json: bool,
    },
    /// Gaze click v2: look at a target 1.5 m ahead; a steady look becomes a lesson (never clicks).
    /// Prints {ok, error_before_deg, error_after_deg} or {ok: false, reason}
    GazeCheck {
        /// Print JSON
        #[arg(long)]
        json: bool,
    },
    /// Alias (inverse of `controllers`): on = hands only, controllers ignored entirely; off = controllers work too
    PreferHands { state: OnOffToggle },
    /// Controllers in Work: on = controllers work too, off (default) = hands only, controllers ignored entirely
    /// (voice: "use controllers" / "hands only")
    Controllers { state: OnOffToggle },
    /// Apply a modification to a panel element
    PanelModify {
        /// The name of the overlay (XML file name without extension)
        overlay: String,
        /// The id of the element to modify, as set in the XML
        element: String,
        /// Command to execute
        #[command(subcommand)]
        command: SubcommandPanelModify,
    },
    SwitchSet {
        /// Set number to switch to, 0 to hide all sets
        set_or_0: usize,
    },
    Handsfree {
        /// Command to execute
        #[command(subcommand)]
        command: SubcommandHandsfree,
    },
    InputCapture {
        command: GrabRelease,
    },
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum ScreenNumbersArg {
    Off,
    #[value(name = "ondemand", alias = "on-demand")]
    Ondemand,
    Always,
    Show,
    Hide,
    Flash,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum GazeLearnAction {
    Status,
    Forget,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum OnOffToggle {
    On,
    Off,
    Toggle,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum GrabRelease {
    Grab,
    Release,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum PosModeEnum {
    Floating,
    Anchored,
    Static,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum HandsfreeMode {
    /// No handsfree pointer control
    None,
    /// Pointer controlled by HMD
    Hmd,
    /// Pointer controlled by HMD. Left pinch click, right pinch grab.
    HmdPinch,
    /// Pointer controlled by eye gaze
    EyeTracking,
    /// Pointer controlled eye gaze. Left pinch click, right pinch grab.
    EyeTrackingPinch,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum HandsfreeAction {
    /// The click action
    Click,
    /// The grab action
    Grab,
    /// Right-click modifier (use with click)
    RightModifier,
    /// Middle-click modifier (use with click)
    MiddleModifier,
}

#[derive(clap::Parser, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum SubcommandHandsfree {
    /// Set the handsfree mode
    SetMode { mode: HandsfreeMode },
    /// Press and hold an action
    Press { action: HandsfreeAction },
    /// Release a held action
    Release { action: HandsfreeAction },
    /// Toggle the state of an action
    Toggle { action: HandsfreeAction },
    /// Emulate a joystick scroll
    Scroll { amount: f32 },
}

#[derive(clap::Parser, Debug)]
#[allow(clippy::enum_variant_names)]
enum SubcommandPanelModify {
    /// Set the text of a <label> or <Button>
    SetText {
        /// Text that needs to be set
        #[arg(num_args = 1.., action = clap::ArgAction::Append)]
        text: Vec<String>,
    },
    /// Set the color of a <rectangle> or <label> or monochrome <sprite>
    SetColor {
        /// Color in HTML hex format (#rrggbb or #rrggbbaa)
        hex_color: String,
    },
    /// Set the content of a <sprite> or <image>. Max size for <sprite> is 256x256.
    SetImage {
        /// Absolute path to a svg, gif, png, jpeg or webp image.
        absolute_path: String,
    },
    /// Set the visibility of a <div>, <rectangle>, <label>, <sprite> or <image>
    SetVisible { visible_0_or_1: u8 },
    /// Set the sticky state of a <Button>. Intended for buttons without `sticky="1"`.
    SetStickyState { sticky_state_0_or_1: u8 },
}

/// `SLOT=NAME` or `SLOT=NAME:hide`
fn parse_desk_spec(spec: &str) -> anyhow::Result<packet_client::WlxDeskScreen> {
    use packet_client::WlxDeskSlot;
    let (slot, rest) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("bad spec '{spec}': expected SLOT=NAME or SLOT=NAME:hide"))?;
    let slot = match slot {
        "main" => WlxDeskSlot::Main,
        "left" => WlxDeskSlot::Left,
        "right" => WlxDeskSlot::Right,
        "up" => WlxDeskSlot::Up,
        "up-left" => WlxDeskSlot::UpLeft,
        "up-right" => WlxDeskSlot::UpRight,
        other => anyhow::bail!("bad slot '{other}' in '{spec}': must be main, left, right, up, up-left or up-right"),
    };
    let (name, visible) = match rest.strip_suffix(":hide") {
        Some(n) => (n, false),
        None => (rest, true),
    };
    if name.is_empty() {
        anyhow::bail!("bad spec '{spec}': empty screen name");
    }
    Ok(packet_client::WlxDeskScreen {
        name: name.to_string(),
        slot,
        visible,
    })
}
