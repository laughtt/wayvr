use std::collections::HashMap;

use anyhow::Context;
use serde::Serialize;
use wayvr_ipc::{
    client::{WayVRClient, WayVRClientMutex},
    ipc,
    packet_client::{self, HandsfreeParams, PositionMode},
    packet_server,
};

pub struct WayVRClientState {
    pub wayvr_client: WayVRClientMutex,
    pub serial_generator: ipc::SerialGenerator,
    pub pretty_print: bool,
}

fn handle_empty_result(result: anyhow::Result<()>) {
    if let Err(e) = result {
        log::error!("{e:?}");
    }
}

fn handle_result<T: Serialize>(pretty_print: bool, result: anyhow::Result<T>) {
    match result {
        Ok(t) => {
            let maybe_json = if pretty_print {
                serde_json::to_string_pretty(&t)
            } else {
                serde_json::to_string(&t)
            };

            match maybe_json {
                Ok(json_string) => println!("{}", json_string),
                Err(e) => log::error!("Failed to serialize JSON: {e:?}"),
            }
        }
        Err(e) => log::error!("{e:?}"),
    }
}

pub async fn wvr_window_list(state: &mut WayVRClientState) {
    handle_result(
        state.pretty_print,
        WayVRClient::fn_wvr_window_list(
            state.wayvr_client.clone(),
            state.serial_generator.increment_get(),
        )
        .await
        .context("failed to list window displays"),
    );
}

pub async fn wvr_window_set_visible(
    state: &mut WayVRClientState,
    handle: packet_server::WvrWindowHandle,
    visible: bool,
) {
    handle_empty_result(
        WayVRClient::fn_wvr_window_set_visible(state.wayvr_client.clone(), handle, visible)
            .await
            .context("failed to set window visibility"),
    )
}

pub async fn wvr_process_get(
    state: &mut WayVRClientState,
    handle: packet_server::WvrProcessHandle,
) {
    handle_result(
        state.pretty_print,
        WayVRClient::fn_wvr_process_get(
            state.wayvr_client.clone(),
            state.serial_generator.increment_get(),
            handle,
        )
        .await
        .context("failed to get process"),
    );
}

pub async fn wvr_process_list(state: &mut WayVRClientState) {
    handle_result(
        state.pretty_print,
        WayVRClient::fn_wvr_process_list(
            state.wayvr_client.clone(),
            state.serial_generator.increment_get(),
        )
        .await
        .context("failed to list processes"),
    )
}

pub async fn wvr_process_terminate(
    state: &mut WayVRClientState,
    handle: packet_server::WvrProcessHandle,
) {
    handle_empty_result(
        WayVRClient::fn_wvr_process_terminate(state.wayvr_client.clone(), handle)
            .await
            .context("failed to terminate process"),
    )
}

#[allow(clippy::too_many_arguments)]
pub async fn wvr_process_launch(
    state: &mut WayVRClientState,
    exec: String,
    name: String,
    env: Vec<String>,
    resolution: [u32; 2],
    pos_mode: PositionMode,
    icon: Option<String>,
    args: String,
    userdata: HashMap<String, String>,
) {
    handle_result(
        state.pretty_print,
        WayVRClient::fn_wvr_process_launch(
            state.wayvr_client.clone(),
            state.serial_generator.increment_get(),
            packet_client::WvrProcessLaunchParams {
                env,
                exec,
                name,
                args,
                resolution,
                pos_mode,
                icon,
                userdata,
            },
        )
        .await
        .context("failed to launch process"),
    )
}

pub async fn wlr_input_capture(state: &mut WayVRClientState, grab: bool) {
    handle_empty_result(
        WayVRClient::fn_wvr_input_capture(state.wayvr_client.clone(), grab)
            .await
            .context("failed to change input capture"),
    )
}

pub async fn wlx_device_haptics(
    state: &mut WayVRClientState,
    device: usize,
    intensity: f32,
    duration: f32,
    frequency: f32,
) {
    handle_empty_result(
        WayVRClient::fn_wlx_device_haptics(
            state.wayvr_client.clone(),
            device,
            packet_client::WlxHapticsParams {
                intensity,
                duration,
                frequency,
            },
        )
        .await
        .context("failed to trigger haptics"),
    )
}

pub async fn wlx_show_hide(state: &mut WayVRClientState) {
    handle_empty_result(
        WayVRClient::fn_wlx_show_hide(state.wayvr_client.clone())
            .await
            .context("failed to trigger overlay show hide"),
    )
}

pub async fn wlx_reset_screens(state: &mut WayVRClientState) {
    handle_empty_result(
        WayVRClient::fn_wlx_reset_screens(state.wayvr_client.clone())
            .await
            .context("failed to reset screens"),
    )
}

pub async fn wlx_recenter(state: &mut WayVRClientState) {
    handle_empty_result(
        WayVRClient::fn_wlx_recenter(state.wayvr_client.clone())
            .await
            .context("failed to recenter"),
    )
}

pub async fn wlx_desk_layout(
    state: &mut WayVRClientState,
    params: wayvr_ipc::packet_client::WlxDeskLayoutParams,
) {
    handle_empty_result(
        WayVRClient::fn_wlx_desk_layout(state.wayvr_client.clone(), params)
            .await
            .context("failed to set the desk layout"),
    )
}

pub async fn wlx_desk_screens(
    state: &mut WayVRClientState,
    params: wayvr_ipc::packet_client::WlxDeskScreensParams,
) {
    handle_empty_result(
        WayVRClient::fn_wlx_desk_screens(state.wayvr_client.clone(), params)
            .await
            .context("failed to set the desk screens"),
    )
}

pub async fn wlx_screen_numbers(
    state: &mut WayVRClientState,
    params: wayvr_ipc::packet_client::WlxScreenNumbersParams,
) {
    handle_empty_result(
        WayVRClient::fn_wlx_screen_numbers(state.wayvr_client.clone(), params)
            .await
            .context("failed to set the screen numbers"),
    )
}

pub async fn wlx_screen_nudge(
    state: &mut WayVRClientState,
    params: wayvr_ipc::packet_client::WlxScreenNudgeParams,
) {
    handle_empty_result(
        WayVRClient::fn_wlx_screen_nudge(state.wayvr_client.clone(), params)
            .await
            .context("failed to nudge the screen"),
    )
}

pub async fn wlx_set_skybox(state: &mut WayVRClientState, skybox: Option<String>) {
    handle_empty_result(
        WayVRClient::fn_wlx_set_skybox(state.wayvr_client.clone(), skybox)
            .await
            .context("failed to set the backdrop"),
    )
}

pub async fn wlx_switch_set(state: &mut WayVRClientState, set: Option<usize>) {
    handle_empty_result(
        WayVRClient::fn_wlx_switch_set(state.wayvr_client.clone(), set)
            .await
            .context("failed to switch to set"),
    )
}

pub async fn wlx_handsfree(state: &mut WayVRClientState, params: HandsfreeParams) {
    handle_empty_result(
        WayVRClient::fn_wlx_handsfree(state.wayvr_client.clone(), params)
            .await
            .context("failed to set handsfree mode"),
    )
}

pub async fn wlx_panel_modify(
    state: &mut WayVRClientState,
    overlay: String,
    element: String,
    command: packet_client::WlxModifyPanelCommand,
) {
    handle_empty_result(
        WayVRClient::fn_wlx_modify_panel(
            state.wayvr_client.clone(),
            packet_client::WlxModifyPanelParams {
                overlay,
                element,
                command,
            },
        )
        .await
        .context("failed to modify panel"),
    )
}

pub async fn wlx_input_state(state: &mut WayVRClientState) {
    handle_result(
        state.pretty_print,
        WayVRClient::fn_wlx_input_state(
            state.wayvr_client.clone(),
            state.serial_generator.increment_get(),
        )
        .await
        .context("failed to get input state"),
    )
}

/// `wayvrctl gaze`: what the user looks at (silent eye tracking), as JSON. Fails (exit code 1) on an IPC error.
pub async fn wlx_gaze(state: &mut WayVRClientState) -> anyhow::Result<()> {
    let g = WayVRClient::fn_wlx_gaze(
        state.wayvr_client.clone(),
        state.serial_generator.increment_get(),
    )
    .await
    .context("failed to get the gaze (a WayVR without gaze support drops the connection)")?;
    let json = if state.pretty_print {
        serde_json::to_string_pretty(&g)
    } else {
        serde_json::to_string(&g)
    }?;
    println!("{json}");
    Ok(())
}

pub async fn wlx_prefer_hands(state: &mut WayVRClientState, on: Option<bool>) {
    handle_empty_result(
        WayVRClient::fn_wlx_prefer_hands(state.wayvr_client.clone(), on)
            .await
            .context("failed to set prefer-hands"),
    )
}

pub async fn wlx_gaze_laser(state: &mut WayVRClientState, on: Option<bool>) {
    handle_empty_result(
        WayVRClient::fn_wlx_gaze_laser(state.wayvr_client.clone(), on)
            .await
            .context("failed to set the gaze laser"),
    )
}
