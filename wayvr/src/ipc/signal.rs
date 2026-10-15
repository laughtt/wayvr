use crate::backend::wayvr::window;

#[derive(Clone)]
pub enum WayVRSignal {
    DeviceHaptics(usize, crate::backend::input::Haptics),
    SwitchSet(Option<usize>),
    Handsfree(wayvr_ipc::packet_client::HandsfreeParams),
    ShowHide,
    ResetScreens,
    Recenter,
    DeskLayout(wayvr_ipc::packet_client::WlxDeskLayoutParams),
    SetSkybox(Option<String>),
    DeskScreens(wayvr_ipc::packet_client::WlxDeskScreensParams),
    /// The gaze laser: Some(on/off), None = toggle
    GazeLaser(Option<bool>),
    CustomTask(crate::backend::task::ModifyPanelTask),
    WindowVisibilityChanged(window::WindowHandle, bool),
}
