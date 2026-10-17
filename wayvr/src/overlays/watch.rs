use std::time::Duration;

use glam::{Affine3A, Quat, Vec3, vec3};
use wgui::{
    assets::AssetPath,
    components::button::ComponentButton,
    parser::{Fetchable, ParseDocumentParams},
};
use wlx_common::{
    common::LeftRight,
    windowing::{OverlayWindowState, Positioning},
};

use crate::{
    gui::{
        panel::{
            GuiPanel, NewGuiPanelParams, apply_custom_command, device_list::DeviceList,
            overlay_list::OverlayList, set_list::SetList,
        },
        timer::GuiTimer,
    },
    state::AppState,
    windowing::{Z_ORDER_WATCH, backend::OverlayEventData, window::OverlayWindowConfig},
};

pub const WATCH_NAME: &str = "watch";

pub const WATCH_POS: Vec3 = vec3(-0.03, -0.01, 0.125);
pub const WATCH_ROT: Quat = Quat::from_xyzw(-0.707_106_6, 0.000_796_361_8, 0.707_106_6, 0.0);

#[derive(Default)]
struct WatchState {
    device_list: DeviceList,
    overlay_list: OverlayList,
    set_list: SetList,
    clock_12h: bool,
}

pub fn create_watch(app: &mut AppState) -> anyhow::Result<OverlayWindowConfig> {
    let state = WatchState {
        clock_12h: app.session.config.clock_12h,
        ..Default::default()
    };
    let watch_xml = "gui/watch.xml";

    let mut panel =
        GuiPanel::new_from_template(app, watch_xml, state, NewGuiPanelParams::default())?;

    sets_or_overlays(&mut panel, app);

    let doc_params = ParseDocumentParams {
        globals: panel.layout.state.globals.clone(),
        path: AssetPath::FileOrBuiltIn(watch_xml),
        extra: panel.doc_extra.take().unwrap_or_default(),
    };

    panel.on_notify = Some(Box::new({
        let name = WATCH_NAME;
        move |panel, app, event_data| {
            let mut elems_changed = panel.state.overlay_list.on_notify(
                &mut panel.layout,
                &mut panel.parser_state,
                &event_data,
                &doc_params,
            )?;

            elems_changed |= panel.state.set_list.on_notify(
                &mut panel.layout,
                &mut panel.parser_state,
                &event_data,
                &doc_params,
            )?;

            elems_changed |= panel.state.device_list.on_notify(
                app,
                &mut panel.layout,
                &mut panel.parser_state,
                &event_data,
                &doc_params,
            )?;

            match event_data {
                OverlayEventData::EditModeChanged(edit_mode) => {
                    if let Ok(btn_edit_mode) = panel
                        .parser_state
                        .fetch_component_as::<ComponentButton>("btn_edit_mode")
                    {
                        btn_edit_mode.set_sticky_state(&mut panel.layout.common(), edit_mode);
                    }
                }
                OverlayEventData::SettingsChanged => {
                    panel.layout.mark_redraw();
                    sets_or_overlays(panel, app);

                    if app.session.config.clock_12h != panel.state.clock_12h {
                        panel.state.clock_12h = app.session.config.clock_12h;

                        let clock_root = panel.parser_state.get_widget_id("clock_root")?;
                        panel.layout.remove_children(clock_root);

                        panel.parser_state.instantiate_template(
                            &doc_params,
                            "Clock",
                            &mut panel.layout,
                            clock_root,
                            Default::default(),
                        )?;

                        elems_changed = true;
                    }
                }
                OverlayEventData::CustomCommand { element, command } => {
                    if let Err(e) = apply_custom_command(panel, app, &element, &command) {
                        log::warn!("Could not apply {command:?} on {name}/{element}: {e:?}");
                    } else {
                        elems_changed = true;
                    }
                }
                _ => {}
            }

            if elems_changed {
                panel.process_custom_elems(app);
            }

            Ok(())
        }
    }));

    panel
        .timers
        .push(GuiTimer::new(Duration::from_millis(100), 0));

    let positioning = Positioning::FollowHand {
        hand: LeftRight::Left,
        lerp: 1.0,
    };

    panel.update_layout(app)?;

    Ok(OverlayWindowConfig {
        name: WATCH_NAME.into(),
        z_order: Z_ORDER_WATCH,
        default_state: OverlayWindowState {
            grabbable: false,
            interactable: true,
            positioning,
            transform: Affine3A::from_scale_rotation_translation(
                Vec3::ONE * 0.115,
                WATCH_ROT,
                WATCH_POS,
            ),
            angle_fade: true,
            ..OverlayWindowState::default()
        },
        show_on_spawn: app.session.config.enable_watch,
        global: true,
        ..OverlayWindowConfig::from_backend(Box::new(panel))
    })
}

fn sets_or_overlays(panel: &mut GuiPanel<WatchState>, app: &mut AppState) {
    let visible = if app.session.config.sets_on_watch {
        [false, true]
    } else {
        [true, false]
    };

    let widget = [
        panel
            .parser_state
            .get_widget_id("panels_root")
            .unwrap_or_default(),
        panel
            .parser_state
            .get_widget_id("sets_root")
            .unwrap_or_default(),
    ];

    for i in 0..2 {
        panel
            .layout
            .alterables
            .set_widget_visible(widget[i], visible[i]);
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, path::PathBuf, rc::Rc};

    use wgui::{
        assets::AssetPath,
        components::{ComponentTrait, button::ComponentButton},
        font_config::WguiFontConfig,
        globals::WguiGlobals,
        layout::{Layout, LayoutParams},
        palette::WguiColorPalette,
        parser::{
            CustomAttribsInfoOwned, Fetchable, ParseDocumentExtra, ParseDocumentParams,
            ParserState, new_layout_from_assets,
        },
    };
    use wlx_common::locale::WayVRLangProvider;

    use crate::gui::asset::GuiAsset;

    /// Parses the built-in `gui/watch.xml` with the real wgui parser (CPU only, no Vulkan) and returns the layout (it
    /// owns the components), the parser state and the custom (`_attr`) attributes of every element that has some, with
    /// the id of the widget they belong to.
    fn parse_watch() -> (Layout, ParserState, Vec<CustomAttribsInfoOwned>) {
        let globals = WguiGlobals::new(
            Box::new(GuiAsset {}),
            &WayVRLangProvider::default(),
            &WguiFontConfig::default(),
            PathBuf::new(),
            WguiColorPalette::get_builtin("Default"),
        )
        .expect("globals");

        let custom: Rc<RefCell<Vec<CustomAttribsInfoOwned>>> = Rc::default();
        let sink = custom.clone();
        let extra = ParseDocumentExtra {
            on_custom_attribs: Some(Rc::new(move |info| sink.borrow_mut().push(info.to_owned()))),
            ..Default::default()
        };

        let (layout, state) = new_layout_from_assets(
            &ParseDocumentParams {
                globals,
                path: AssetPath::BuiltIn("gui/watch.xml"),
                extra,
            },
            LayoutParams::default(),
        )
        .expect("gui/watch.xml parses");

        let custom = custom.take();
        (layout, state, custom)
    }

    #[test]
    fn exit_work_button_runs_omarchyvr_desk_exit() {
        let (_layout, state, custom) = parse_watch();
        let button = state
            .fetch_component_as::<ComponentButton>("btn_exit_work")
            .expect("the watch has a btn_exit_work button");
        let id = button.base().get_id();
        let attribs = custom
            .iter()
            .find(|a| a.widget_id == id)
            .expect("btn_exit_work has custom attributes");
        // `::ShellExec` joins the words after the command with single spaces and runs them as `sh -c`
        assert_eq!(
            attribs.get_value("_press"),
            Some("::ShellExec omarchyvr desk exit")
        );
    }

    #[test]
    fn the_other_watch_elements_are_still_there() {
        let (_layout, state, _) = parse_watch();
        for id in [
            "btn_dashboard",
            "btn_keyboard",
            "btn_edit_mode",
            "btn_ai_mode",
            "btn_ai_preset",
        ] {
            assert!(
                state.fetch_component_by_id(id).is_ok(),
                "watch.xml lost button {id}"
            );
        }
        for id in ["sets_root", "panels_root", "clock_root", "devices_root"] {
            assert!(
                state.get_widget_id(id).is_ok(),
                "watch.xml lost element {id}"
            );
        }
    }
}
