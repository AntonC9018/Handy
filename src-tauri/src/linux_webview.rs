//! WebKitGTK needs a positive screen font resolution to compute CSS pixels.

use gtk::prelude::GtkSettingsExt;

pub fn ensure_font_resolution() {
    let Some(screen) = gtk::gdk::Screen::default() else {
        return;
    };
    let resolution = screen.resolution();
    if resolution.is_finite() && resolution > 0.0 {
        return;
    }

    // GDK uses -1 for an unspecified resolution. WebKit can treat that sentinel
    // as a real DPI, producing a negative devicePixelRatio and broken layouts.
    // Prefer an explicit GTK font DPI (stored in 1024ths), then the 96-DPI base.
    let dpi = gtk::Settings::default()
        .map(|settings| settings.gtk_xft_dpi())
        .filter(|dpi| *dpi > 0)
        .map(|dpi| f64::from(dpi) / 1024.0)
        .unwrap_or(96.0);
    screen.set_resolution(dpi);
    log::info!("Initialized unspecified GDK font resolution to {dpi} DPI");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a GTK display; run under Xvfb or a graphical session"]
    fn initializes_unspecified_resolution_and_preserves_desktop_dpi() {
        gtk::init().expect("GTK test display");
        let screen = gtk::gdk::Screen::default().expect("GDK test screen");
        let settings = gtk::Settings::default().expect("GTK test settings");

        settings.set_gtk_xft_dpi(144 * 1024);
        screen.set_resolution(-1.0);
        ensure_font_resolution();
        assert_eq!(screen.resolution(), 144.0);

        screen.set_resolution(120.0);
        ensure_font_resolution();
        assert_eq!(screen.resolution(), 120.0);

        settings.set_gtk_xft_dpi(-1);
        screen.set_resolution(-1.0);
        ensure_font_resolution();
        assert_eq!(screen.resolution(), 96.0);
    }
}
