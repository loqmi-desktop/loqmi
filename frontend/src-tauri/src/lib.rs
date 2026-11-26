use std::io;
use std::net::{SocketAddr, TcpStream};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, RunEvent};
#[cfg(target_os = "macos")]
use tauri::WindowEvent;
use tauri_plugin_opener::OpenerExt;

const BACKEND_PORT: u16 = 8765;

/// Handle to the spawned backend, so it can be stopped on app exit.
struct BackendProcess(Mutex<Option<Child>>);

/// Debug builds use the bundle in src-tauri/binaries/ (resource_dir() points
/// at target/debug in dev, where the bundle isn't copied). Release builds use
/// the real bundled resource directory.
fn backend_exe(app: &AppHandle) -> Option<PathBuf> {
    if cfg!(debug_assertions) {
        let _ = app;
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("binaries")
                .join("loqmi-backend-dist")
                .join("loqmi-backend"),
        )
    } else {
        app.path()
            .resource_dir()
            .ok()
            .map(|d| d.join("loqmi-backend-dist").join("loqmi-backend"))
    }
}

fn port_in_use() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], BACKEND_PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn spawn_backend(app: &AppHandle) -> io::Result<Child> {
    let exe = backend_exe(app)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "backend path unavailable"))?;
    Command::new(&exe)
        // Tells the backend's watchdog to exit if this app disappears.
        .env("loqmi_PARENT_WATCH", "1")
        // Own process group, so shutdown can signal the backend and every
        // child it forked in one go.
        .process_group(0)
        .spawn()
}

/// Graceful stop: SIGTERM the whole group, wait up to 3s, then SIGKILL.
/// Idempotent, because the child is taken out of the mutex on first call.
fn stop_backend(app: &AppHandle) {
    let Some(mut child) = app.state::<BackendProcess>().0.lock().unwrap().take() else {
        return;
    };
    let pgid = child.id() as i32;

    unsafe {
        libc::killpg(pgid, libc::SIGTERM);
    }

    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // Sweep anything still alive in the group (ESRCH if none, which is fine).
    unsafe {
        libc::killpg(pgid, libc::SIGKILL);
    }
    let _ = child.wait();
}

/// URLs the webview is allowed to navigate to itself.
fn is_app_url(url: &tauri::Url) -> bool {
    match url.scheme() {
        "tauri" | "about" | "blob" | "data" => true,
        "http" | "https" => {
            let host = url.host_str().unwrap_or("");
            host == "tauri.localhost"
                || (cfg!(debug_assertions) && host == "localhost" && url.port() == Some(3000))
        }
        _ => false,
    }
}

/// Safety net: the webview can never navigate away from the app.
/// External links are handed to the system browser / mail client instead.
fn navigation_guard<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::<R>::new("navigation-guard")
        .on_navigation(|webview, url| {
            if is_app_url(url) {
                return true;
            }
            if matches!(url.scheme(), "http" | "https" | "mailto" | "tel") {
                let _ = webview.opener().open_url(url.as_str(), None::<&str>);
            }
            false
        })
        .build()
}

/// Leave native full screen first, then hide, so macOS doesn't leave an
/// empty black Space behind.
#[cfg(target_os = "macos")]
fn hide_window(window: &tauri::Window) {
    if window.is_fullscreen().unwrap_or(false) {
        let _ = window.set_fullscreen(false);
        let w = window.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            while w.is_fullscreen().unwrap_or(false) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(50));
            }
            // is_fullscreen() can flip before the exit animation ends
            std::thread::sleep(Duration::from_millis(800));
            let _ = w.hide();
        });
    } else {
        let _ = window.hide();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // Register the Tauri Opener plugin so external links open in the default browser
        .plugin(tauri_plugin_opener::init())
        .plugin(navigation_guard())
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();

            if port_in_use() {
                // A stale backend from an earlier run would silently serve the UI.
                log::warn!(
                    "port {BACKEND_PORT} already in use; not spawning. \
                     Run `pkill -f loqmi-backend` if this is a stale process."
                );
            } else {
                match spawn_backend(&handle) {
                    Ok(child) => {
                        *app.state::<BackendProcess>().0.lock().unwrap() = Some(child);
                    }
                    Err(e) => log::error!("failed to spawn backend: {e}"),
                }
            }

            if let Some(window) = handle.get_webview_window("main") {
                let _ = window.show();
            }
            Ok(())
        })
        // macOS convention: the red X hides the window; the app keeps running
        // and stays in the Dock. Cmd+Q / Dock > Quit really quits.
        .on_window_event(|window, event| {
            #[cfg(target_os = "macos")]
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                hide_window(window);
            }
            #[cfg(not(target_os = "macos"))]
            let _ = (window, event);
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| match event {
            // Dock icon clicked while the window is hidden.
            #[cfg(target_os = "macos")]
            RunEvent::Reopen { has_visible_windows, .. } => {
                if !has_visible_windows {
                    if let Some(w) = app.get_webview_window("main") {
                        let _ = w.show();
                        let _ = w.set_focus();
                    }
                }
            }
            RunEvent::ExitRequested { .. } | RunEvent::Exit => stop_backend(app),
            _ => {}
        });
}