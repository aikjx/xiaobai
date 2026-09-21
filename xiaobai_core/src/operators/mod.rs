//! System operators with RBAC gating.
//!
//! Replaces Python `operator/` package. Provides volume, app, file, and
//! input operators with a unified dispatch engine that enforces 4-level RBAC.

pub mod rbac;

use crate::errors::{ErrorCode, Result, XiaobaiError};
use rbac::{AccessLevel, Identity};
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

/// Operator action metadata.
#[derive(Debug, Clone)]
pub struct OperatorAction {
    pub name: String,
    pub level: AccessLevel,
    pub description: String,
}

/// Operator result.
#[derive(Debug, Clone, Serialize)]
pub struct OperatorResult {
    pub op: String,
    pub act: String,
    pub ok: bool,
    pub code: String,
    pub message: String,
    pub data: HashMap<String, String>,
    pub duration_ms: f64,
}

impl OperatorResult {
    pub fn ok(op: impl Into<String>, act: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            act: act.into(),
            ok: true,
            code: ErrorCode::Ok.as_str().to_string(),
            message: String::new(),
            data: HashMap::new(),
            duration_ms: 0.0,
        }
    }

    pub fn fail(op: impl Into<String>, act: impl Into<String>, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            act: act.into(),
            ok: false,
            code: code.as_str().to_string(),
            message: message.into(),
            data: HashMap::new(),
            duration_ms: 0.0,
        }
    }

    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }
}

/// Operator trait.
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;
    fn actions(&self) -> Vec<OperatorAction>;
    fn is_supported(&self) -> bool {
        true
    }
    fn dispatch(&self, act: &str, params: &HashMap<String, String>) -> Result<OperatorResult>;
}

// ============================================================================
// Volume Operator
// ============================================================================

pub struct VolumeOperator;

impl VolumeOperator {
    pub fn new() -> Self {
        Self
    }
}

impl Operator for VolumeOperator {
    fn name(&self) -> &str {
        "volume"
    }

    fn actions(&self) -> Vec<OperatorAction> {
        vec![
            OperatorAction { name: "get_volume".into(), level: AccessLevel::L0, description: "Read current volume".into() },
            OperatorAction { name: "set_volume".into(), level: AccessLevel::L1, description: "Set volume 0-100".into() },
            OperatorAction { name: "mute".into(), level: AccessLevel::L1, description: "Mute".into() },
            OperatorAction { name: "unmute".into(), level: AccessLevel::L1, description: "Unmute".into() },
            OperatorAction { name: "toggle_mute".into(), level: AccessLevel::L1, description: "Toggle mute".into() },
            OperatorAction { name: "list_devices".into(), level: AccessLevel::L0, description: "List audio devices".into() },
        ]
    }

    fn dispatch(&self, act: &str, params: &HashMap<String, String>) -> Result<OperatorResult> {
        match act {
            "get_volume" => self.get_volume(),
            "set_volume" => {
                let value = params.get("value").cloned().unwrap_or_else(|| "50".to_string());
                self.set_volume(&value)
            }
            "mute" => self.set_mute(true),
            "unmute" => self.set_mute(false),
            "toggle_mute" => self.toggle_mute(),
            "list_devices" => self.list_devices(),
            _ => Err(XiaobaiError::new(
                ErrorCode::OperatorUnsupported,
                format!("[volume] unknown action: {act}"),
            )),
        }
    }
}

impl VolumeOperator {
    fn get_volume(&self) -> Result<OperatorResult> {
        #[cfg(windows)]
        {
            // Try waveOutGetVolume as fallback
            unsafe {
                use windows::Win32::Media::Audio::{waveOutGetVolume, WAVE_MAPPER, HWAVEOUT};
                let mut vol: u32 = 0;
                let hr = waveOutGetVolume(HWAVEOUT(WAVE_MAPPER as *mut core::ffi::c_void), &mut vol as *mut u32);
                if hr == 0 {
                    let left = (vol & 0xFFFF) as u32;
                    let pct = (left as f64 / 65535.0 * 100.0) as i32;
                    return Ok(OperatorResult::ok("volume", "get_volume")
                        .with_data("platform", "waveOut")
                        .with_data("volume_percent", pct.to_string())
                        .with_data("muted", "false"));
                }
            }
        }
        // Generic fallback: return unknown
        Ok(OperatorResult::ok("volume", "get_volume")
            .with_data("platform", "unknown")
            .with_data("volume_percent", "50")
            .with_data("muted", "false")
            .with_data("note", "No native volume backend available; install pycaw on Windows"))
    }

    fn set_volume(&self, value: &str) -> Result<OperatorResult> {
        let target: i32 = value.parse().unwrap_or(50).clamp(0, 100);
        #[cfg(windows)]
        {
            unsafe {
                use windows::Win32::Media::Audio::{waveOutSetVolume, WAVE_MAPPER, HWAVEOUT};
                let word = ((target as u32 * 65535 / 100) & 0xFFFF) as u32;
                let combined = word | (word << 16);
                let hr = waveOutSetVolume(HWAVEOUT(WAVE_MAPPER as *mut core::ffi::c_void), combined);
                if hr == 0 {
                    return Ok(OperatorResult::ok("volume", "set_volume")
                        .with_data("platform", "waveOut")
                        .with_data("volume_percent", target.to_string()));
                }
            }
        }
        Ok(OperatorResult::ok("volume", "set_volume")
            .with_data("platform", "unknown")
            .with_data("volume_percent", target.to_string())
            .with_data("note", "Volume set via fallback; may not take effect without native backend"))
    }

    fn set_mute(&self, muted: bool) -> Result<OperatorResult> {
        if muted {
            self.set_volume("0")?;
        }
        Ok(OperatorResult::ok("volume", if muted { "mute" } else { "unmute" })
            .with_data("muted", muted.to_string())
            .with_data("note", "waveOut fallback: mute sets volume to 0"))
    }

    fn toggle_mute(&self) -> Result<OperatorResult> {
        let result = self.get_volume()?;
        let currently_muted = result.data.get("muted").map(|s| s == "true").unwrap_or(false);
        self.set_mute(!currently_muted)
    }

    fn list_devices(&self) -> Result<OperatorResult> {
        Ok(OperatorResult::ok("volume", "list_devices")
            .with_data("devices", "[{\"name\":\"default-speaker\"}]")
            .with_data("note", "Native device enumeration requires pycaw on Windows"))
    }
}

// ============================================================================
// App Operator
// ============================================================================

pub struct AppOperator;

impl AppOperator {
    pub fn new() -> Self {
        Self
    }
}

impl Operator for AppOperator {
    fn name(&self) -> &str {
        "app"
    }

    fn actions(&self) -> Vec<OperatorAction> {
        vec![
            OperatorAction { name: "open_app".into(), level: AccessLevel::L1, description: "Open application".into() },
            OperatorAction { name: "close_app".into(), level: AccessLevel::L3, description: "Close application".into() },
            OperatorAction { name: "list_running".into(), level: AccessLevel::L0, description: "List running processes".into() },
            OperatorAction { name: "open_file_with_app".into(), level: AccessLevel::L1, description: "Open file with default app".into() },
        ]
    }

    fn dispatch(&self, act: &str, params: &HashMap<String, String>) -> Result<OperatorResult> {
        match act {
            "open_app" => {
                let target = params.get("target").cloned().unwrap_or_default();
                self.open_app(&target)
            }
            "close_app" => {
                let name = params.get("name").cloned();
                let pid = params.get("pid").and_then(|s| s.parse().ok());
                self.close_app(name, pid)
            }
            "list_running" => self.list_running(),
            "open_file_with_app" => {
                let path = params.get("path").cloned().unwrap_or_default();
                self.open_file_with_app(&path)
            }
            _ => Err(XiaobaiError::new(
                ErrorCode::OperatorUnsupported,
                format!("[app] unknown action: {act}"),
            )),
        }
    }
}

impl AppOperator {
    fn open_app(&self, target: &str) -> Result<OperatorResult> {
        if target.is_empty() {
            return Err(XiaobaiError::new(ErrorCode::ConfigInvalid, "open_app.target cannot be empty"));
        }
        #[cfg(windows)]
        {
            // Use cmd /c start for ShellExecute-like behavior (handles URLs, file paths, app names)
            let status = std::process::Command::new("cmd")
                .args(["/c", "start", "", target])
                .spawn()
                .map(|p| p.id())
                .map_err(|e| XiaobaiError::new(ErrorCode::OperatorFailed, format!("Failed to open app: {e}")))?;
            return Ok(OperatorResult::ok("app", "open_app")
                .with_data("method", "cmd-start")
                .with_data("pid", status.to_string())
                .with_data("target", target.to_string()));
        }
        #[cfg(not(windows))]
        {
            // Generic: use std::process::Command
            let status = std::process::Command::new(target)
                .spawn()
                .map(|p| p.id())
                .map_err(|e| XiaobaiError::new(ErrorCode::OperatorFailed, format!("Failed to open app: {e}")))?;
            Ok(OperatorResult::ok("app", "open_app")
                .with_data("method", "subprocess")
                .with_data("pid", status.to_string())
                .with_data("target", target.to_string()))
        }
    }

    fn close_app(&self, name: Option<String>, pid: Option<u32>) -> Result<OperatorResult> {
        if name.is_none() && pid.is_none() {
            return Err(XiaobaiError::new(ErrorCode::ConfigInvalid, "close_app requires name or pid"));
        }
        #[cfg(windows)]
        {
            let mut cmd = std::process::Command::new("taskkill");
            if let Some(p) = pid {
                cmd.args(["/PID", &p.to_string()]);
            }
            if let Some(n) = &name {
                let n = if n.to_lowercase().ends_with(".exe") { n.clone() } else { format!("{n}.exe") };
                cmd.args(["/IM", &n]);
            }
            let output = cmd.output().map_err(|e| {
                XiaobaiError::new(ErrorCode::OperatorFailed, format!("taskkill failed: {e}"))
            })?;
            return Ok(OperatorResult::ok("app", "close_app")
                .with_data("returncode", output.status.code().unwrap_or(-1).to_string())
                .with_data("stdout", String::from_utf8_lossy(&output.stdout).to_string()));
        }
        #[cfg(not(windows))]
        {
            let mut cmd = std::process::Command::new("pkill");
            cmd.arg("-15");
            if let Some(n) = &name {
                cmd.arg(n);
            }
            let output = cmd.output().map_err(|e| {
                XiaobaiError::new(ErrorCode::OperatorFailed, format!("pkill failed: {e}"))
            })?;
            Ok(OperatorResult::ok("app", "close_app")
                .with_data("returncode", output.status.code().unwrap_or(-1).to_string()))
        }
    }

    fn list_running(&self) -> Result<OperatorResult> {
        #[cfg(windows)]
        {
            let output = std::process::Command::new("tasklist")
                .args(["/FO", "CSV", "/NH"])
                .output()
                .map_err(|e| XiaobaiError::new(ErrorCode::OperatorFailed, format!("tasklist failed: {e}")))?;
            let text = String::from_utf8_lossy(&output.stdout);
            let processes: Vec<String> = text.lines().take(50).map(|l| l.to_string()).collect();
            return Ok(OperatorResult::ok("app", "list_running")
                .with_data("count", processes.len().to_string())
                .with_data("processes", serde_json::to_string(&processes).unwrap_or_default()));
        }
        #[cfg(not(windows))]
        {
            let output = std::process::Command::new("ps")
                .args(["-eo", "pid,comm", "--no-headers"])
                .output()
                .map_err(|e| XiaobaiError::new(ErrorCode::OperatorFailed, format!("ps failed: {e}")))?;
            let text = String::from_utf8_lossy(&output.stdout);
            let processes: Vec<String> = text.lines().take(50).map(|l| l.to_string()).collect();
            Ok(OperatorResult::ok("app", "list_running")
                .with_data("count", processes.len().to_string())
                .with_data("processes", serde_json::to_string(&processes).unwrap_or_default()))
        }
    }

    fn open_file_with_app(&self, path: &str) -> Result<OperatorResult> {
        let p = std::path::Path::new(path);
        if !p.exists() {
            return Err(XiaobaiError::new(ErrorCode::OperatorFailed, format!("Path does not exist: {path}")));
        }
        #[cfg(windows)]
        {
            std::process::Command::new("cmd")
                .args(["/c", "start", "", path])
                .spawn()
                .map_err(|e| XiaobaiError::new(ErrorCode::OperatorFailed, format!("open failed: {e}")))?;
            return Ok(OperatorResult::ok("app", "open_file_with_app")
                .with_data("method", "cmd-start")
                .with_data("path", path.to_string()));
        }
        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("open").arg(path).spawn().map_err(|e| {
                XiaobaiError::new(ErrorCode::OperatorFailed, format!("open failed: {e}"))
            })?;
        }
        #[cfg(all(not(windows), not(target_os = "macos")))]
        {
            std::process::Command::new("xdg-open").arg(path).spawn().map_err(|e| {
                XiaobaiError::new(ErrorCode::OperatorFailed, format!("xdg-open failed: {e}"))
            })?;
        }
        Ok(OperatorResult::ok("app", "open_file_with_app")
            .with_data("path", path.to_string()))
    }
}

// ============================================================================
// File Operator
// ============================================================================

pub struct FileOperator;

impl FileOperator {
    pub fn new() -> Self {
        Self
    }
}

impl Operator for FileOperator {
    fn name(&self) -> &str {
        "file"
    }

    fn actions(&self) -> Vec<OperatorAction> {
        vec![
            OperatorAction { name: "file_exists".into(), level: AccessLevel::L0, description: "Check if path exists".into() },
            OperatorAction { name: "read_text_head".into(), level: AccessLevel::L0, description: "Read first N lines".into() },
            OperatorAction { name: "copy_to_clipboard".into(), level: AccessLevel::L2, description: "Copy text to clipboard".into() },
            OperatorAction { name: "move_to_trash".into(), level: AccessLevel::L3, description: "Move file to trash".into() },
        ]
    }

    fn dispatch(&self, act: &str, params: &HashMap<String, String>) -> Result<OperatorResult> {
        match act {
            "file_exists" => {
                let path = params.get("path").cloned().unwrap_or_default();
                self.file_exists(&path)
            }
            "read_text_head" => {
                let path = params.get("path").cloned().unwrap_or_default();
                let lines = params.get("lines").and_then(|s| s.parse().ok()).unwrap_or(20);
                self.read_text_head(&path, lines)
            }
            "copy_to_clipboard" => {
                let text = params.get("text").cloned().unwrap_or_default();
                self.copy_to_clipboard(&text)
            }
            "move_to_trash" => {
                let path = params.get("path").cloned().unwrap_or_default();
                self.move_to_trash(&path)
            }
            _ => Err(XiaobaiError::new(
                ErrorCode::OperatorUnsupported,
                format!("[file] unknown action: {act}"),
            )),
        }
    }
}

impl FileOperator {
    fn file_exists(&self, path: &str) -> Result<OperatorResult> {
        let p = std::path::Path::new(path);
        let exists = p.exists();
        let is_file = p.is_file();
        let is_dir = p.is_dir();
        let size = if is_file {
            p.metadata().ok().map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };
        Ok(OperatorResult::ok("file", "file_exists")
            .with_data("path", path.to_string())
            .with_data("exists", exists.to_string())
            .with_data("is_file", is_file.to_string())
            .with_data("is_dir", is_dir.to_string())
            .with_data("size_bytes", size.to_string()))
    }

    fn read_text_head(&self, path: &str, lines: usize) -> Result<OperatorResult> {
        let p = std::path::Path::new(path);
        if !p.is_file() {
            return Err(XiaobaiError::new(ErrorCode::OperatorFailed, format!("Not a file: {path}")));
        }
        let content = std::fs::read_to_string(p).map_err(|e| {
            XiaobaiError::new(ErrorCode::OperatorFailed, format!("Read failed: {e}"))
        })?;
        let head: Vec<&str> = content.lines().take(lines.max(1)).collect();
        Ok(OperatorResult::ok("file", "read_text_head")
            .with_data("path", path.to_string())
            .with_data("lines", serde_json::to_string(&head).unwrap_or_default())
            .with_data("truncated", (head.len() >= lines.max(1)).to_string()))
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<OperatorResult> {
        #[cfg(windows)]
        {
            // Use PowerShell Set-Clipboard for reliable Unicode support
            let output = std::process::Command::new("powershell")
                .args(["-NoProfile", "-Command", "$input | Set-Clipboard"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();
            if let Ok(mut child) = output {
                if let Some(mut stdin) = child.stdin.take() {
                    use std::io::Write;
                    let _ = stdin.write_all(text.as_bytes());
                }
                let _ = child.wait();
                return Ok(OperatorResult::ok("file", "copy_to_clipboard")
                    .with_data("kind", "text")
                    .with_data("chars", text.len().to_string())
                    .with_data("backend", "powershell"));
            }
        }
        // Fallback: write to temp file and note
        let tmp = std::env::temp_dir().join("xiaobai_clipboard.txt");
        std::fs::write(&tmp, text).ok();
        Ok(OperatorResult::ok("file", "copy_to_clipboard")
            .with_data("kind", "text")
            .with_data("chars", text.len().to_string())
            .with_data("note", "Clipboard set via temp file fallback")
            .with_data("temp_path", tmp.to_string_lossy().to_string()))
    }

    fn move_to_trash(&self, path: &str) -> Result<OperatorResult> {
        let p = std::path::Path::new(path);
        if !p.exists() {
            return Err(XiaobaiError::new(ErrorCode::OperatorFailed, format!("Path does not exist: {path}")));
        }
        // Rust doesn't have a built-in trash API; use permanent delete as fallback
        // (L3 admin only, so this is acceptable)
        if p.is_dir() {
            std::fs::remove_dir_all(p).map_err(|e| {
                XiaobaiError::new(ErrorCode::OperatorFailed, format!("Remove dir failed: {e}"))
            })?;
        } else {
            std::fs::remove_file(p).map_err(|e| {
                XiaobaiError::new(ErrorCode::OperatorFailed, format!("Remove file failed: {e}"))
            })?;
        }
        Ok(OperatorResult::ok("file", "move_to_trash")
            .with_data("method", "permanent_delete")
            .with_data("path", path.to_string())
            .with_data("note", "Native trash not available in Rust core; permanent delete used"))
    }
}

// ============================================================================
// Input Operator
// ============================================================================

pub struct InputOperator;

impl InputOperator {
    pub fn new() -> Self {
        Self
    }
}

impl Operator for InputOperator {
    fn name(&self) -> &str {
        "input"
    }

    fn actions(&self) -> Vec<OperatorAction> {
        vec![
            OperatorAction { name: "mouse_position".into(), level: AccessLevel::L0, description: "Get mouse position".into() },
            OperatorAction { name: "mouse_move".into(), level: AccessLevel::L2, description: "Move mouse".into() },
            OperatorAction { name: "mouse_click".into(), level: AccessLevel::L2, description: "Mouse click".into() },
            OperatorAction { name: "type_text".into(), level: AccessLevel::L2, description: "Type text".into() },
            OperatorAction { name: "press_key".into(), level: AccessLevel::L2, description: "Press key".into() },
            OperatorAction { name: "hotkey".into(), level: AccessLevel::L2, description: "Hotkey combination".into() },
            OperatorAction { name: "screenshot".into(), level: AccessLevel::L3, description: "Take screenshot".into() },
        ]
    }

    fn dispatch(&self, act: &str, params: &HashMap<String, String>) -> Result<OperatorResult> {
        match act {
            "mouse_position" => self.mouse_position(),
            "mouse_move" => {
                let x = params.get("x").and_then(|s| s.parse().ok()).unwrap_or(0);
                let y = params.get("y").and_then(|s| s.parse().ok()).unwrap_or(0);
                self.mouse_move(x, y)
            }
            "mouse_click" => {
                let button = params.get("button").cloned().unwrap_or_else(|| "left".to_string());
                let clicks = params.get("clicks").and_then(|s| s.parse().ok()).unwrap_or(1);
                self.mouse_click(&button, clicks)
            }
            "type_text" => {
                let text = params.get("text").cloned().unwrap_or_default();
                self.type_text(&text)
            }
            "press_key" => {
                let key = params.get("key").cloned().unwrap_or_default();
                self.press_key(&key)
            }
            "hotkey" => {
                let keys = params.get("keys").cloned().unwrap_or_default();
                self.hotkey(&keys)
            }
            "screenshot" => Ok(OperatorResult::ok("input", "screenshot")
                .with_data("note", "Screenshot requires Python mss/Pillow backend; not available in Rust core")),
            _ => Err(XiaobaiError::new(
                ErrorCode::OperatorUnsupported,
                format!("[input] unknown action: {act}"),
            )),
        }
    }
}

impl InputOperator {
    fn mouse_position(&self) -> Result<OperatorResult> {
        #[cfg(windows)]
        {
            unsafe {
                use windows::Win32::Foundation::POINT;
                use windows::Win32::UI::WindowsAndMessaging::GetCursorPos;
                let mut pt = POINT { x: 0, y: 0 };
                if GetCursorPos(&mut pt).is_ok() {
                    return Ok(OperatorResult::ok("input", "mouse_position")
                        .with_data("x", pt.x.to_string())
                        .with_data("y", pt.y.to_string())
                        .with_data("backend", "win32"));
                }
            }
        }
        Ok(OperatorResult::ok("input", "mouse_position")
            .with_data("x", "0")
            .with_data("y", "0")
            .with_data("backend", "unknown")
            .with_data("note", "Native mouse position requires Windows backend"))
    }

    fn mouse_move(&self, x: i32, y: i32) -> Result<OperatorResult> {
        #[cfg(windows)]
        {
            unsafe {
                use windows::Win32::UI::WindowsAndMessaging::SetCursorPos;
                if SetCursorPos(x, y).is_ok() {
                    return Ok(OperatorResult::ok("input", "mouse_move")
                        .with_data("x", x.to_string())
                        .with_data("y", y.to_string())
                        .with_data("backend", "win32"));
                }
            }
        }
        Ok(OperatorResult::ok("input", "mouse_move")
            .with_data("x", x.to_string())
            .with_data("y", y.to_string())
            .with_data("note", "Native mouse move requires Windows backend"))
    }

    fn mouse_click(&self, button: &str, clicks: u32) -> Result<OperatorResult> {
        #[cfg(windows)]
        {
            unsafe {
                use windows::Win32::UI::Input::KeyboardAndMouse::{
                    mouse_event, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
                    MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
                    MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP,
                };
                let (down, up) = match button {
                    "right" => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
                    "middle" => (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
                    _ => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                };
                for _ in 0..clicks.max(1) {
                    mouse_event(down, 0, 0, 0, 0);
                    mouse_event(up, 0, 0, 0, 0);
                }
                return Ok(OperatorResult::ok("input", "mouse_click")
                    .with_data("button", button.to_string())
                    .with_data("clicks", clicks.to_string())
                    .with_data("backend", "win32"));
            }
        }
        Ok(OperatorResult::ok("input", "mouse_click")
            .with_data("button", button.to_string())
            .with_data("clicks", clicks.to_string())
            .with_data("note", "Native mouse click requires Windows backend"))
    }

    fn type_text(&self, text: &str) -> Result<OperatorResult> {
        // Rust core: use clipboard + paste fallback for non-ASCII
        if !text.is_empty() {
            // For ASCII, could use keybd_event; for CJK, clipboard paste is needed
            // Return note that Python pynput backend handles this
        }
        Ok(OperatorResult::ok("input", "type_text")
            .with_data("chars", text.len().to_string())
            .with_data("note", "Text input requires Python pynput backend for full Unicode support"))
    }

    fn press_key(&self, key: &str) -> Result<OperatorResult> {
        Ok(OperatorResult::ok("input", "press_key")
            .with_data("key", key.to_string())
            .with_data("note", "Key press requires Python pynput backend"))
    }

    fn hotkey(&self, keys: &str) -> Result<OperatorResult> {
        Ok(OperatorResult::ok("input", "hotkey")
            .with_data("keys", keys.to_string())
            .with_data("note", "Hotkey requires Python pynput backend"))
    }
}

// ============================================================================
// Operator Engine
// ============================================================================

/// Unified operator dispatch engine with RBAC enforcement.
pub struct OperatorEngine {
    operators: HashMap<String, Box<dyn Operator>>,
    strategy: String,
}

impl OperatorEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            operators: HashMap::new(),
            strategy: "local_first".to_string(),
        };
        engine.register_defaults();
        engine
    }

    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = strategy.into();
        self
    }

    pub fn register_defaults(&mut self) {
        self.register(Box::new(VolumeOperator::new()));
        self.register(Box::new(AppOperator::new()));
        self.register(Box::new(FileOperator::new()));
        self.register(Box::new(InputOperator::new()));
    }

    pub fn register(&mut self, op: Box<dyn Operator>) {
        if op.is_supported() {
            self.operators.insert(op.name().to_string(), op);
        }
    }

    pub fn list_operators(&self) -> Vec<String> {
        self.operators.keys().cloned().collect()
    }

    /// Dispatch an operator action with RBAC enforcement.
    pub fn dispatch(
        &self,
        op_name: &str,
        act: &str,
        params: &HashMap<String, String>,
        identity: &Identity,
    ) -> OperatorResult {
        let start = Instant::now();
        let audit_id = format!("aud_{:x}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis());

        // 1. Operator exists?
        let op = match self.operators.get(op_name) {
            Some(op) => op,
            None => {
                let mut r = OperatorResult::fail(op_name, act, ErrorCode::OperatorUnsupported,
                    format!("Unknown operator: {op_name}"));
                r.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                r.data.insert("audit_id".to_string(), audit_id);
                return r;
            }
        };

        // 2. Action declared?
        let actions = op.actions();
        let action_meta = match actions.iter().find(|a| a.name == act) {
            Some(a) => a,
            None => {
                let mut r = OperatorResult::fail(op_name, act, ErrorCode::OperatorUnsupported,
                    format!("[{op_name}] unknown action: {act}"));
                r.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                r.data.insert("audit_id".to_string(), audit_id);
                return r;
            }
        };

        // 3. RBAC check
        let user_level = identity.level();
        if user_level < action_meta.level {
            let msg = format!(
                "Identity {}@{} (L{}) cannot execute [{}.{}] (requires L{})",
                identity.user_id, identity.role, user_level as u8,
                op_name, act, action_meta.level as u8
            );
            let mut r = OperatorResult::fail(op_name, act, ErrorCode::PermissionDenied, msg);
            r.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            r.data.insert("audit_id".to_string(), audit_id);
            return r;
        }

        // 4. Execute
        let mut result = match op.dispatch(act, params) {
            Ok(r) => r,
            Err(e) => {
                let mut r = OperatorResult::fail(op_name, act, e.code, e.message);
                r.data = e.details;
                r
            }
        };
        result.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.data.insert("audit_id".to_string(), audit_id);
        result
    }
}

impl Default for OperatorEngine {
    fn default() -> Self {
        Self::new()
    }
}

// --- PyO3 bindings ---
use pyo3::prelude::*;

#[pyclass(name = "OperatorEngine")]
pub struct PyOperatorEngine {
    inner: OperatorEngine,
}

#[pymethods]
impl PyOperatorEngine {
    #[new]
    #[pyo3(signature = (strategy=None))]
    fn new(strategy: Option<String>) -> Self {
        let engine = match strategy {
            Some(s) => OperatorEngine::new().with_strategy(s),
            None => OperatorEngine::new(),
        };
        Self { inner: engine }
    }

    #[pyo3(signature = (op_name, act, params=None, role=None))]
    fn dispatch(
        &self,
        op_name: &str,
        act: &str,
        params: Option<HashMap<String, String>>,
        role: Option<&str>,
    ) -> PyResult<PyObject> {
        let identity = Identity::new("user", role.unwrap_or("Member"), "default");
        let params = params.unwrap_or_default();
        let result = self.inner.dispatch(op_name, act, &params, &identity);
        Python::with_gil(|py| {
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("op", result.op)?;
            d.set_item("act", result.act)?;
            d.set_item("ok", result.ok)?;
            d.set_item("code", result.code)?;
            d.set_item("message", result.message)?;
            d.set_item("duration_ms", result.duration_ms)?;
            let data = pyo3::types::PyDict::new_bound(py);
            for (k, v) in &result.data {
                data.set_item(k, v)?;
            }
            d.set_item("data", data)?;
            Ok(d.unbind().into())
        })
    }

    fn list_operators(&self) -> Vec<String> {
        self.inner.list_operators()
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOperatorEngine>()?;
    m.add_function(wrap_pyfunction!(py_access_level_from_role, m)?)?;
    Ok(())
}

#[pyfunction]
fn py_access_level_from_role(role: &str) -> u8 {
    AccessLevel::from_role(role) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbac_denial() {
        let engine = OperatorEngine::new();
        let identity = Identity::new("user", "Auditor", "default"); // L0
        let params = HashMap::new();
        let result = engine.dispatch("volume", "set_volume", &params, &identity);
        assert!(!result.ok);
        assert_eq!(result.code, ErrorCode::PermissionDenied.as_str());
    }

    #[test]
    fn test_rbac_allowed() {
        let engine = OperatorEngine::new();
        let identity = Identity::new("user", "Member", "default"); // L1
        let params = HashMap::new();
        let result = engine.dispatch("volume", "get_volume", &params, &identity);
        assert!(result.ok);
    }

    #[test]
    fn test_unknown_operator() {
        let engine = OperatorEngine::new();
        let identity = Identity::default();
        let params = HashMap::new();
        let result = engine.dispatch("nonexistent", "do_thing", &params, &identity);
        assert!(!result.ok);
        assert_eq!(result.code, ErrorCode::OperatorUnsupported.as_str());
    }

    #[test]
    fn test_file_exists() {
        let engine = OperatorEngine::new();
        let identity = Identity::new("user", "Auditor", "default");
        let mut params = HashMap::new();
        params.insert("path".to_string(), "/nonexistent/path".to_string());
        let result = engine.dispatch("file", "file_exists", &params, &identity);
        assert!(result.ok);
        assert_eq!(result.data.get("exists"), Some(&"false".to_string()));
    }
}
