//! Rule-based intent router (S1 minimal delivery).
//!
//! Replaces Python `intent/router.py`. Uses regex patterns with named
//! groups, parameter transforms, and RBAC access level gating.

use crate::errors::Result;
use crate::operators::rbac::AccessLevel;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A routing rule: regex pattern → operator action.
#[derive(Clone)]
pub struct Rule {
    pub name: String,
    pub op: String,
    pub act: String,
    pub pattern: Regex,
    pub base_confidence: f32,
    pub required_level: AccessLevel,
}

impl Rule {
    pub fn new(
        name: impl Into<String>,
        op: impl Into<String>,
        act: impl Into<String>,
        pattern: &str,
        base_confidence: f32,
        required_level: AccessLevel,
    ) -> Result<Self> {
        let name_str = name.into();
        let re = Regex::new(pattern).map_err(|e| {
            crate::errors::XiaobaiError::new(
                crate::errors::ErrorCode::ConfigInvalid,
                format!("Invalid regex for rule {}: {}", name_str, e),
            )
        })?;
        Ok(Self {
            name: name_str,
            op: op.into(),
            act: act.into(),
            pattern: re,
            base_confidence,
            required_level,
        })
    }
}

/// Result of intent routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResult {
    pub op_name: String,
    pub act: String,
    pub params: HashMap<String, String>,
    pub confidence: f32,
    pub ambiguous: bool,
    pub candidates: Vec<Candidate>,
    pub matched_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub op: String,
    pub act: String,
    pub params: HashMap<String, String>,
    pub confidence: f32,
    pub rule: String,
    pub required_level: u8,
}

impl RouteResult {
    pub fn empty() -> Self {
        Self {
            op_name: String::new(),
            act: String::new(),
            params: HashMap::new(),
            confidence: 0.0,
            ambiguous: false,
            candidates: Vec::new(),
            matched_rule: String::new(),
        }
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("op".to_string(), self.op_name.clone());
        m.insert("act".to_string(), self.act.clone());
        m.insert("confidence".to_string(), format!("{:.4}", self.confidence));
        m.insert("ambiguous".to_string(), self.ambiguous.to_string());
        m.insert("matched_rule".to_string(), self.matched_rule.clone());
        m
    }
}

/// Identity for RBAC checks.
#[derive(Debug, Clone)]
pub struct Identity {
    pub user_id: String,
    pub role: String,
    pub tenant_id: String,
}

impl Identity {
    pub fn new(user_id: impl Into<String>, role: impl Into<String>, tenant_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            role: role.into(),
            tenant_id: tenant_id.into(),
        }
    }

    pub fn level(&self) -> AccessLevel {
        AccessLevel::from_role(&self.role)
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new("anon", "Auditor", "default")
    }
}

/// The intent router.
pub struct IntentRouter {
    rules: Vec<Rule>,
    ambiguous_threshold: f32,
}

impl IntentRouter {
    pub fn new() -> Self {
        Self {
            rules: build_default_rules().expect("default rules should compile"),
            ambiguous_threshold: 0.1,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.ambiguous_threshold = threshold;
        self
    }

    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Route text to an operator action.
    pub fn route(&self, text: &str, identity: Option<&Identity>) -> RouteResult {
        let text = text.trim();
        if text.is_empty() {
            return RouteResult::empty();
        }
        let ident = identity.cloned().unwrap_or_default();
        let mut candidates: Vec<Candidate> = Vec::new();

        for rule in &self.rules {
            if let Some(m) = rule.pattern.captures(text) {
                let mut params = HashMap::new();
                // Extract named groups
                for name in rule.pattern.capture_names().flatten() {
                    if let Some(val) = m.name(name) {
                        params.insert(name.to_string(), val.as_str().to_string());
                    }
                }

                let mut conf = rule.base_confidence;
                // RBAC: lower confidence if identity level < required level
                if ident.level() < rule.required_level {
                    conf *= 0.6;
                }
                // Match length bonus
                let span_len = (m.get(0).map(|m| m.end() - m.start()).unwrap_or(1)).max(1) as f32;
                conf = (conf + (0.005 * span_len).min(0.06)).min(1.0);

                candidates.push(Candidate {
                    op: rule.op.clone(),
                    act: rule.act.clone(),
                    params,
                    confidence: (conf * 10000.0).round() / 10000.0,
                    rule: rule.name.clone(),
                    required_level: rule.required_level as u8,
                });
            }
        }

        if candidates.is_empty() {
            return RouteResult::empty();
        }

        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        let top = &candidates[0];
        let ambiguous = candidates.len() >= 2
            && (candidates[0].confidence - candidates[1].confidence) < self.ambiguous_threshold;

        RouteResult {
            op_name: top.op.clone(),
            act: top.act.clone(),
            params: top.params.clone(),
            confidence: top.confidence,
            ambiguous,
            candidates: candidates.iter().take(5).cloned().collect(),
            matched_rule: top.rule.clone(),
        }
    }

    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
}

impl Default for IntentRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the default rule set (matching Python router).
fn build_default_rules() -> Result<Vec<Rule>> {
    let mut rules = Vec::new();

    // Volume (L0/L1)
    rules.push(Rule::new("vol.get", "volume", "get_volume",
        r"(当前|现在|系统)音量|音量.*多大|多大声", 0.95, AccessLevel::L0)?);
    rules.push(Rule::new("vol.set_pct", "volume", "set_volume",
        r"把?音量(调(到|成)?|开(到|成)?|设(到|成)?)?\s*(?P<value>[0-9]{1,3})\s*(%|个|格)?|声音(?P<value2>[0-9]{1,3})",
        0.92, AccessLevel::L1)?);
    rules.push(Rule::new("vol.relative", "volume", "set_volume",
        r"音量(加|大|提高|升|往上|调高)(?P<plus>[0-9]{1,2})|音量(减|小|降|调低|往下)(?P<minus>[0-9]{1,2})",
        0.92, AccessLevel::L1)?);
    rules.push(Rule::new("vol.mute", "volume", "mute",
        r"静音(开启|打开|一下)?|别出声|禁声|闭嘴", 0.92, AccessLevel::L1)?);
    rules.push(Rule::new("vol.unmute", "volume", "unmute",
        r"(取消|解除|去掉)静音|(开|恢复)声音|出声", 0.92, AccessLevel::L1)?);
    rules.push(Rule::new("vol.toggle", "volume", "toggle_mute",
        r"切换静音|切静音", 0.92, AccessLevel::L1)?);

    // App (L1/L3)
    rules.push(Rule::new("app.open", "app", "open_app",
        r"(打开|启动|运行|开一下|点开)\s*(?P<target>[\u4e00-\u9fa5A-Za-z0-9_.\-·：:（）()/\\]+)",
        0.88, AccessLevel::L1)?);
    rules.push(Rule::new("app.close", "app", "close_app",
        r"(关闭|关掉|结束|停止|杀)\s*(进程|应用)?\s*(?P<name>[\u4e00-\u9fa5A-Za-z0-9_.\-·（）()]+)",
        0.86, AccessLevel::L3)?);
    rules.push(Rule::new("app.list", "app", "list_running",
        r"(列|查看|看一下|看看|列出).*进程|任务(列表|管理器)|开了什么",
        0.9, AccessLevel::L0)?);
    rules.push(Rule::new("app.open_file", "app", "open_file_with_app",
        r"(打开|浏览|查看).*(文件|目录|文件夹|C:|D:|E:|/Users|/home|/tmp|桌面|文档|下载)",
        0.85, AccessLevel::L1)?);

    // File (L0/L1/L2/L3)
    rules.push(Rule::new("file.copy_txt", "file", "copy_to_clipboard",
        r"(把|将|帮我)\s*(?P<text1>.*?)\s*(复制|拷|粘)到(剪贴板|剪切板)|复制(?P<text2>.*)",
        0.82, AccessLevel::L2)?);
    rules.push(Rule::new("file.delete", "file", "move_to_trash",
        r"(删除|删掉|清掉|移除)\s*(文件|目录)?\s*(?P<path>\S+)",
        0.90, AccessLevel::L3)?);
    rules.push(Rule::new("file.read", "file", "read_text_head",
        r"(读|看|查看|预览).*(文件)?\s*(?P<path>\S+)",
        0.78, AccessLevel::L0)?);
    rules.push(Rule::new("file.exists", "file", "file_exists",
        r"路径?\s*(?P<path>\S+)\s*存在吗|有没有文件(?P<path2>\S+)",
        0.92, AccessLevel::L0)?);

    // Input (L2/L3)
    rules.push(Rule::new("inp.type", "input", "type_text",
        r"(输入|键入|打字|写|敲入)\s*(：|:)?\s*(?P<text>.*)",
        0.80, AccessLevel::L2)?);
    rules.push(Rule::new("inp.click", "input", "mouse_click",
        r"(单击|点击|点一下|点)\s*(?P<button>左键|右键|中键)?",
        0.88, AccessLevel::L2)?);
    rules.push(Rule::new("inp.dblclick", "input", "mouse_click",
        r"双击|点两下", 0.92, AccessLevel::L2)?);
    rules.push(Rule::new("inp.move", "input", "mouse_move",
        r"鼠标(移动|移到|挪到|去)\s*\(?(?P<x>\d{1,5})\s*[,， ]\s*(?P<y>\d{1,5})\)?",
        0.95, AccessLevel::L2)?);
    rules.push(Rule::new("inp.pos", "input", "mouse_position",
        r"鼠标在哪|鼠标坐标|鼠标位置", 0.95, AccessLevel::L0)?);
    rules.push(Rule::new("inp.key", "input", "press_key",
        r"按(一下|下)?键?\s*(?P<key>[A-Za-z0-9]+|Enter|回车|Space|空格|Esc|Escape|Tab|Backspace|Delete|Insert|Home|End|PgUp|PgDn|Up|Down|Left|Right)",
        0.92, AccessLevel::L2)?);
    rules.push(Rule::new("inp.hotkey", "input", "hotkey",
        r"(?P<keys>(ctrl|alt|shift|cmd|win|command)\s*[+＋]\s*([A-Za-z0-9]+|Enter|回车|Space|空格|Esc|Tab|Backspace|Delete|Insert|Home|End|PgUp|PgDn|Up|Down|Left|Right)(\s*[+＋]\s*([A-Za-z0-9]+|Enter|回车|Space|空格|Esc|Tab|Backspace|Delete|Insert|Home|End|PgUp|PgDn|Up|Down|Left|Right))*)",
        0.95, AccessLevel::L2)?);
    rules.push(Rule::new("inp.screenshot", "input", "screenshot",
        r"截屏|截图|抓屏|屏幕快照", 0.96, AccessLevel::L3)?);

    Ok(rules)
}

// --- PyO3 bindings ---
use pyo3::prelude::*;

#[pyclass(name = "IntentRouter")]
pub struct PyIntentRouter {
    inner: IntentRouter,
}

#[pymethods]
impl PyIntentRouter {
    #[new]
    fn new() -> Self {
        Self { inner: IntentRouter::new() }
    }

    #[pyo3(signature = (text, role=None))]
    fn route(&self, text: &str, role: Option<&str>) -> PyResult<PyObject> {
        let identity = role.map(|r| Identity::new("user", r, "default"));
        let result = self.inner.route(text, identity.as_ref());
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("op", result.op_name)?;
            dict.set_item("act", result.act)?;
            dict.set_item("confidence", result.confidence)?;
            dict.set_item("ambiguous", result.ambiguous)?;
            dict.set_item("matched_rule", result.matched_rule)?;
            let params = pyo3::types::PyDict::new_bound(py);
            for (k, v) in &result.params {
                params.set_item(k, v)?;
            }
            dict.set_item("params", params)?;
            let cands = pyo3::types::PyList::empty_bound(py);
            for c in &result.candidates {
                let cd = pyo3::types::PyDict::new_bound(py);
                cd.set_item("op", &c.op)?;
                cd.set_item("act", &c.act)?;
                cd.set_item("confidence", c.confidence)?;
                cd.set_item("rule", &c.rule)?;
                cands.append(cd)?;
            }
            dict.set_item("candidates", cands)?;
            Ok(dict.unbind().into())
        })
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIntentRouter>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_notepad() {
        let router = IntentRouter::new();
        let result = router.route("打开记事本", None);
        assert_eq!(result.op_name, "app");
        assert_eq!(result.act, "open_app");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_volume_get() {
        let router = IntentRouter::new();
        let result = router.route("当前音量多大", None);
        assert_eq!(result.op_name, "volume");
        assert_eq!(result.act, "get_volume");
    }

    #[test]
    fn test_empty_text() {
        let router = IntentRouter::new();
        let result = router.route("", None);
        assert!(result.op_name.is_empty());
    }

    #[test]
    fn test_unknown_intent() {
        let router = IntentRouter::new();
        let result = router.route("今天天气怎么样", None);
        assert!(result.op_name.is_empty());
    }
}
