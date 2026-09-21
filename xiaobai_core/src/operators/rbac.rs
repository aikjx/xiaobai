//! RBAC (Role-Based Access Control) 4-level system.
//!
//! Replaces Python `operator/base.py` AccessLevel and Identity types.

use serde::{Deserialize, Serialize};

/// 4 access levels aligned with mox-system Role hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AccessLevel {
    /// L0: Auditor — read-only harmless (list apps, read volume)
    L0 = 0,
    /// L1: Member — non-destructive writes (open app, set volume, open file)
    L1 = 1,
    /// L2: Expert/Coordinator — clipboard / keyboard-mouse input / mouse click
    L2 = 2,
    /// L3: MoxAdmin — destructive (close app / delete file / keyboard automation)
    L3 = 3,
}

impl AccessLevel {
    pub fn from_role(role: &str) -> Self {
        match role.trim() {
            "MoxAdmin" => AccessLevel::L3,
            "Coordinator" | "Expert" => AccessLevel::L2,
            "Member" => AccessLevel::L1,
            "Auditor" => AccessLevel::L0,
            other => {
                let s = other.trim().to_uppercase();
                if s.starts_with('L') && s[1..].chars().all(|c| c.is_ascii_digit()) {
                    let v: u8 = s[1..].parse().unwrap_or(0);
                    match v {
                        0 => AccessLevel::L0,
                        1 => AccessLevel::L1,
                        2 => AccessLevel::L2,
                        _ => AccessLevel::L3,
                    }
                } else if other.chars().all(|c| c.is_ascii_digit()) {
                    match other.parse::<u8>().unwrap_or(0) {
                        0 => AccessLevel::L0,
                        1 => AccessLevel::L1,
                        2 => AccessLevel::L2,
                        _ => AccessLevel::L3,
                    }
                } else {
                    AccessLevel::L0 // safe default: unauthenticated = L0
                }
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AccessLevel::L0 => "L0_PUBLIC",
            AccessLevel::L1 => "L1_USER",
            AccessLevel::L2 => "L2_POWER",
            AccessLevel::L3 => "L3_ADMIN",
        }
    }
}

impl std::fmt::Display for AccessLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Execution identity.
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

/// Public (L0) identity constant.
pub const ID_PUBLIC: Identity = Identity {
    user_id: String::new(),
    role: String::new(),
    tenant_id: String::new(),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_mapping() {
        assert_eq!(AccessLevel::from_role("MoxAdmin"), AccessLevel::L3);
        assert_eq!(AccessLevel::from_role("Expert"), AccessLevel::L2);
        assert_eq!(AccessLevel::from_role("Member"), AccessLevel::L1);
        assert_eq!(AccessLevel::from_role("Auditor"), AccessLevel::L0);
    }

    #[test]
    fn test_level_string() {
        assert_eq!(AccessLevel::from_role("L2"), AccessLevel::L2);
        assert_eq!(AccessLevel::from_role("3"), AccessLevel::L3);
    }

    #[test]
    fn test_unknown_defaults_l0() {
        assert_eq!(AccessLevel::from_role("unknown"), AccessLevel::L0);
    }

    #[test]
    fn test_ordering() {
        assert!(AccessLevel::L3 > AccessLevel::L0);
        assert!(AccessLevel::L1 >= AccessLevel::L1);
    }
}
