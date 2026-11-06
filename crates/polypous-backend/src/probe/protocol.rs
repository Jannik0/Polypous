//! The prober protocol: what the app sends the prober and the JSON lines the prober answers with
//! (ADR-0013 §1; placement.md §1, "Prober protocol").
//!
//! Both sides use these types, so they cannot drift apart. The app passes a [`ProbeRequest`] as
//! one command-line argument; the prober writes one [`ProberLine`] per line to stdout and logs to
//! stderr. Everything the prober writes is untrusted input to the app: the supervisor
//! ([`super::supervisor`]) enforces [`MAX_LINE_BYTES`] and [`MAX_LINES`], and the collector
//! ([`super::collect`]) rejects lines that don't fit the protocol's order.

use serde::{Deserialize, Serialize};

use crate::device::{BackendKind, DeviceId, DeviceInfo};

/// Version of this protocol. The prober's [`ProberLine::Header`] must carry exactly this value.
/// Bump it on any incompatible change to [`ProbeRequest`] or [`ProberLine`].
pub const PROTOCOL_VERSION: u32 = 1;

/// The first argument that makes a Polypous executable run as the prober
/// (`polypous::probe::run_prober_if_requested`). The second argument is `--request`, the third
/// the [`ProbeRequest`] as JSON.
pub const PROBER_FLAG: &str = "--polypous-prober";

/// The argument before the JSON request.
pub const REQUEST_FLAG: &str = "--request";

/// Longest line the app accepts from the prober, in bytes without the newline. A device line is
/// well under 2 KiB; anything longer is a protocol error.
pub const MAX_LINE_BYTES: usize = 64 * 1024;

/// Most lines the app reads from one prober process. A header, a few backend lines and three
/// lines per device stay far below; more is a protocol error.
pub const MAX_LINES: usize = 1024;

/// Most devices one prober process may enumerate.
pub const MAX_DEVICES: usize = 128;

/// What the app asks the prober to do.
///
/// JSON: `{"protocol":1,"software_rasterizers":false,"devices":null,"skip":[],"skip_backends":[],
/// "exclude":[]}`;
/// every field is optional, `{}` is "probe everything".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProbeRequest {
    /// [`PROTOCOL_VERSION`] of the app. The prober refuses other versions.
    pub protocol: u32,
    /// Open and probe software rasterizers (placement.md §1). Off: they are reported as
    /// `software_rasterizer_skipped` without being opened.
    pub software_rasterizers: bool,
    /// Probe only these devices (`None`: all). Every enumerated device is still reported with an
    /// `enumerated` line.
    pub devices: Option<Vec<DeviceId>>,
    /// Devices the app already knows from its probe cache: a matching device is reported with a
    /// `skipped` line and never opened. Checked in order; the first match wins.
    pub skip: Vec<SkipRule>,
    /// Backends not to enumerate at all, because enumerating them killed an earlier prober.
    pub skip_backends: Vec<BackendKind>,
    /// Devices an earlier prober process of the same probe already reported: enumerated, but
    /// neither probed nor reported again (restart after a crash, ADR-0013 §1 "Recovery").
    pub exclude: Vec<DeviceId>,
}

impl Default for ProbeRequest {
    fn default() -> Self {
        ProbeRequest {
            protocol: PROTOCOL_VERSION,
            software_rasterizers: false,
            devices: None,
            skip: Vec::new(),
            skip_backends: Vec::new(),
            exclude: Vec::new(),
        }
    }
}

impl ProbeRequest {
    /// Whether the request asks for `id` to be probed (no `devices` filter, or `id` is in it).
    pub fn wants(&self, id: DeviceId) -> bool {
        self.devices
            .as_ref()
            .is_none_or(|devices| devices.contains(&id))
    }

    /// The index of the first skip rule that matches `device`.
    pub fn skip_rule_for(&self, device: &EnumeratedDevice) -> Option<usize> {
        self.skip.iter().position(|rule| rule.matches(device))
    }
}

/// How the prober recognizes a device the app already knows (placement.md §1, "Probe cache").
///
/// JSON: `{"by":"key","key":"pci:0000:01:00.0"}`, `{"by":"id","id":{"backend":"cpu"}}`,
/// `{"by":"enumeration","id":…,"key":null,"enum_fingerprint":"…"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum SkipRule {
    /// The device whose stable key is `key` (a PCI address on Linux), whatever its id now.
    Key {
        /// The stable key.
        key: String,
    },
    /// The device with this id. Used only when the device set is unchanged, so ids still
    /// address the same devices.
    Id {
        /// The device id.
        id: DeviceId,
    },
    /// The device with this id whose enumeration-time key and fingerprint are exactly these
    /// (enumerate-only comparison when the OS driver fingerprint is unknown).
    Enumeration {
        /// The device id.
        id: DeviceId,
        /// The stable key the enumeration reported, if any.
        key: Option<String>,
        /// The enumeration fingerprint, if any.
        enum_fingerprint: Option<String>,
    },
}

impl SkipRule {
    /// Whether this rule matches an enumerated device.
    pub fn matches(&self, device: &EnumeratedDevice) -> bool {
        match self {
            SkipRule::Key { key } => device.key.as_deref() == Some(key.as_str()),
            SkipRule::Id { id } => device.id == *id,
            SkipRule::Enumeration {
                id,
                key,
                enum_fingerprint,
            } => {
                device.id == *id
                    && device.key == *key
                    && device.enum_fingerprint == *enum_fingerprint
            }
        }
    }
}

/// What enumeration alone tells about a device, without opening it.
///
/// JSON: `{"id":{"backend":"vulkan","adapter":"discrete_gpu","index":0},"key":"pci:0000:01:00.0",
/// "enum_fingerprint":"…"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnumeratedDevice {
    /// The device id.
    pub id: DeviceId,
    /// A key that stays the same across reboots and device reordering, known before the device is
    /// opened: `pci:<address>` where the graphics API reports the PCI address at enumeration
    /// (Vulkan, Metal). `None` for CUDA (its PCI address needs a context), the CPU and software
    /// rasterizers.
    pub key: Option<String>,
    /// Adapter name, vendor/device ids and driver strings as the graphics API reports them at
    /// enumeration, where it does (Vulkan, Metal). Compared to detect a changed device or driver
    /// when the OS driver fingerprint is unknown.
    pub enum_fingerprint: Option<String>,
}

/// One line the prober writes to stdout (JSON, one object per line, `type` names the variant).
///
/// Order: `header`; per backend `enumerating`, then `enumerated` per device or
/// `enumeration_failed`; then per device in probe order either `skipped`, or `probing` followed by
/// `probed`; finally `done`. `probing` is written and flushed before the device is opened, so the
/// app knows which device was in progress if the prober dies or hangs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProberLine {
    /// The first line.
    Header {
        /// [`PROTOCOL_VERSION`] of the prober.
        protocol: u32,
        /// The prober's Polypous version (`CARGO_PKG_VERSION`), for logs.
        polypous_version: String,
    },
    /// The prober starts enumerating a backend.
    Enumerating {
        /// The backend.
        backend: BackendKind,
    },
    /// Enumerating a backend failed (a caught panic); its devices are unknown.
    EnumerationFailed {
        /// The backend.
        backend: BackendKind,
        /// What went wrong.
        message: String,
    },
    /// A device the enumeration found.
    Enumerated {
        /// The device.
        device: EnumeratedDevice,
    },
    /// A device matched the request's skip rule `rule` and was not opened.
    Skipped {
        /// The device.
        id: DeviceId,
        /// Index into [`ProbeRequest::skip`].
        rule: usize,
    },
    /// The prober is about to open and probe this device.
    Probing {
        /// The device.
        id: DeviceId,
    },
    /// The device's description and probe result.
    Probed {
        /// The device, with its status (boxed: the other lines are small).
        device: Box<DeviceInfo>,
    },
    /// The prober finished; it exits after this line.
    Done,
}

impl ProberLine {
    /// The line's JSON form, without the newline.
    pub fn to_json(&self) -> String {
        // Plain serde data of strings, integers and enums: serializing cannot fail. The fallback
        // is a line the app rejects as malformed, never a silent success.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::AdapterKind;

    #[test]
    fn request_json_is_stable_and_defaults() {
        let request = ProbeRequest::default();
        assert_eq!(
            serde_json::to_string(&request).unwrap(),
            r#"{"protocol":1,"software_rasterizers":false,"devices":null,"skip":[],"skip_backends":[],"exclude":[]}"#
        );
        assert_eq!(serde_json::from_str::<ProbeRequest>("{}").unwrap(), request);
        assert!(serde_json::from_str::<ProbeRequest>(r#"{"bogus":1}"#).is_err());
    }

    #[test]
    fn line_json_is_stable() {
        let lines = [
            (
                ProberLine::Header {
                    protocol: 1,
                    polypous_version: "0.0.0".into(),
                },
                r#"{"type":"header","protocol":1,"polypous_version":"0.0.0"}"#,
            ),
            (
                ProberLine::Enumerated {
                    device: EnumeratedDevice {
                        id: DeviceId::Vulkan {
                            adapter: AdapterKind::DiscreteGpu,
                            index: 0,
                        },
                        key: Some("pci:0000:01:00.0".into()),
                        enum_fingerprint: None,
                    },
                },
                r#"{"type":"enumerated","device":{"id":{"backend":"vulkan","adapter":"discrete_gpu","index":0},"key":"pci:0000:01:00.0","enum_fingerprint":null}}"#,
            ),
            (
                ProberLine::Probing { id: DeviceId::Cpu },
                r#"{"type":"probing","id":{"backend":"cpu"}}"#,
            ),
            (
                ProberLine::Skipped {
                    id: DeviceId::Cpu,
                    rule: 2,
                },
                r#"{"type":"skipped","id":{"backend":"cpu"},"rule":2}"#,
            ),
            (ProberLine::Done, r#"{"type":"done"}"#),
        ];
        for (line, json) in lines {
            assert_eq!(line.to_json(), json);
            assert_eq!(serde_json::from_str::<ProberLine>(json).unwrap(), line);
        }
        assert!(serde_json::from_str::<ProberLine>(r#"{"type":"explode"}"#).is_err());
    }

    #[test]
    fn skip_rules_match() {
        let device = EnumeratedDevice {
            id: DeviceId::Vulkan {
                adapter: AdapterKind::DiscreteGpu,
                index: 1,
            },
            key: Some("pci:0000:01:00.0".into()),
            enum_fingerprint: Some("NVIDIA 1.0".into()),
        };
        assert!(
            SkipRule::Key {
                key: "pci:0000:01:00.0".into()
            }
            .matches(&device)
        );
        assert!(
            !SkipRule::Key {
                key: "pci:0000:02:00.0".into()
            }
            .matches(&device)
        );
        assert!(SkipRule::Id { id: device.id }.matches(&device));
        let enumeration = |fingerprint: &str| SkipRule::Enumeration {
            id: device.id,
            key: device.key.clone(),
            enum_fingerprint: Some(fingerprint.into()),
        };
        assert!(enumeration("NVIDIA 1.0").matches(&device));
        assert!(!enumeration("NVIDIA 2.0").matches(&device));

        let request = ProbeRequest {
            skip: vec![
                SkipRule::Id { id: DeviceId::Cpu },
                enumeration("NVIDIA 1.0"),
            ],
            ..Default::default()
        };
        assert_eq!(request.skip_rule_for(&device), Some(1));
        assert!(request.wants(DeviceId::Cpu));
        let only_cpu = ProbeRequest {
            devices: Some(vec![DeviceId::Cpu]),
            ..Default::default()
        };
        assert!(!only_cpu.wants(device.id));
    }
}
