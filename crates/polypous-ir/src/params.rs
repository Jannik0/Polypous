//! What the trainer and the runtime need to know about each param (`docs/specs/graph-ir.md`
//! §2.4, §2.4.1; `docs/specs/training.md` §1.1; `docs/specs/runtime.md` §3.1): its uses after
//! expanding repeats, its fans, whether it is a `batch_norm` running statistic, and whether it is
//! trainable.
//!
//! Shape inference ([`crate::infer`]) records every use while it resolves the params of each node,
//! with the `repeat` prefixes expanded (§2.6): one use per node, role and iteration. The fans of a
//! use follow the table of §2.4.1 ("Fans"); [`fans_of`] implements it. After inference, the fan
//! checks of §2.4.1 report `invalid_init` (§4.2), and [`ParamInfo`]s become part of the
//! [`Analysis`](crate::Analysis).

use std::collections::HashMap;
use std::fmt;

use crate::model::Init;

/// The fans of a param (§2.4.1, "Fans"): PyTorch's `_calculate_fan_in_and_fan_out`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fans {
    /// `fan_in = shape[1] · rf`, `rf` the product of the dims after the second.
    pub fan_in: u64,
    /// `fan_out = shape[0] · rf`.
    pub fan_out: u64,
}

impl fmt::Display for Fans {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fan_in {}, fan_out {}", self.fan_in, self.fan_out)
    }
}

/// One use of a param by a node, in one iteration of its enclosing repeats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamUse {
    /// The node path of runtime.md §5: `fc1`, `layers[3]/attn`, `outer[1]/inner[0]/bn`.
    pub node: String,
    /// The op, `nn.linear`.
    pub op: &'static str,
    /// The op param (the role), `weight`.
    pub role: &'static str,
}

impl fmt::Display for ParamUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node `{}` ({}) as `{}`", self.node, self.op, self.role)
    }
}

/// Whether a use of `op`'s param `role` is a `batch_norm` running statistic (§3.3).
pub fn is_running_stat_role(op: &str, role: &str) -> bool {
    op == "nn.batch_norm" && (role == "running_mean" || role == "running_var")
}

/// The fans of `op`'s param `role` with the literal `shape` (§2.4.1, "Fans"), or `None` for a
/// role without fans. Only the weights of `nn.linear`, `nn.embedding`, `nn.conv1d` and
/// `nn.conv2d` have fans. Also `None` if the product overflows `u64` (it can't for a param whose
/// element count was checked).
pub fn fans_of(op: &str, role: &str, shape: &[u64]) -> Option<Fans> {
    let has_fans =
        role == "weight" && matches!(op, "nn.linear" | "nn.embedding" | "nn.conv1d" | "nn.conv2d");
    if !has_fans || shape.len() < 2 {
        return None;
    }
    let rf = shape[2..]
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))?;
    Some(Fans {
        fan_in: shape[1].checked_mul(rf)?,
        fan_out: shape[0].checked_mul(rf)?,
    })
}

/// What the analysis knows about one declared param.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamInfo {
    /// The param name.
    pub name: String,
    /// The declared (literal) shape.
    pub shape: Vec<usize>,
    /// `init`, if written.
    pub init: Option<Init>,
    /// `trainable` as written (default `true`). See [`ParamInfo::is_trainable`].
    pub trainable_written: bool,
    /// Some `nn.batch_norm` node uses the param as `running_mean` or `running_var`.
    pub running_stat: bool,
    /// The number of uses after expanding repeats: one per node, role and iteration.
    pub uses: u64,
    /// The first uses, at most two, in inference order (for error messages).
    pub first_uses: Vec<ParamUse>,
    /// The param's fans: `Some` if it has at least one use and every use is in a role with fans
    /// and gives the same fans (§2.4.1). A fan-based `init` validates only then.
    pub fans: Option<Fans>,
}

impl ParamInfo {
    /// Whether the param is trainable (training.md §1.1): `trainable: true` (the default) and not
    /// a `batch_norm` running statistic.
    pub fn is_trainable(&self) -> bool {
        self.trainable_written && !self.running_stat
    }
}

/// Whether `init` takes fans (`kaiming_*`, `xavier_*`).
pub fn is_fan_based(init: &Init) -> bool {
    matches!(
        init,
        Init::KaimingUniform { .. }
            | Init::KaimingNormal
            | Init::XavierUniform
            | Init::XavierNormal
    )
}

/// The uses of one param, aggregated while inference runs.
#[derive(Debug, Clone, Default)]
pub(crate) struct Uses {
    pub count: u64,
    pub first: Vec<ParamUse>,
    pub running_stat: bool,
    /// The fans of the first use with fans.
    pub fans: Option<(Fans, ParamUse)>,
    /// The first use in a role without fans.
    pub no_fans: Option<ParamUse>,
    /// The first use whose fans differ from `fans`.
    pub disagree: Option<(Fans, ParamUse)>,
}

/// Every param's uses, by param name.
#[derive(Debug, Default)]
pub(crate) struct UseRecorder {
    pub by_param: HashMap<String, Uses>,
    /// `false` if some node's params could not be resolved (an error was reported), so "no node
    /// uses the param" can't be decided.
    pub complete: bool,
}

impl UseRecorder {
    pub fn new() -> Self {
        UseRecorder {
            by_param: HashMap::new(),
            complete: true,
        }
    }

    /// Records `multiplicity` uses of `param` in `op`'s role `role`, with `fans` (from
    /// [`fans_of`]). `path(k)` gives the node path of the `k`-th of these uses; it is called at
    /// most twice in total per param.
    pub fn record(
        &mut self,
        param: &str,
        op: &'static str,
        role: &'static str,
        fans: Option<Fans>,
        multiplicity: u64,
        mut path: impl FnMut(u64) -> String,
    ) {
        if multiplicity == 0 {
            return;
        }
        let uses = match self.by_param.get_mut(param) {
            Some(u) => u,
            None => self.by_param.entry(param.to_owned()).or_default(),
        };
        let mut k = 0;
        while uses.first.len() < 2 && k < multiplicity {
            uses.first.push(ParamUse {
                node: path(k),
                op,
                role,
            });
            k += 1;
        }
        let mut this = || ParamUse {
            node: path(0),
            op,
            role,
        };
        uses.count = uses.count.saturating_add(multiplicity);
        if is_running_stat_role(op, role) {
            uses.running_stat = true;
        }
        match fans {
            None => {
                if uses.no_fans.is_none() {
                    uses.no_fans = Some(this());
                }
            }
            Some(f) => match &uses.fans {
                None => uses.fans = Some((f, this())),
                Some((first, _)) if *first != f && uses.disagree.is_none() => {
                    uses.disagree = Some((f, this()));
                }
                Some(_) => {}
            },
        }
    }
}

/// The message of `invalid_init` for a fan-based `init` (§2.4.1), or `None` if the uses give
/// the param its fans. `complete` is [`UseRecorder::complete`].
pub(crate) fn fan_problem(
    name: &str,
    kind: &str,
    uses: Option<&Uses>,
    complete: bool,
) -> Option<String> {
    let needs = format!(
        "param `{name}` has init `{kind}`, which takes its fans from the nodes that use it"
    );
    match uses {
        None if complete => Some(format!("{needs}, but no node uses it")),
        None => None,
        Some(u) => {
            if let Some(use_) = &u.no_fans {
                Some(format!(
                    "{needs}, but {use_} uses it in a role without fans (only the `weight` of nn.linear, nn.embedding, nn.conv1d and nn.conv2d has fans)"
                ))
            } else if let (Some((f1, u1)), Some((f2, u2))) = (&u.fans, &u.disagree) {
                Some(format!(
                    "{needs}, but its uses give different fans: {u1} gives {f1}, {u2} gives {f2}"
                ))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fans_follow_the_table() {
        let f = |fan_in, fan_out| Some(Fans { fan_in, fan_out });
        assert_eq!(fans_of("nn.linear", "weight", &[128, 784]), f(784, 128));
        assert_eq!(fans_of("nn.embedding", "weight", &[1000, 64]), f(64, 1000));
        // conv1d: [C_out, C_in/groups, k].
        assert_eq!(fans_of("nn.conv1d", "weight", &[8, 2, 3]), f(6, 24));
        // conv2d: [C_out, C_in/groups, kh, kw]; fan_out is not divided by groups.
        assert_eq!(fans_of("nn.conv2d", "weight", &[16, 4, 3, 5]), f(60, 240));
        for (op, role) in [
            ("nn.linear", "bias"),
            ("nn.conv2d", "bias"),
            ("nn.layer_norm", "weight"),
            ("nn.rms_norm", "weight"),
            ("nn.group_norm", "weight"),
            ("nn.batch_norm", "weight"),
            ("nn.batch_norm", "running_mean"),
            ("nn.batch_norm", "running_var"),
        ] {
            assert_eq!(fans_of(op, role, &[4, 4]), None, "{op} {role}");
        }
        assert_eq!(fans_of("nn.linear", "weight", &[u64::MAX, 2, 2]), None);
    }

    #[test]
    fn disagreeing_fans_are_reported() {
        // Can't happen with the roles of nn v1 (fans depend only on the shape); the rule is
        // checked on the recorder directly.
        let mut r = UseRecorder::new();
        let a = Fans {
            fan_in: 3,
            fan_out: 4,
        };
        let b = Fans {
            fan_in: 5,
            fan_out: 4,
        };
        r.record("w", "nn.linear", "weight", Some(a), 1, |_| "fc1".into());
        r.record("w", "nn.linear", "weight", Some(a), 2, |k| {
            format!("l[{k}]/fc")
        });
        assert_eq!(
            fan_problem("w", "xavier_uniform", r.by_param.get("w"), true),
            None
        );
        r.record("w", "nn.conv1d", "weight", Some(b), 1, |_| "conv".into());
        let msg = fan_problem("w", "xavier_uniform", r.by_param.get("w"), true).unwrap();
        assert!(msg.contains("different fans"), "{msg}");
        assert!(
            msg.contains("node `fc1` (nn.linear) as `weight` gives fan_in 3, fan_out 4"),
            "{msg}"
        );
        assert!(
            msg.contains("node `conv` (nn.conv1d) as `weight` gives fan_in 5, fan_out 4"),
            "{msg}"
        );
        let u = &r.by_param["w"];
        assert_eq!(u.count, 4);
        assert_eq!(u.first.len(), 2);
        assert_eq!(u.first[1].node, "l[0]/fc");
    }

    #[test]
    fn unused_is_reported_only_when_complete() {
        assert!(
            fan_problem("w", "kaiming_normal", None, true)
                .unwrap()
                .contains("no node uses it")
        );
        assert_eq!(fan_problem("w", "kaiming_normal", None, false), None);
    }
}
