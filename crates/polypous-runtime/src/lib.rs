//! Graph runtime for Polypous.
//!
//! Responsibility: graph interpreter, executable plan compilation, placement planner, weight
//! residency (resident / streamed), prefetching, KV/state management. M1 implements the
//! single-device interpreter of `docs/specs/runtime.md`; placement and streaming (M4) and
//! sessions (M3) come later. See `docs/architecture.md` §2.
//!
//! - [`Plan`]: a validated graph compiled for one device and compute dtype (runtime.md §1):
//!   device check, topological order, value lifetimes.
//! - [`Model`]: a plan with its weights loaded from a [`WeightFile`] (runtime.md §2), and
//!   [`Model::run`], one call with inputs and state by name (runtime.md §3).
//! - [`CancellationToken`] (runtime.md §4) and [`RuntimeError`] with its [`ErrorKind`]
//!   (runtime.md §5).
//!
//! IR nodes map onto `polypous-backend`'s [`DynTensor`](polypous_backend::DynTensor) ops
//! ([`ops::MAPPED_OPS`]). No Burn type appears here (ADR-0007).
//!
//! ```
//! use polypous_backend::{DType, DynTensor, TensorData};
//! use polypous_backend::device::DeviceId;
//! use polypous_ir::Graph;
//! use polypous_runtime::{Model, Plan, PlanOptions, RunOptions};
//!
//! let graph = Graph::from_json(br#"{
//!   "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
//!   "dims": { "batch": {} },
//!   "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 2] } ],
//!   "outputs": [ { "name": "y", "value": "act" } ],
//!   "nodes": [ { "id": "act", "op": "nn.relu", "inputs": { "x": "x" } } ]
//! }"#).expect("valid graph");
//! let plan = Plan::new(&graph, PlanOptions::new(DeviceId::Cpu))?;
//! let model = Model::without_weights(plan)?;
//!
//! let bytes = [-1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
//! let x = DynTensor::from_data(&TensorData::new(DType::F32, vec![1, 2], bytes)?, DeviceId::Cpu)?;
//! let out = model.run(vec![("x".into(), x)], vec![], &RunOptions::default())?;
//! assert_eq!(out.output("y").map(|t| t.shape()), Some(vec![1, 2]));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod cancel;
mod error;
mod exec;
pub mod ops;
mod plan;
mod weights;

pub use cancel::CancellationToken;
pub use error::{ErrorKind, Result, RuntimeError, WeightProblem};
pub use exec::{Model, RunOptions, RunOutput, TraceStep};
pub use plan::{Plan, PlanOptions, ScheduledStep};
pub use weights::WeightFile;
