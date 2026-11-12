//! [`CancellationToken`] (`docs/specs/runtime.md` §4).

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cancels a running call from any thread (runtime.md §4).
///
/// A cheap-to-clone handle: clones share one flag. Pass `&token` to
/// [`Model::run`](crate::Model::run) and call [`CancellationToken::cancel`] on a clone; the call
/// stops before its next node or `repeat` iteration with [`RuntimeError::Cancelled`].
///
/// [`RuntimeError::Cancelled`]: crate::RuntimeError::Cancelled
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. Idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_flag() {
        let token = CancellationToken::new();
        let clone = token.clone();
        assert!(!token.is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
        assert!(format!("{token:?}").contains("true"));
    }
}
