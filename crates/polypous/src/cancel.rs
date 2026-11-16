//! [`CancellationToken`] (public-api.md §2 rule 5, §1.3).

use std::fmt;

/// Cancels a long operation from any thread: probing, loading and running.
///
/// A cheap-to-clone handle: clones share one flag. Pass `&token` to the operation and call
/// [`CancellationToken::cancel`] on a clone; the operation stops soon after and returns an
/// [`ErrorKind::Cancelled`](crate::ErrorKind::Cancelled) error.
///
/// It holds the runtime's token, so the same flag reaches the graph interpreter, which checks it
/// before every node and `repeat` iteration (runtime.md §4).
#[derive(Clone, Default)]
pub struct CancellationToken {
    inner: polypous_runtime::CancellationToken,
}

impl CancellationToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. Idempotent.
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// The runtime's token, sharing this token's flag.
    pub(crate) fn runtime(&self) -> &polypous_runtime::CancellationToken {
        &self.inner
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
        assert!(!token.runtime().is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
        assert!(token.runtime().is_cancelled());
        assert!(format!("{token:?}").contains("true"));
    }
}
