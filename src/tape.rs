use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::tensor::Tensor;

/// Sentinel stored in `Tensor::tape_node` for tensors that are not the output of
/// a recorded operation. Node ids start at 0, so 0 cannot be used as "no node".
pub const NO_NODE: usize = usize::MAX;

thread_local! {
    /// The tape never escapes its thread, so `Rc<RefCell<_>>` is the honest
    /// representation; the previous `Arc<RwLock<_>>` paid for atomic refcounts
    /// and lock arbitration that could never be contended.
    static TAPE: RefCell<Vec<Rc<dyn Fn()>>> = const { RefCell::new(Vec::new()) };

    /// Whether operations should record backward closures on the tape.
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Namespace for the thread-local autograd tape.
///
/// The tape is per-thread: a graph built on one thread can only be
/// differentiated on that same thread.
pub struct Tape;

/// RAII guard that disables gradient recording for its lifetime.
///
/// Without this, every forward pass — including evaluation — appends closures to
/// the thread-local tape that keep their input tensors alive, so an inference
/// loop grows memory without bound until the next [`Tape::reset`].
///
/// ```
/// # use taper::{Tensor, tape};
/// let x = Tensor::new(vec![1.0, 2.0], &[2]).requires_grad();
/// let _guard = tape::no_grad();
/// let y = x.relu(); // not recorded
/// ```
#[must_use = "gradients stay disabled only while the guard is alive"]
pub struct NoGrad {
    previous: bool,
}

impl Drop for NoGrad {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.previous));
    }
}

/// Disable gradient recording until the returned guard is dropped.
pub fn no_grad() -> NoGrad {
    let previous = GRAD_ENABLED.with(|g| g.replace(false));
    NoGrad { previous }
}

/// Whether operations on this thread currently record backward closures.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

impl Tape {
    /// Clear recorded nodes but keep the tape alive.
    pub fn reset() {
        TAPE.with(|t| t.borrow_mut().clear());
    }

    /// Number of operations currently recorded on this thread's tape.
    pub fn len() -> usize {
        TAPE.with(|t| t.borrow().len())
    }

    pub fn is_empty() -> bool {
        Self::len() == 0
    }

    pub fn push_binary_op<F>(a: &Tensor, b: &Tensor, output: &Tensor, backward_fn: F)
    where
        F: Fn() + 'static,
    {
        if !(a.requires_grad || b.requires_grad) {
            return;
        }
        Self::push(output, backward_fn);
    }

    pub fn push_unary_op<F>(input: &Tensor, output: &Tensor, backward_fn: F)
    where
        F: Fn() + 'static,
    {
        if !input.requires_grad {
            return;
        }
        Self::push(output, backward_fn);
    }

    fn push<F>(output: &Tensor, backward_fn: F)
    where
        F: Fn() + 'static,
    {
        if !is_grad_enabled() {
            return;
        }

        let id = TAPE.with(|tape| {
            let mut nodes = tape.borrow_mut();
            nodes.push(Rc::new(backward_fn));
            nodes.len() - 1
        });
        output
            .tape_node
            .store(id, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Execute backward functions up to `final_node_id` (inclusive), in reverse.
/// We clone closures out first to avoid holding any borrows while executing.
pub fn backward(final_node_id: usize) {
    if final_node_id == NO_NODE {
        return;
    }

    // Clone the handles out first so no borrow is held while they run: a
    // backward closure is free to record new operations.
    let fns: Vec<Rc<dyn Fn()>> = TAPE.with(|t| {
        let nodes = t.borrow();
        if nodes.is_empty() {
            return Vec::new();
        }
        let end = final_node_id.min(nodes.len() - 1);
        nodes[..=end].to_vec()
    });

    // Run in reverse with no outstanding borrows.
    for f in fns.into_iter().rev() {
        (f)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_runs_for_the_first_recorded_node() {
        // Node ids start at 0; using 0 as the "no node" sentinel silently dropped
        // gradients for any graph whose output was the very first recorded op.
        Tape::reset();
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let c = a.matmul(&b);
        c.backward();
        assert!(
            a.grad_ref().is_some(),
            "single-op graph produced no gradient"
        );
    }

    #[test]
    fn no_grad_suppresses_recording() {
        Tape::reset();
        let x = Tensor::new(vec![1.0, -2.0], &[2]).requires_grad();
        {
            let _guard = no_grad();
            let _y = x.relu();
            assert!(Tape::is_empty(), "no_grad still recorded a node");
        }
        assert!(
            is_grad_enabled(),
            "guard did not restore the previous state"
        );
        let _y = x.relu();
        assert_eq!(Tape::len(), 1);
    }
}
