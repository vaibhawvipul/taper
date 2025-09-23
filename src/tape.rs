use std::sync::{Arc, RwLock};
use std::rc::Rc;

use crate::tensor::Tensor;

thread_local! {
    static TAPE: std::cell::RefCell<Option<Arc<RwLock<TapeInner>>>> =
        std::cell::RefCell::new(None);
}

#[allow(dead_code)]
pub struct Tape {
    inner: Arc<RwLock<TapeInner>>,
}

struct TapeInner {
    // Store closures as Rc so we can clone them out of the borrow and run safely.
    nodes: Vec<Node>,
}

struct Node {
    backward_fn: Rc<dyn Fn()>,
}

impl Tape {
    /// Ensure a tape exists for this thread and return a handle.
    pub fn new() -> Self {
        Self::ensure_active();
        let inner = TAPE.with(|t| t.borrow().as_ref().cloned().expect("tape missing"));
        Tape { inner }
    }

    /// Make sure the thread-local tape is initialized.
    pub fn ensure_active() {
        TAPE.with(|t| {
            if t.borrow().is_none() {
                *t.borrow_mut() = Some(Arc::new(RwLock::new(TapeInner { nodes: Vec::new() })));
            }
        });
    }

    /// Clear recorded nodes but keep the tape alive.
    pub fn reset() {
        TAPE.with(|t| {
            if let Some(rc) = t.borrow().as_ref().cloned() {
                rc.write().unwrap().nodes.clear();
            }
        });
    }

    pub fn push_binary_op<F>(a: &Tensor, b: &Tensor, output: &Tensor, backward_fn: F)
    where
        F: Fn() + 'static,
    {
        if !(a.requires_grad || b.requires_grad) {
            return;
        }
        Self::ensure_active();

        // Take Rc out while the RefCell borrow is active, then drop it before mut borrow.
        let rc_opt = TAPE.with(|tape| tape.borrow().as_ref().cloned());
        if let Some(rc) = rc_opt {
            let id = {
                let mut inner = rc.write().unwrap();
                let id = inner.nodes.len();
                inner.nodes.push(Node {
                    backward_fn: Rc::new(backward_fn),
                });
                id
            };
            // stamp after releasing inner borrow
            output.tape_node.store(id, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn push_unary_op<F>(input: &Tensor, output: &Tensor, backward_fn: F)
    where
        F: Fn() + 'static,
    {
        if !input.requires_grad {
            return;
        }
        Self::ensure_active();

        let rc_opt = TAPE.with(|tape| tape.borrow().as_ref().cloned());
        if let Some(rc) = rc_opt {
            let id = {
                let mut inner = rc.write().unwrap();
                let id = inner.nodes.len();
                inner.nodes.push(Node {
                    backward_fn: Rc::new(backward_fn),
                });
                id
            };
            output.tape_node.store(id, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Execute backward functions up to `final_node_id` (inclusive), in reverse.
/// We clone closures out first to avoid holding any RefCell borrows while executing.
pub fn backward(final_node_id: usize) {
    // Clone the closures we’ll run (no borrows alive afterwards).
    let fns: Vec<Rc<dyn Fn()>> = TAPE.with(|t| {
        let Some(rc) = t.borrow().as_ref().cloned() else {
            return Vec::new();
        };
        let inner = rc.read().unwrap();
        if inner.nodes.is_empty() {
            return Vec::new();
        }
        let end = final_node_id.min(inner.nodes.len().saturating_sub(1));
        inner.nodes[..=end]
            .iter()
            .map(|n| n.backward_fn.clone())
            .collect()
    });

    // Run in reverse with no outstanding borrows.
    for f in fns.into_iter().rev() {
        (f)();
    }
}

impl Drop for Tape {
    fn drop(&mut self) {
        // Keep the tape alive; prefer explicit Tape::reset() per batch.
    }
}
