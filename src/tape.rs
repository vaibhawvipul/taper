use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    static TAPE: RefCell<Option<Rc<RefCell<TapeInner>>>> = RefCell::new(None);
}

#[allow(dead_code)]
pub struct Tape {
    inner: Rc<RefCell<TapeInner>>,
}

struct TapeInner {
    nodes: Vec<Node>,
}

type BackwardFn = Box<dyn Fn()>;

struct Node {
    backward_fn: BackwardFn,
}

impl Tape {
    pub fn new() -> Self {
        let inner = Rc::new(RefCell::new(TapeInner {
            nodes: Vec::new(),
        }));

        // Set this tape as the active one
        TAPE.with(|t| {
            *t.borrow_mut() = Some(inner.clone());
        });

        Tape { inner }
    }

    pub fn push_binary_op<F>(
        a: &Tensor,
        b: &Tensor,
        output: &Tensor,
        backward_fn: F,
    ) where
        F: Fn() + 'static
    {
        if !a.requires_grad && !b.requires_grad {
            return;
        }

        TAPE.with(|tape| {
            if let Some(ref tape_inner) = *tape.borrow() {
                let mut tape_inner = tape_inner.borrow_mut();
                let node_id = tape_inner.nodes.len();

                // Set the node ID on the output tensor
                output.tape_node.set(Some(node_id));

                tape_inner.nodes.push(Node {
                    backward_fn: Box::new(backward_fn),
                });
            }
        });
    }

    pub fn push_unary_op<F>(
        input: &Tensor,
        output: &Tensor,
        backward_fn: F,
    ) where
        F: Fn() + 'static
    {
        if !input.requires_grad {
            return;
        }

        TAPE.with(|tape| {
            if let Some(ref tape_inner) = *tape.borrow() {
                let mut tape_inner = tape_inner.borrow_mut();
                let node_id = tape_inner.nodes.len();

                output.tape_node.set(Some(node_id));

                tape_inner.nodes.push(Node {
                    backward_fn: Box::new(backward_fn),
                });
            }
        });
    }
}

pub fn backward(final_node_id: usize) {
    TAPE.with(|tape| {
        if let Some(ref tape_inner) = *tape.borrow() {
            let tape_inner = tape_inner.borrow();

            // Execute backward functions in reverse order
            for i in (0..=final_node_id).rev() {
                if i < tape_inner.nodes.len() {
                    (tape_inner.nodes[i].backward_fn)();
                }
            }
        }
    });
}

impl Drop for Tape {
    fn drop(&mut self) {
        // Clear the thread-local tape when this Tape is dropped
        TAPE.with(|t| {
            *t.borrow_mut() = None;
        });
    }
}
