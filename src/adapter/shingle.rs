use std::collections::VecDeque;

use ndarray::{Array1, ArrayBase, Data, Ix1};

pub struct Shingle<I: Iterator> {
    iter: I,
    buf: VecDeque<I::Item>,
    s: usize,
}

impl<S: Data<Elem = f32>, I: Iterator<Item = ArrayBase<S, Ix1>>> Shingle<I> {
    fn new(iter: I, s: usize) -> Self {
        let buf = VecDeque::with_capacity(s);
        Self { iter, buf, s }
    }

    fn shingled_item(&self) -> Array1<f32> {
        self.buf.iter().flatten().copied().collect()
    }
}

impl<S: Data<Elem = f32>, I: Iterator<Item = ArrayBase<S, Ix1>>> Iterator for Shingle<I> {
    type Item = Array1<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                None => return None,
                Some(item) => {
                    if self.buf.len() < self.s {
                        self.buf.push_back(item);
                    } else {
                        self.buf.pop_front();
                        self.buf.push_back(item);
                        return Some(self.shingled_item());
                    }
                }
            }
        }
    }
}

pub trait ShingleIter<S: Data<Elem = f32>>: Iterator<Item = ArrayBase<S, Ix1>> + Sized {
    fn shingle(self, s: usize) -> Shingle<Self> {
        Shingle::new(self, s)
    }
}

impl<S: Data<Elem = f32>, I: Iterator<Item = ArrayBase<S, Ix1>>> ShingleIter<S> for I {}
