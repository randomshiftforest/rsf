use std::mem;

use rand::{prelude::ThreadRng, thread_rng, Rng};

pub enum ReservoirUpdate<T> {
    Skip(T),
    Insert(T),
    Replace(T, T),
}

pub struct Reservoir<T> {
    r: usize,
    i: usize,
    buf: Vec<T>,
    rng: ThreadRng,
}

impl<T: Clone> Reservoir<T> {
    pub fn new(r: usize) -> Self {
        Self {
            r,
            i: 0,
            buf: Vec::with_capacity(r),
            rng: thread_rng(),
        }
    }

    pub fn insert(&mut self, item: T) -> ReservoirUpdate<T> {
        self.i += 1;
        if self.buf.len() < self.r {
            self.buf.push(item.clone());
            ReservoirUpdate::Insert(item)
        } else {
            let j = self.rng.gen_range(0..self.i);
            if j < self.r {
                let old_item = mem::replace(&mut self.buf[j], item.clone());
                ReservoirUpdate::Replace(old_item, item)
            } else {
                ReservoirUpdate::Skip(item)
            }
        }
    }
}
