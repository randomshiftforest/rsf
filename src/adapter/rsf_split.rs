use ndarray::{ArrayBase, Data, Ix1};

use crate::algorithm::{config::Config, forest::RSF};

pub struct RSFSplit<I> {
    iter: I,
    f: RSF,
}

impl<S, I> RSFSplit<I>
where
    S: Data<Elem = f32>,
    I: Iterator<Item = ArrayBase<S, Ix1>>,
{
    fn new(mut iter: I, cfg: &Config) -> Self {
        let mut f = RSF::from_config(cfg);
        iter.by_ref().take(cfg.n_points).for_each(|p| f.insert(&p));
        Self { iter, f }
    }
}

impl<S, I> Iterator for RSFSplit<I>
where
    S: Data<Elem = f32>,
    I: Iterator<Item = ArrayBase<S, Ix1>>,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|p| self.f.score(&p))
    }
}

pub trait RSFSplitIter<S: Data<Elem = f32>>: Iterator<Item = ArrayBase<S, Ix1>> + Sized {
    fn rsf_split(self, cfg: &Config) -> RSFSplit<Self> {
        RSFSplit::new(self, cfg)
    }
}

impl<S: Data<Elem = f32>, I: Iterator<Item = ArrayBase<S, Ix1>>> RSFSplitIter<S> for I {}
