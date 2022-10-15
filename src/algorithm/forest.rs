use std::{
    ops::{Index, IndexMut},
    slice::IterMut,
};

use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};

use super::{
    config::Config,
    tree::{RandShiftTree, RSQT, RST},
};

pub struct RandShiftForest<T: RandShiftTree> {
    trees: Vec<T>,
}

impl<T: RandShiftTree> RandShiftForest<T> {
    pub fn from_config(cfg: &Config) -> Self {
        let mut rng = cfg.get_rng();
        let trees = (0..cfg.n_trees)
            .map(|i| T::from_config(cfg, i, &mut rng))
            .collect::<Vec<_>>();
        Self { trees }
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn iter_trees_mut(&mut self) -> IterMut<T> {
        self.trees.iter_mut()
    }

    pub fn insert<S: Data<Elem = f32>>(&mut self, p: &ArrayBase<S, Ix1>) {
        self.trees.iter_mut().for_each(|t| t.insert(p));
    }

    pub fn batch_insert<S: Data<Elem = f32>>(&mut self, ps: &ArrayBase<S, Ix2>) {
        self.trees.iter_mut().for_each(|t| t.batch_insert(ps));
    }

    pub fn remove<S: Data<Elem = f32>>(&mut self, p: &ArrayBase<S, Ix1>) {
        self.trees.iter_mut().for_each(|t| t.remove(p));
    }

    pub fn score<S: Data<Elem = f32>>(&self, p: &ArrayBase<S, Ix1>) -> f32 {
        self.trees.iter().map(|t| t.score(p)).sum::<f32>() / (self.n_trees() as f32)
    }

    pub fn n_points(&self) -> f32 {
        let point_sum: usize = self.trees.iter().map(|t| t.n_points()).sum();
        (point_sum as f32) / (self.n_trees() as f32)
    }

    pub fn batch_score<S: Data<Elem = f32>>(&self, ps: &ArrayBase<S, Ix2>) -> Array1<f32> {
        let zero = Array1::from_elem(ps.dim().0, 0.);
        let sum = self
            .trees
            .iter()
            .map(|t| t.batch_score(ps))
            .fold(zero, |sum, scores| sum + scores);
        sum / (self.n_trees() as f32)
    }

    pub fn sketch(&mut self, sketch_size: usize) {
        self.trees.iter_mut().for_each(|t| t.sketch(sketch_size));
    }

    pub fn extend(&mut self, other: Self) {
        self.trees
            .iter_mut()
            .zip(other.trees.into_iter())
            .for_each(|(t1, t2)| t1.extend(t2));
    }
}

impl<T: RandShiftTree> Index<usize> for RandShiftForest<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.trees[index]
    }
}

impl<T: RandShiftTree> IndexMut<usize> for RandShiftForest<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.trees[index]
    }
}

pub type RSF = RandShiftForest<RST>;
pub type RSQF = RandShiftForest<RSQT>;
