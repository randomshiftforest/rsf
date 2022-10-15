use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use num::Float;
use rand::Rng;

use super::{aabb::AABB, tree::RandShiftTree};

#[derive(Debug)]
pub struct RandShiftForest<F: Float> {
    trees: Vec<RandShiftTree<F>>,
}

impl<F: 'static + Float> RandShiftForest<F> {
    pub fn new_using<R: Rng>(aabb: &AABB<F>, n_trees: usize, n_points: usize, rng: &mut R) -> Self {
        let trees: Vec<_> = (0..n_trees)
            .map(|_i| RandShiftTree::new_using(aabb, n_points, rng))
            .collect();
        Self { trees }
    }

    pub fn insert<S: Data<Elem = F>>(&mut self, p: &ArrayBase<S, Ix1>) {
        self.trees.iter_mut().for_each(|t| t.insert(p))
    }

    pub fn batch_insert<S: Data<Elem = F>>(&mut self, ps: &ArrayBase<S, Ix2>) {
        self.trees.iter_mut().for_each(|t| t.batch_insert(ps));
    }

    pub fn delete<S: Data<Elem = F>>(&mut self, p: &ArrayBase<S, Ix1>) {
        self.trees.iter_mut().for_each(|t| t.delete(p))
    }

    pub fn score<S: Data<Elem = F>>(&self, p: &ArrayBase<S, Ix1>) -> f32 {
        self.trees.iter().map(|t| t.score(p)).sum::<f32>() / (self.trees.len() as f32)
    }

    pub fn batch_score<S: Data<Elem = F>>(&self, ps: &ArrayBase<S, Ix2>) -> Array1<f32> {
        let zero = Array1::from_elem(ps.dim().0, 0.);
        let sum = self
            .trees
            .iter()
            .map(|t| t.batch_score(ps))
            .fold(zero, |sum, scores| sum + scores);
        sum / (self.trees.len() as f32)
    }
}
