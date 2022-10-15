use std::num::NonZeroUsize;

use ndarray::Array1;
use num::Float;

use super::aabb::AABB;

#[derive(Debug, Clone)]
pub struct RandShiftNode<F: Float> {
    pub points: Vec<Array1<F>>,
    pub split: (usize, F),
    pub parent_idx: Option<usize>,
    pub left_child_idx: Option<NonZeroUsize>,
    pub right_child_idx: Option<NonZeroUsize>,
}

impl<F: 'static + Float> RandShiftNode<F> {
    pub fn cut(&self, aabb: &mut AABB<F>, point: &Array1<F>) {
        if point[self.split.0] <= self.split.1 {
            aabb.keep_left(self.split)
        } else {
            aabb.keep_right(self.split)
        }
    }

    /// Returns the index of the child node whose subtree would contain the point.
    pub fn child_idx(&self, point: &Array1<F>) -> Option<NonZeroUsize> {
        if point[self.split.0] <= self.split.1 {
            self.left_child_idx
        } else {
            self.right_child_idx
        }
    }

    /// Adds the point to the list of points of this node.
    pub fn insert(&mut self, point: Array1<F>) {
        self.points.push(point);
    }

    /// Deletes the point from the list of points of this node.
    pub fn delete(&mut self, point: &Array1<F>) {
        let point_idx = self.points.iter().position(|p| p == point).unwrap();
        self.points.swap_remove(point_idx);
    }

    /// Returns `true` if the node has child nodes.
    pub fn is_parent(&self) -> bool {
        self.left_child_idx.is_some()
    }

    /// Returns the number of points of this node.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }
}
