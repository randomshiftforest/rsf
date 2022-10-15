use ndarray::{stack, Array1, Array2, Axis};
use num::Float;

#[derive(Debug, Clone)]
pub struct AABB<F: Float>(Array2<F>);

impl<F: 'static + Float> AABB<F> {
    pub fn unit(d: usize) -> Self {
        let lb = Array1::from_elem(d, F::zero());
        let ub = Array1::from_elem(d, F::one());
        let bounds = stack![Axis(1), lb, ub];
        Self(bounds)
    }

    pub fn from_data(x: &Array2<F>) -> Self {
        let mut points = x.outer_iter();
        let p0 = points.next().unwrap();
        let mut bounds = stack![Axis(1), p0, p0];
        for p in points {
            for (mut bound, &v) in bounds.outer_iter_mut().zip(&p) {
                bound[0] = F::min(bound[0], v);
                bound[1] = F::max(bound[1], v);
            }
        }
        Self(bounds)
    }

    pub fn keep_left(&mut self, split: (usize, F)) {
        self.0[(split.0, 1)] = split.1;
    }

    pub fn keep_right(&mut self, split: (usize, F)) {
        self.0[(split.0, 0)] = split.1;
    }

    pub fn d(&self) -> usize {
        self.0.dim().0
    }

    pub fn range(&self) -> Array1<F> {
        &self.0.column(1) - &self.0.column(0)
    }

    pub fn add_assign_ub(&mut self, rhs: &Array1<F>) {
        let new_ub = &self.0.column(1) + rhs;
        self.0.column_mut(1).assign(&new_ub);
    }

    pub fn rotate(&mut self, rotation: &Array2<F>) {
        self.0.assign(&self.0.t().dot(rotation).t());
        // self.0 = rotation.dot(&self.0);
    }

    pub fn split_val_at(&self, split_dim: usize) -> F {
        let lb = self.0[(split_dim, 0)];
        let ub = self.0[(split_dim, 1)];
        lb + (ub - lb) / F::from(2.).unwrap()
    }
}
