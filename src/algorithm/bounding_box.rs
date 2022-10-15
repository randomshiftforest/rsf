use std::iter::repeat;

use ndarray::{concatenate, stack, Array1, Array2, ArrayBase, ArrayViewMut1, Axis, Data, Ix1};
use ndarray_rand::RandomExt;
use rand::{distributions::Slice, Rng};
use rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub bounds: Array2<f32>,
}

impl BoundingBox {
    pub fn new(bounds: Array2<f32>) -> Self {
        Self { bounds }
    }

    pub fn unit(d: usize) -> Self {
        let min = Array1::from_elem(d, 0.0);
        let max = Array1::from_elem(d, 1.0);
        let bounds = stack![Axis(1), min, max];
        Self { bounds }
    }

    pub fn with_double_range(bb: &Self) -> Self {
        let mut bb2 = bb.clone();
        for (i, r) in bb.range().into_iter().enumerate() {
            bb2.bounds[(i, 1)] += r;
        }
        bb2
    }

    pub fn shingle(&mut self, s: usize) {
        self.bounds = concatenate(
            Axis(0),
            repeat(self.bounds.view())
                .take(s)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap();
    }

    pub fn normalise(&self, mut p: ArrayViewMut1<f32>) {
        p -= &self.bounds.column(0);
        for (v, r) in p.iter_mut().zip(self.range()) {
            if r != 0. {
                *v /= r
            }
        }
    }

    pub fn range(&self) -> Array1<f32> {
        &self.bounds.column(1) - &self.bounds.column(0)
    }

    pub fn d(&self) -> usize {
        self.bounds.dim().0
    }

    pub fn split_at(&self, dim: usize) -> [Self; 2] {
        let lb = self.bounds[(dim, 0)];
        let ub = self.bounds[(dim, 1)];
        let split_val = lb + (ub - lb) / 2.0;

        let mut left_bb = self.clone();
        let mut right_bb = self.clone();
        left_bb.bounds[(dim, 1)] = split_val;
        right_bb.bounds[(dim, 0)] = split_val;

        [left_bb, right_bb]
    }

    pub fn split_line_at(&self, dim: usize) -> (Array1<f32>, Array1<f32>) {
        let mut p1 = self.bounds.column(0).to_owned();
        let mut p2 = self.bounds.column(1).to_owned();
        let split_val = p1[dim] + (p2[dim] - p1[dim]) / 2.0;
        p1[dim] = split_val;
        p2[dim] = split_val;
        (p1, p2)
    }

    pub fn split_xy(&self) -> [Self; 4] {
        let [l, r] = self.split_at(0);
        let [tl, bl] = l.split_at(1);
        let [tr, br] = r.split_at(1);
        [tl, tr, bl, br]
    }

    pub fn contains(&self, p: &Array1<f32>) -> bool {
        self.bounds
            .axis_iter(Axis(0))
            .zip(p.iter())
            .all(|(bound, &coord)| bound[0] <= coord && coord <= bound[1])
    }

    pub fn contains_at(&self, p: &Array1<f32>, dim: usize) -> bool {
        self.bounds[(dim, 0)] <= p[dim] && p[dim] <= self.bounds[(dim, 1)]
    }

    pub fn gen_shift_using<R: Rng>(&self, rng: &mut R) -> Array1<f32> {
        let d = self.d();
        let range = self.range();
        Array1::random_using(d, Uniform::new(0.0, 1.0), rng) * range
    }

    pub fn gen_splits_using<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize> {
        let nonzero: Vec<_> = self
            .range()
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| if r == 0. { None } else { Some(i) })
            .collect();
        let distr = Slice::new(&nonzero).unwrap();
        rng.sample_iter(distr).take(n).cloned().collect()
    }

    // pub fn gen_splits_using_alt<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize> {
    //     let range_sum = self.range().into_iter().sum::<f32>();
    //     rng.sample_iter(Uniform::new(0.0, range_sum))
    //         .map(|f| {
    //             let mut sum = 0.0;
    //             let mut res = 0;
    //             for (i, r) in self.range().iter().enumerate() {
    //                 sum += r;
    //                 if f < sum {
    //                     res = i;
    //                     break;
    //                 }
    //             }
    //             res
    //         })
    //         .take(n)
    //         .collect()
    // }
}

pub trait BoundingBoxIter<S: Data<Elem = f32>>: Iterator<Item = ArrayBase<S, Ix1>> + Sized {
    fn bb(mut self) -> Option<BoundingBox> {
        self.next().map(|p0| {
            let mut bounds = stack![Axis(1), p0, p0];
            for p in self {
                for (mut bound, &v) in bounds.outer_iter_mut().zip(&p) {
                    bound[0] = f32::min(bound[0], v);
                    bound[1] = f32::max(bound[1], v);
                }
            }
            BoundingBox::new(bounds)
        })
    }
}

impl<S: Data<Elem = f32>, I: Iterator<Item = ArrayBase<S, Ix1>>> BoundingBoxIter<S> for I {}
