use std::mem;

use ndarray::Array1;

use super::bounding_box::BoundingBox;

pub struct Point {
    pub coords: Array1<f32>,
    pub weight: usize,
}

impl Point {
    fn new(coords: Array1<f32>) -> Self {
        Self { coords, weight: 1 }
    }
}

pub struct PointList(pub Vec<Point>);

impl PointList {
    pub fn new() -> Self {
        Self::with_points(Vec::new())
    }

    pub fn with_points(points: Vec<Point>) -> Self {
        Self(points)
    }

    pub fn n_points(&self) -> usize {
        self.0.len()
    }

    pub fn weight(&self) -> usize {
        self.0.iter().map(|p| p.weight).sum()
    }

    pub fn remove(&mut self, coords: &Array1<f32>) {
        if let Some(i) = self.0.iter().position(|p| p.coords == coords) {
            if self.0[i].weight == 1 {
                self.0.swap_remove(i);
            } else {
                self.0[i].weight -= 1;
            }
        }
    }

    pub fn insert(&mut self, coords: Array1<f32>) {
        if let Some(i) = self.0.iter().position(|point| point.coords == coords) {
            self.0[i].weight += 1;
        } else {
            self.0.push(Point::new(coords));
        }
    }

    pub fn split_off(&mut self, bb: &BoundingBox) -> Self {
        let (in_bb, out_bb) = mem::take(&mut self.0)
            .into_iter()
            .partition(|p| bb.contains(&p.coords));
        self.0 = out_bb;
        Self(in_bb)
    }

    pub fn partition_at(self, bb: &BoundingBox, dim: usize) -> (Self, Self) {
        let (in_bb, out_bb) = self
            .0
            .into_iter()
            .partition(|p| bb.contains_at(&p.coords, dim));
        (Self(in_bb), Self(out_bb))
    }

    pub fn sketch(&mut self, sketch_size: usize) {
        let n_points = self.n_points();
        let n_excess = n_points.saturating_sub(sketch_size);
        if n_excess > 0 {
            for j in sketch_size..n_points {
                let i = j % sketch_size;
                self.0[i].weight += self.0[j].weight;
            }
        }
        self.0.truncate(sketch_size);
        self.0.shrink_to_fit();
    }
}
