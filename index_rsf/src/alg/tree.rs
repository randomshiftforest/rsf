use std::{mem, num::NonZeroUsize};

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use num::Float;
use rand::Rng;

use super::{aabb::AABB, node::RandShiftNode, rotation::random_rotation_using};

#[derive(Debug, Clone)]
pub struct RandShiftTree<F: Float> {
    aabb: AABB<F>,
    nodes: Vec<RandShiftNode<F>>,
    max_depth: usize,
    max_points: usize,
    shift: Array1<F>,
    rotation: Array2<F>,
    split_dims: Array1<usize>,
}

impl<F: 'static + Float> RandShiftTree<F> {
    pub fn new_using<R: Rng>(aabb: &AABB<F>, n_points: usize, rng: &mut R) -> Self {
        let max_depth = 2 * (n_points as f32).log2() as usize;
        let max_points = 1;

        // splits
        let d = aabb.d();
        let split_dims = Array1::random_using(max_depth, Uniform::new(0, d), rng);

        // rotation
        let rotation = random_rotation_using(d, rng).map(|&v| F::from(v).unwrap());
        let mut aabb = aabb.clone();
        aabb.rotate(&rotation);

        // shift
        let r = aabb.range();
        let shift = Array1::<f64>::random_using(d, Uniform::new_inclusive(0., 1.), rng)
            .map(|&v| F::from(v).unwrap())
            * &r;
        aabb.add_assign_ub(&r);

        // root
        let split_dim = split_dims[0];
        let split_val = aabb.split_val_at(split_dim);
        let root = RandShiftNode {
            split: (split_dim, split_val),
            points: Vec::new(),
            parent_idx: None,
            left_child_idx: None,
            right_child_idx: None,
        };
        let mut nodes = Vec::with_capacity(n_points);
        nodes.push(root);

        Self {
            aabb,
            nodes,
            max_depth,
            max_points,
            shift,
            rotation,
            split_dims,
        }
    }

    pub fn batch_insert<S: Data<Elem = F>>(&mut self, x: &ArrayBase<S, Ix2>) {
        x.outer_iter().for_each(|p| self.insert(&p));
    }

    pub fn insert<S: Data<Elem = F>>(&mut self, p: &ArrayBase<S, Ix1>) {
        let p = p.dot(&self.rotation) + &self.shift;
        let mut aabb = self.aabb.clone();
        let mut node_idx = 0;
        let mut depth = 1;

        while self.nodes[node_idx].is_parent() {
            let child_idx = self.nodes[node_idx].child_idx(&p).unwrap();
            self.nodes[node_idx].cut(&mut aabb, &p);
            node_idx = child_idx.get();
            depth += 1;
        }

        while depth < self.max_depth && self.nodes[node_idx].n_points() >= self.max_points {
            self.split(node_idx, depth, &aabb);
            let child_idx = self.nodes[node_idx].child_idx(&p).unwrap();
            self.nodes[node_idx].cut(&mut aabb, &p);
            node_idx = child_idx.get();
            depth += 1;
        }

        self.nodes[node_idx].insert(p);
    }

    fn split(&mut self, node_idx: usize, depth: usize, aabb: &AABB<F>) {
        let (split_dim, split_val) = self.nodes[node_idx].split;
        let points = mem::take(&mut self.nodes[node_idx].points);
        let (left_points, right_points): (Vec<_>, Vec<_>) =
            points.into_iter().partition(|p| p[split_dim] <= split_val);
        let next_split_dim = self.split_dims[depth];
        let next_split_val = aabb.split_val_at(next_split_dim);

        let left_node = RandShiftNode {
            split: (next_split_dim, next_split_val),
            points: left_points,
            parent_idx: Some(node_idx),
            left_child_idx: None,
            right_child_idx: None,
        };
        let right_node = RandShiftNode {
            split: (next_split_dim, next_split_val),
            points: right_points,
            parent_idx: Some(node_idx),
            left_child_idx: None,
            right_child_idx: None,
        };
        self.nodes[node_idx].left_child_idx = Some(self.insert_node(left_node));
        self.nodes[node_idx].right_child_idx = Some(self.insert_node(right_node));
    }

    fn insert_node(&mut self, node: RandShiftNode<F>) -> NonZeroUsize {
        let new_idx = NonZeroUsize::new(self.nodes.len()).unwrap();
        self.nodes.push(node);
        new_idx
    }

    pub fn batch_score<S: Data<Elem = F>>(&self, x: &ArrayBase<S, Ix2>) -> Array1<f32> {
        x.outer_iter().map(|p| self.score(&p)).collect()
    }

    pub fn delete<S: Data<Elem = F>>(&mut self, p: &ArrayBase<S, Ix1>) {
        let p = p.dot(&self.rotation) + &self.shift;
        let mut node = &mut self.nodes[0];
        while node.is_parent() {
            let child_idx = node.child_idx(&p).unwrap();
            node = &mut self.nodes[child_idx.get()];
        }
        node.delete(&p);
        while let Some(parent_idx) = node.parent_idx {
            if self.n_leaf(parent_idx) <= self.max_points {
                self.contract(parent_idx);
            }
            node = &mut self.nodes[parent_idx];
        }
    }

    fn n_leaf(&self, parent_idx: usize) -> usize {
        let parent = &self.nodes[parent_idx];
        let left_child = &self.nodes[parent.left_child_idx.unwrap().get()];
        let right_child = &self.nodes[parent.right_child_idx.unwrap().get()];
        left_child.n_points() + right_child.n_points()
    }

    fn swap_remove(&mut self, child_idx: NonZeroUsize) -> RandShiftNode<F> {
        let last_node_idx = NonZeroUsize::new(self.nodes.len() - 1).unwrap();
        let last_node_parent_idx = self.nodes[last_node_idx.get()].parent_idx.unwrap();
        let parent = &mut self.nodes[last_node_parent_idx];
        let left_child_idx = parent.left_child_idx.as_mut().unwrap();
        if *left_child_idx == last_node_idx {
            *left_child_idx = child_idx;
        }
        let right_child_idx = parent.right_child_idx.as_mut().unwrap();
        if *right_child_idx == last_node_idx {
            *right_child_idx = child_idx;
        }
        self.nodes.swap_remove(child_idx.get())
    }

    fn contract(&mut self, parent_idx: usize) {
        let left_child_idx = self.nodes[parent_idx].left_child_idx.unwrap();
        let right_child_idx = self.nodes[parent_idx].right_child_idx.unwrap();
        let mut left_child = self.swap_remove(left_child_idx);
        let mut right_child = self.swap_remove(right_child_idx);
        let parent = &mut self.nodes[parent_idx];
        parent.points.append(&mut left_child.points);
        parent.points.append(&mut right_child.points);
        parent.left_child_idx = None;
        parent.right_child_idx = None;
    }

    pub fn score<S: Data<Elem = F>>(&self, p: &ArrayBase<S, Ix1>) -> f32 {
        let p = p.dot(&self.rotation) + &self.shift;
        let mut node = &self.nodes[0];
        let mut path_length = 1;
        while node.is_parent() {
            let child_idx = node.child_idx(&p).unwrap();
            node = &self.nodes[child_idx.get()];
            path_length += 1;
        }
        if path_length + 1 == self.max_depth {
            let weight = node.n_points();
            if weight > self.max_points {
                return (path_length as f32) + avg_bst_path_length(weight);
            }
        }
        path_length as f32
    }
}

fn avg_bst_path_length(n: usize) -> f32 {
    let h = |i: f32| i.ln() + 0.577_215_7;
    let c = |n: f32| 2.0 * h(n - 1.0) - (2.0 * (n - 1.0) / n);
    c(n as f32)
}
