use std::{collections::VecDeque, iter::repeat};

use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use plotly::{
    layout::{Shape, ShapeLayer, ShapeLine, ShapeType},
    Layout,
};
use rand::Rng;

use crate::metric::exp_bst_path_length;

use super::{
    bounding_box::BoundingBox,
    config::Config,
    node::{RSQTNode, RSTNode, RandShiftNode},
};

pub trait RandShiftTree
where
    Self: Sized,
{
    type Node: RandShiftNode;

    fn from_config<R: Rng>(cfg: &Config, tree_i: usize, rng: &mut R) -> Self;
    fn root(&self) -> &Self::Node;
    fn root_mut(&mut self) -> &mut Self::Node;
    fn into_root(self) -> Self::Node;
    fn shift(&self) -> &Array1<f32>;
    fn max_depth(&self) -> usize;
    fn max_points(&self) -> usize;
    fn insert<S: Data<Elem = f32>>(&mut self, p: &ArrayBase<S, Ix1>);
    fn add_splits(&self, layout: &mut Layout);

    fn node_iter(&self) -> NodeIter<Self> {
        NodeIter::new(self)
    }

    fn n_points(&self) -> usize {
        self.node_iter().map(|node| node.n_points()).sum()
    }

    fn batch_insert<S: Data<Elem = f32>>(&mut self, x: &ArrayBase<S, Ix2>) {
        x.outer_iter().for_each(|p| self.insert(&p));
    }

    fn score<S: Data<Elem = f32>>(&self, p: &ArrayBase<S, Ix1>) -> f32 {
        let p_shift = p + self.shift();
        let node = self.root().find(&p_shift);
        if node.depth() == self.max_depth() {
            let weight = node.weight();
            if weight > self.max_points() {
                return (node.path_length() as f32) + exp_bst_path_length(weight);
            }
        }
        node.path_length() as f32
    }

    fn batch_score<S: Data<Elem = f32>>(&self, x: &ArrayBase<S, Ix2>) -> Array1<f32> {
        x.outer_iter().map(|p| self.score(&p)).collect()
    }

    fn remove<S: Data<Elem = f32>>(&mut self, p: &ArrayBase<S, Ix1>) {
        let max_points = self.max_points();
        let p_shift = p + self.shift();
        let root = self.root_mut();
        let node = root.find_mut(&p_shift);
        node.remove(&p_shift);
        root.contract_at(&p_shift, max_points);
    }

    fn extend(&mut self, other: Self) {
        for node in other.node_iter() {
            for point in &node.point_list().0 {
                let unshifted = &point.coords - self.shift();
                repeat(unshifted.view())
                    .take(point.weight)
                    .for_each(|p| self.insert(&p));
            }
        }
    }

    fn sketch(&mut self, sketch_size: usize) {
        self.root_mut().sketch(sketch_size);
    }
}

pub struct NodeIter<'a, T: RandShiftTree> {
    deque: VecDeque<&'a T::Node>,
}

impl<'a, T: RandShiftTree> NodeIter<'a, T> {
    fn new(tree: &'a T) -> Self {
        let mut deque = VecDeque::new();
        deque.push_back(tree.root());
        Self { deque }
    }
}

impl<'a, T: RandShiftTree> Iterator for NodeIter<'a, T> {
    type Item = &'a T::Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.deque.pop_front().map(|n| {
            self.deque.extend(n.children());
            n
        })
    }
}

pub struct RST {
    max_depth: usize,
    max_points: usize,
    root: RSTNode,
    splits: Vec<usize>,
    shift: Array1<f32>,
}

impl RandShiftTree for RST {
    type Node = RSTNode;

    fn from_config<R: Rng>(cfg: &Config, tree_i: usize, rng: &mut R) -> Self {
        let max_depth = cfg.max_depth();
        let max_points = cfg.max_points(tree_i);
        let shift = cfg.bb.gen_shift_using(rng);
        let splits = cfg.bb.gen_splits_using(max_depth, rng);
        let root = RSTNode::root(BoundingBox::with_double_range(&cfg.bb), splits[0]);

        Self {
            max_depth,
            max_points,
            root,
            splits,
            shift,
        }
    }

    fn root(&self) -> &Self::Node {
        &self.root
    }

    fn root_mut(&mut self) -> &mut Self::Node {
        &mut self.root
    }

    fn into_root(self) -> Self::Node {
        self.root
    }

    fn shift(&self) -> &Array1<f32> {
        &self.shift
    }

    fn max_depth(&self) -> usize {
        self.max_depth
    }

    fn max_points(&self) -> usize {
        self.max_points
    }

    fn insert<S: Data<Elem = f32>>(&mut self, p: &ArrayBase<S, Ix1>) {
        let p_shift = p + self.shift();
        let mut node = self.root.find_mut(&p_shift);
        loop {
            if node.depth() < self.max_depth && node.weight() == self.max_points {
                node.split(&self.splits);
                if let Some(c) = node.child_mut(&p_shift) {
                    node = c;
                } else {
                    println!("{:?}", p);
                    panic!("OOPS");
                }
            } else {
                node.insert(p_shift);
                break;
            }
        }
    }

    fn add_splits(&self, layout: &mut Layout) {
        for node in self.node_iter() {
            if !node.is_leaf() {
                let split_dim = self.splits[node.depth() - 1];
                let (p0, p1) = node.bb().split_line_at(split_dim);
                let (x0, y0) = (p0[0], p0[1]);
                let (x1, y1) = (p1[0], p1[1]);
                layout.add_shape(
                    Shape::new()
                        .shape_type(ShapeType::Line)
                        .layer(ShapeLayer::Below)
                        .line(ShapeLine::new().width(1.0))
                        .x0(x0 as f64)
                        .y0(y0 as f64)
                        .x1(x1 as f64)
                        .y1(y1 as f64),
                );
            }
        }
    }
}

pub struct RSQT {
    max_depth: usize,
    max_points: usize,
    root: RSQTNode,
    pub shift: Array1<f32>,
}

impl RSQT {}

impl RandShiftTree for RSQT {
    type Node = RSQTNode;

    fn from_config<R: Rng>(cfg: &Config, tree_i: usize, rng: &mut R) -> Self {
        let max_depth = cfg.max_depth() / 2;
        let max_points = cfg.max_points(tree_i);
        let shift = cfg.bb.gen_shift_using(rng);
        let root = RSQTNode::root(BoundingBox::with_double_range(&cfg.bb));

        Self {
            max_depth,
            max_points,
            root,
            shift,
        }
    }

    fn root(&self) -> &Self::Node {
        &self.root
    }

    fn root_mut(&mut self) -> &mut Self::Node {
        &mut self.root
    }

    fn into_root(self) -> Self::Node {
        self.root
    }

    fn shift(&self) -> &Array1<f32> {
        &self.shift
    }

    fn max_depth(&self) -> usize {
        self.max_depth
    }

    fn max_points(&self) -> usize {
        self.max_points
    }

    fn insert<S: ndarray::Data<Elem = f32>>(&mut self, p: &ndarray::ArrayBase<S, ndarray::Ix1>) {
        let p_shift = p + self.shift();
        let mut node = self.root.find_mut(&p_shift);
        loop {
            if node.depth() < self.max_depth && node.weight() == self.max_points {
                node.split_xy();
                node = node.child_mut(&p_shift).unwrap();
            } else {
                node.insert(p_shift);
                break;
            }
        }
    }

    fn add_splits(&self, layout: &mut Layout) {
        for node in self.node_iter() {
            if !node.is_leaf() {
                for split_dim in [0, 1] {
                    let (p0, p1) = node.bb().split_line_at(split_dim);
                    let (x0, y0) = (p0[0], p0[1]);
                    let (x1, y1) = (p1[0], p1[1]);
                    layout.add_shape(
                        Shape::new()
                            .shape_type(ShapeType::Line)
                            .layer(ShapeLayer::Below)
                            .x0(x0 as f64)
                            .y0(y0 as f64)
                            .x1(x1 as f64)
                            .y1(y1 as f64),
                    );
                }
            }
        }
    }
}
