use std::mem;

use ndarray::Array1;

use super::{
    bounding_box::BoundingBox,
    point_list::{Point, PointList},
};

pub trait RandShiftNode
where
    Self: Sized,
{
    fn level(&self) -> usize;
    fn bb(&self) -> &BoundingBox;
    fn children(&self) -> &Vec<Self>;
    fn children_mut(&mut self) -> &mut Vec<Self>;
    fn into_children(self) -> Vec<Self>;
    fn point_list(&self) -> &PointList;
    fn point_list_mut(&mut self) -> &mut PointList;
    fn into_point_list(self) -> PointList;
    fn child(&self, p: &Array1<f32>) -> Option<&Self>;
    fn child_mut(&mut self, p: &Array1<f32>) -> Option<&mut Self>;

    fn depth(&self) -> usize {
        self.level() + 1
    }

    fn path_length(&self) -> usize {
        self.level()
    }

    fn is_leaf(&self) -> bool {
        self.children().is_empty()
    }

    fn n_points(&self) -> usize {
        self.point_list().n_points()
    }

    fn weight(&self) -> usize {
        self.point_list().weight()
    }

    fn insert(&mut self, coords: Array1<f32>) {
        self.point_list_mut().insert(coords);
    }

    fn remove(&mut self, coords: &Array1<f32>) {
        self.point_list_mut().remove(coords);
    }

    fn sketch(&mut self, sketch_size: usize) {
        self.point_list_mut().sketch(sketch_size);
        self.children_mut()
            .iter_mut()
            .for_each(|c| c.sketch(sketch_size));
    }

    fn find(&self, p: &Array1<f32>) -> &Self {
        let mut node = self;
        while !node.is_leaf() {
            node = node.child(p).unwrap();
        }
        node
    }

    fn find_mut(&mut self, p: &Array1<f32>) -> &mut Self {
        let mut node = self;
        while !node.is_leaf() {
            node = node.child_mut(p).unwrap();
        }
        node
    }

    fn contract(&mut self) {
        let children = mem::take(self.children_mut());
        let points: Vec<Point> = children
            .into_iter()
            .flat_map(|c| c.into_point_list().0)
            .collect();
        let point_list = PointList::with_points(points);
        let _ = mem::replace(self.point_list_mut(), point_list);
    }

    fn can_contract(&self, max_points: usize) -> bool {
        let all_leaf = self.children().iter().all(|c| c.is_leaf());
        let weight_sum: usize = self.children().iter().map(|c| c.weight()).sum();
        all_leaf && weight_sum <= max_points
    }

    fn contract_full(&mut self, max_points: usize) {
        self.children_mut().iter_mut().for_each(|c| {
            c.contract_full(max_points);
        });
        if self.can_contract(max_points) {
            self.contract();
        }
    }

    fn contract_at(&mut self, p: &Array1<f32>, max_points: usize) {
        if let Some(child) = self.child_mut(p) {
            child.contract_at(p, max_points);
            if self.can_contract(max_points) {
                self.contract();
            }
        }
    }
}

pub struct RSQTNode {
    bb: BoundingBox,
    children: Vec<RSQTNode>,
    point_list: PointList,
    level: usize,
}

impl RSQTNode {
    pub fn root(bb: BoundingBox) -> Self {
        Self {
            bb,
            children: Vec::new(),
            point_list: PointList::new(),
            level: 0,
        }
    }

    pub fn split_xy(&mut self) {
        let mut point_list = mem::replace(&mut self.point_list, PointList::new());
        let bbs = self.bb.split_xy();
        self.children = bbs
            .into_iter()
            .map(|bb| {
                let in_bb = point_list.split_off(&bb);
                Self {
                    bb,
                    children: Vec::new(),
                    point_list: in_bb,
                    level: self.level + 1,
                }
            })
            .collect();
    }
}

impl RandShiftNode for RSQTNode {
    fn level(&self) -> usize {
        self.level
    }

    fn bb(&self) -> &BoundingBox {
        &self.bb
    }

    fn children(&self) -> &Vec<Self> {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<Self> {
        &mut self.children
    }

    fn into_children(self) -> Vec<Self> {
        self.children
    }

    fn point_list(&self) -> &PointList {
        &self.point_list
    }

    fn point_list_mut(&mut self) -> &mut PointList {
        &mut self.point_list
    }

    fn into_point_list(self) -> PointList {
        self.point_list
    }

    fn child(&self, p: &Array1<f32>) -> Option<&Self> {
        self.children.iter().find(|c| c.bb.contains(p))
    }

    fn child_mut(&mut self, p: &Array1<f32>) -> Option<&mut Self> {
        self.children.iter_mut().find(|c| c.bb.contains(p))
    }
}

pub struct RSTNode {
    bb: BoundingBox,
    children: Vec<RSTNode>,
    point_list: PointList,
    level: usize,
    split_dim: usize,
}

impl RSTNode {
    pub fn root(bb: BoundingBox, split_dim: usize) -> Self {
        Self {
            bb,
            children: Vec::new(),
            point_list: PointList::new(),
            level: 0,
            split_dim,
        }
    }

    pub fn split(&mut self, splits: &[usize]) {
        let point_list = mem::replace(&mut self.point_list, PointList::new());
        let [left_bb, right_bb] = self.bb.split_at(self.split_dim);
        let (in_left_bb, in_right_bb) = point_list.partition_at(&left_bb, self.split_dim);
        let next_level = self.level + 1;
        let next_split_dim = splits[next_level];
        self.children = vec![
            Self {
                bb: left_bb,
                children: Vec::new(),
                point_list: in_left_bb,
                level: next_level,
                split_dim: next_split_dim,
            },
            Self {
                bb: right_bb,
                children: Vec::new(),
                point_list: in_right_bb,
                level: next_level,
                split_dim: next_split_dim,
            },
        ];
    }
}

impl RandShiftNode for RSTNode {
    fn level(&self) -> usize {
        self.level
    }

    fn bb(&self) -> &BoundingBox {
        &self.bb
    }

    fn children(&self) -> &Vec<Self> {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<Self> {
        &mut self.children
    }

    fn into_children(self) -> Vec<Self> {
        self.children
    }

    fn point_list(&self) -> &PointList {
        &self.point_list
    }

    fn point_list_mut(&mut self) -> &mut PointList {
        &mut self.point_list
    }

    fn into_point_list(self) -> PointList {
        self.point_list
    }

    fn child(&self, p: &Array1<f32>) -> Option<&Self> {
        self.children
            .iter()
            .find(|c| c.bb.contains_at(p, self.split_dim))
    }

    fn child_mut(&mut self, p: &Array1<f32>) -> Option<&mut Self> {
        self.children
            .iter_mut()
            .find(|c| c.bb.contains_at(p, self.split_dim))
    }
}
