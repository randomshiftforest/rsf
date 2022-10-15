use std::hash::Hash;

use ndarray::Array1;

use super::hash_picker::HashPicker;

#[derive(Clone)]
pub struct Graph<S, D> {
    edges: Vec<(S, D, f32)>,
}

impl<S, D> Graph<S, D> {
    pub fn new(edges: Vec<(S, D, f32)>) -> Self {
        Self { edges }
    }

    pub fn edge_iter(&self) -> std::slice::Iter<(S, D, f32)> {
        self.edges.iter()
    }
}

pub struct SpotLightConfig {
    pub k: usize,
    pub p: f64,
    pub q: f64,
}

impl SpotLightConfig {
    pub fn new(k: usize, p: f64, q: f64) -> Self {
        Self { k, p, q }
    }
}

pub struct SpotLight<I> {
    iter: I,
    k: usize,
    src_pickers: Vec<HashPicker>,
    dst_pickers: Vec<HashPicker>,
}

impl<I> SpotLight<I> {
    fn new(iter: I, cfg: &SpotLightConfig) -> Self {
        let src_pickers = (0..cfg.k).map(|_| HashPicker::from_prob(cfg.p)).collect();
        let dst_pickers = (0..cfg.k).map(|_| HashPicker::from_prob(cfg.q)).collect();
        Self {
            iter,
            k: cfg.k,
            src_pickers,
            dst_pickers,
        }
    }

    fn sketch<S: Hash, D: Hash>(&self, g: Graph<S, D>) -> Array1<f32> {
        let mut v = Array1::zeros(self.k);
        for (src, dst, w) in g.edges {
            for i in 0..self.k {
                if self.src_pickers[i].picks(&src) && self.dst_pickers[i].picks(&dst) {
                    v[i] += w;
                }
            }
        }
        v
    }
}

impl<S, D, I> Iterator for SpotLight<I>
where
    S: Hash,
    D: Hash,
    I: Iterator<Item = Graph<S, D>>,
{
    type Item = Array1<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|g| self.sketch(g))
    }
}

pub trait SpotLightIter<S: Hash, D: Hash>: Iterator<Item = Graph<S, D>> + Sized {
    fn spotlight(self, cfg: &SpotLightConfig) -> SpotLight<Self> {
        SpotLight::new(self, cfg)
    }
}

impl<S: Hash, D: Hash, I: Iterator<Item = Graph<S, D>>> SpotLightIter<S, D> for I {}
