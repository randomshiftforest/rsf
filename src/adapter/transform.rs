use crate::algorithm::config::Config;

pub struct Transform<I> {
    iter: I,
    cn: f32,
}

impl<I> Transform<I> {
    fn new(iter: I, cfg: &Config) -> Self {
        let h = |i: f32| i.ln() + 0.577_215_7;
        let c = |n: f32| 2.0 * h(n - 1.0) - (2.0 * (n - 1.0) / n);
        let cn = c(cfg.n_points as f32);
        Self { iter, cn }
    }
}

impl<I: Iterator<Item = f32>> Iterator for Transform<I> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|s| (-s / self.cn).exp2())
    }
}

pub trait TransformIter: Iterator<Item = f32> + Sized {
    fn transform(self, cfg: &Config) -> Transform<Self> {
        Transform::new(self, cfg)
    }
}

impl<I: Iterator<Item = f32>> TransformIter for I {}
