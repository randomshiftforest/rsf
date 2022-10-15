use crate::prelude::*;
use ndarray::prelude::*;
use rand::prelude::*;

pub struct ParStreamSampler {
    us: Array1<f64>,
    u: f64,
    sample: Vec<(Array1<f32>, f64)>,
    rng: ThreadRng,
    n: usize,
}

impl ParStreamSampler {
    pub fn new(cfg: &Config) -> Self {
        Self {
            us: Array1::ones(cfg.n_machines),
            u: 1.0,
            sample: Vec::new(),
            rng: thread_rng(),
            n: cfg.n_points,
        }
    }

    pub fn insert(&mut self, m: usize, point: &Array1<f32>) {
        let w = self.rng.gen_range(0.0..1.0);
        if w < self.us[m] {
            self.us[m] = self.update(point, w);
        }
    }

    fn update(&mut self, point: &Array1<f32>, w: f64) -> f64 {
        if w < self.u {
            let i = self
                .sample
                .binary_search_by(|probe| probe.1.total_cmp(&w))
                .unwrap_or_else(|i| i);
            self.sample.insert(i, (point.clone(), w));
            if self.sample.len() > self.n {
                self.sample.pop();
                self.u = self.sample.last().unwrap().1;
            }
        }
        self.u
    }

    pub fn query(&self, cfg: &Config) -> Array2<f32> {
        let (n, d) = (self.sample.len(), cfg.bb.d());
        self.sample
            .iter()
            .flat_map(|p| p.0.to_owned())
            .collect::<Array1<f32>>()
            .into_shape((n, d))
            .unwrap()
    }
}
