use rand::{prelude::StdRng, thread_rng, SeedableRng};

use super::bounding_box::BoundingBox;

#[derive(Clone)]
pub struct Config {
    pub bb: BoundingBox,
    pub n_trees: usize,
    pub n_points: usize,
    pub granularity: usize,
    pub window: usize,
    pub shingle: usize,
    pub seed: Option<u64>,
    pub sketch_size: usize,
    pub n_machines: usize,
}

impl Config {
    pub fn max_depth(&self) -> usize {
        (self.n_points as f64).log2().ceil() as usize
    }

    pub fn max_points(&self, tree_i: usize) -> usize {
        tree_i * self.granularity / self.n_trees + 1
    }

    pub fn get_rng(&self) -> StdRng {
        if let Some(seed) = self.seed {
            SeedableRng::seed_from_u64(seed)
        } else {
            SeedableRng::from_rng(thread_rng()).unwrap()
        }
    }
}

#[derive(Default, Clone)]
pub struct ConfigBuilder {
    bounding_box: Option<BoundingBox>,
    n_trees: Option<usize>,
    n_points: Option<usize>,
    granularity: Option<usize>,
    shingle: Option<usize>,
    window: Option<usize>,
    seed: Option<u64>,
    sketch_size: Option<usize>,
    n_machines: Option<usize>,
}

impl ConfigBuilder {
    pub fn bounding_box(mut self, bounding_box: BoundingBox) -> Self {
        self.bounding_box = Some(bounding_box);
        self
    }

    pub fn n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = Some(n_trees);
        self
    }

    pub fn n_points(mut self, n_points: usize) -> Self {
        self.n_points = Some(n_points);
        self
    }

    pub fn granularity(mut self, granularity: usize) -> Self {
        self.granularity = Some(granularity);
        self
    }

    pub fn shingle(mut self, shingle: usize) -> Self {
        self.shingle = Some(shingle);
        self
    }

    pub fn window(mut self, window: usize) -> Self {
        self.window = Some(window);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn sketch_size(mut self, sketch_size: usize) -> Self {
        self.sketch_size = Some(sketch_size);
        self
    }

    pub fn n_machines(mut self, n_machines: usize) -> Self {
        self.n_machines = Some(n_machines);
        self
    }

    pub fn build(&self) -> Config {
        let shingle = self.shingle.unwrap_or(1);
        let n_points = self.n_points.unwrap_or(128);
        let mut bb = self.bounding_box.clone().expect("no bounding box provided");
        bb.shingle(shingle);

        Config {
            bb,
            n_trees: self.n_trees.unwrap_or(64),
            n_points,
            granularity: self.granularity.unwrap_or(1),
            window: self.window.unwrap_or(n_points),
            shingle,
            seed: self.seed,
            sketch_size: self.sketch_size.unwrap_or(2),
            n_machines: self.n_machines.unwrap_or(2),
        }
    }
}
