use lazy_static::lazy_static;

use crate::prelude::ConfigBuilder;

mod bench;
mod cmp_algorithms;
mod cmp_window_sizes;
mod utils;

lazy_static! {
    static ref BASE_CB: ConfigBuilder = {
        ConfigBuilder::default()
            .n_trees(32)
            .n_points(256)
            .granularity(4)
    };
}

const N_REPETITIONS: usize = 32;
const ROOT: &str = "out/classify";
// const WINDOW_SIZES: [usize; 6] = [1000, 2000, 5000, 10000, 20000, 50000];
const WINDOW_SIZES: [usize; 1] = [1000];
