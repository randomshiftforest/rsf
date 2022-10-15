use crate::prelude::ConfigBuilder;
use lazy_static::lazy_static;

mod bench;
mod cmp_machine_sketch_sizes;
mod cmp_sample_sketch_sizes;

lazy_static! {
    static ref BASE_CB: ConfigBuilder = {
        ConfigBuilder::default()
            .n_trees(32)
            .n_points(256)
            .granularity(4)
    };
}

const N_REPETITIONS: usize = 32;
const MACHINE_SIZES: [usize; 3] = [1, 8, 16];
const SKETCH_SIZES: [usize; 4] = [1, 2, 4, 8];
const SAMPLE_SIZES: [usize; 5] = [128, 256, 512, 1024, 2048];
