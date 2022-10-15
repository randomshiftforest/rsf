use crate::prelude::ConfigBuilder;
use lazy_static::lazy_static;

mod forest_heatmap;
mod forest_scores;
mod forest_splits;
mod sine_wave;
mod tree_splits;
mod two_blobs;

lazy_static! {
    static ref BASE_CB: ConfigBuilder = {
        ConfigBuilder::default()
            .n_trees(64)
            .n_points(1024)
            .granularity(4)
    };
}

pub const ROOT: &str = "out/examples";
