mod bench;
mod cmp_dt_et;
mod cmp_k_pq;
pub mod input;
mod twitter;

const N_REPETITIONS: usize = 32;
pub const DT_VALS: [usize; 3] = [300, 1800, 3600];
pub const ET_VALS: [usize; 3] = [10, 30, 60];
const ROOT: &str = "out/graphs";
