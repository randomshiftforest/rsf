use super::{
    bench::{bench_one_way_coordinator, bench_two_way_par_streams},
    BASE_CB, MACHINE_SIZES, SKETCH_SIZES,
};
use crate::{
    prelude::*,
    tests::utils::{read_npz, run_globs, save_txt, BenchRes},
};
use ndarray::prelude::*;
use std::{error::Error, fmt::Write, path::PathBuf};

fn run<B>(bencher: B, name: &str, path: PathBuf) -> Result<String, Box<dyn Error>>
where
    B: Fn(&Array2<f32>, &Array1<bool>, &ConfigBuilder) -> BenchRes,
{
    let mut out = String::new();
    writeln!(out, "--- {name} ---")?;
    writeln!(out, "m\ts\tsize\tpr")?;

    let (x, y_true) = read_npz(path);
    let bb = x.outer_iter().bb().ok_or("no bounding box")?;
    let base_cb = BASE_CB.clone().bounding_box(bb);

    for m in MACHINE_SIZES {
        for s in SKETCH_SIZES {
            let cb = base_cb.clone().n_machines(m).sketch_size(s);
            let res = bencher(&x, &y_true, &cb);
            write!(out, "{}\t{}\t", m, s)?;
            write!(out, "{:.2} ({:.2})\t", res.means[0], res.stds[0])?;
            writeln!(out, "{:.2} ({:.2})", res.means[1], res.stds[1])?;
        }
    }

    Ok(out)
}

fn run_one_way_coordinator(name: &str, path: PathBuf) -> Result<String, Box<dyn Error>> {
    run(bench_one_way_coordinator, name, path)
}

fn run_two_way_par_streams(name: &str, path: PathBuf) -> Result<String, Box<dyn Error>> {
    run(bench_two_way_par_streams, name, path)
}

#[test]
fn cmp_machine_sketch_sizes() {
    let globs = ["in/toy/*.npz", "in/real/*.npz"];

    let out = run_globs(run_one_way_coordinator, &globs).concat();
    save_txt("out/one_way_coordinator", "cmp_machine_sketch_sizes", &out);

    let out = run_globs(run_two_way_par_streams, &globs).concat();
    save_txt("out/two_way_par_streams", "cmp_machine_sketch_sizes", &out);
}
