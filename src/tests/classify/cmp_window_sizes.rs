use std::{error::Error, fmt::Write, path::PathBuf};

use crate::{
    prelude::*,
    tests::{
        classify::WINDOW_SIZES,
        utils::{read_npz, run_globs, save_txt},
    },
};

use super::{bench::bench_rsf_win, BASE_CB, ROOT};

fn run(name: &str, path: PathBuf) -> Result<String, Box<dyn Error>> {
    let mut out = String::new();
    writeln!(out, "--- {name} ---")?;
    writeln!(out, "window size\trocauc\tprauc\tf-measure")?;

    let (mut x, y_true) = read_npz(&path);
    let bb = x.outer_iter().bb().unwrap();
    x.outer_iter_mut().normalise(&bb);
    let bb = x.outer_iter().bb().unwrap();
    let cb = BASE_CB.clone().bounding_box(bb);

    for w in WINDOW_SIZES.into_iter().filter(|&w| w < x.dim().0) {
        let cfg = cb.clone().window(w).build();
        let res = bench_rsf_win(&x, &y_true, &cfg);
        write!(out, "{w}\t")?;
        write!(out, "{:.2} ({:.2})\t", res.means[0], res.stds[0])?;
        write!(out, "{:.2} ({:.2})\t", res.means[1], res.stds[1])?;
        writeln!(out, "{:.2} ({:.2})", res.means[2], res.stds[2])?;
    }

    Ok(out)
}

#[test]
fn cmp_window_sizes_toy() {
    let out = run_globs(run, &["in/toy/*.npz"]).concat();
    save_txt(ROOT, "cmp_window_sizes_toy", &out);
}

#[test]
fn cmp_window_sizes_real() {
    let out = run_globs(run, &["in/real/*.npz"]).concat();
    save_txt(ROOT, "cmp_window_sizes_real", &out);
}
