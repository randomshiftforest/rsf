use std::{error::Error, fmt::Write, path::PathBuf};

use crate::{
    prelude::*,
    tests::utils::{read_npz, run_globs, save_txt},
};

use super::{
    bench::{bench_eif_split, bench_rrcf_split, bench_rsf_res, bench_rsf_split},
    BASE_CB, ROOT,
};

fn run(name: &str, path: PathBuf) -> Result<String, Box<dyn Error>> {
    let mut out = String::new();
    writeln!(out, "--- {name} ---")?;
    writeln!(out, "algorithm\trocauc\tprauc\tf-measure")?;

    let (mut x, y_true) = read_npz(&path);
    println!("name: {:?}", x.dim());
    let bb = x.outer_iter().bb().unwrap();
    x.outer_iter_mut().normalise(&bb);
    let bb = x.outer_iter().bb().unwrap();
    let cfg = BASE_CB.clone().bounding_box(bb).build();

    for (alg, res) in [
        (
            "iForest-split",
            bench_eif_split::<false>(&x, &y_true, &cfg, name),
        ),
        (
            "EIF-split",
            bench_eif_split::<true>(&x, &y_true, &cfg, name),
        ),
        ("RRCF-split", bench_rrcf_split(&x, &y_true, &cfg)),
        ("RSF-split", bench_rsf_split(&x, &y_true, &cfg)),
        ("RSF-reservoir", bench_rsf_res(&x, &y_true, &cfg)),
    ] {
        write!(out, "{alg}\t")?;
        write!(out, "{:.2} ({:.2})\t", res.means[0], res.stds[0])?;
        write!(out, "{:.2} ({:.2})\t", res.means[1], res.stds[1])?;
        writeln!(out, "{:.2} ({:.2})", res.means[2], res.stds[2])?;
    }

    Ok(out)
}

#[test]
fn cmp_algorithms_toy() {
    let out = run_globs(run, &["in/toy/*.npz"]).concat();
    save_txt(ROOT, "cmp_algorithms_toy", &out);
}

#[test]
fn cmp_algorithms_real() {
    let out = run_globs(run, &["in/real/*.npz"]).concat();
    save_txt(ROOT, "cmp_algorithms_real", &out);
}
