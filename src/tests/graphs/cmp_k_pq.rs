use std::{error::Error, fmt::Write, path::Path};

use crate::{prelude::*, tests::utils::save_txt};

use super::{
    bench::{bench_rsf_split, Mode},
    input::input,
    ROOT,
};

fn run<P: AsRef<Path>>(path: P) -> Result<String, Box<dyn Error>> {
    let mut out = String::new();
    writeln!(out, "K\tp=q\trocauc\tprauc\tf-measure")?;

    for k in [20, 50, 100, 200] {
        for p in [0.01, 0.05, 0.1, 0.2] {
            let sl_cfg = SpotLightConfig::new(k, p, p);
            let rsf_cb = ConfigBuilder::default().n_trees(50).n_points(256);

            let (graphs, y_true) = input(&path, 3600, 60);
            let res = bench_rsf_split(&graphs, &y_true, &rsf_cb, &sl_cfg, Mode::Normal);

            write!(out, "{k}\t{p}\t")?;
            write!(out, "{:.2} ({:.2})\t", res.means[0], res.stds[0])?;
            write!(out, "{:.2} ({:.2})\t", res.means[1], res.stds[1])?;
            writeln!(out, "{:.2} ({:.2})", res.means[2], res.stds[2])?;
        }
    }

    Ok(out)
}

#[test]
fn cmp_k_pq() {
    let out = run("in/graph/darpa.csv").unwrap();
    save_txt(ROOT, "cmp_K_pq", &out);
}
