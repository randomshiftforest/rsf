use super::{
    bench::{bench_rrcf_split, bench_rsf_split, Mode},
    input::input,
    DT_VALS, ET_VALS, ROOT,
};
use crate::{prelude::*, tests::utils::save_txt};
use std::{
    fmt::{self, Write},
    path::Path,
};

fn run<P: AsRef<Path>>(path: P, mode: Mode) -> Result<String, fmt::Error> {
    let mut rrcf_out = String::new();
    rrcf_out += "--- RRCF-split ---\n";
    rrcf_out += "dt\tet\trocauc\tprauc\tf-measure\n";

    let mut rsf_out = String::new();
    rsf_out += "--- RSF-split ---\n";
    rsf_out += "dt\tet\trocauc\tprauc\tf-measure\n";

    let sl_cfg = SpotLightConfig::new(50, 0.2, 0.2);
    let rsf_cb = ConfigBuilder::default().n_trees(50).n_points(256);

    for dt in DT_VALS {
        for et in ET_VALS {
            let (graphs, y_true) = input(&path, dt, et);

            let rrcf_res = bench_rrcf_split(&graphs, &y_true, &rsf_cb, &sl_cfg, mode);
            write!(rrcf_out, "{dt}\t{et}\t")?;
            write!(
                rrcf_out,
                "{:.2} ({:.2})\t",
                rrcf_res.means[0], rrcf_res.stds[0]
            )?;
            write!(
                rrcf_out,
                "{:.2} ({:.2})\t",
                rrcf_res.means[1], rrcf_res.stds[1]
            )?;
            writeln!(
                rrcf_out,
                "{:.2} ({:.2})",
                rrcf_res.means[2], rrcf_res.stds[2]
            )?;

            let rsf_res = bench_rsf_split(&graphs, &y_true, &rsf_cb, &sl_cfg, mode);
            write!(rsf_out, "{dt}\t{et}\t")?;
            write!(
                rsf_out,
                "{:.2} ({:.2})\t",
                rsf_res.means[0], rsf_res.stds[0]
            )?;
            write!(
                rsf_out,
                "{:.2} ({:.2})\t",
                rsf_res.means[1], rsf_res.stds[1]
            )?;
            writeln!(rsf_out, "{:.2} ({:.2})", rsf_res.means[2], rsf_res.stds[2])?;
        }
    }

    let out = [rrcf_out, rsf_out].concat();
    Ok(out)
}

#[test]
fn cmp_dt_et_normal() {
    let out = run("in/graph/darpa.csv", Mode::Normal).unwrap();
    save_txt(&format!("{ROOT}/cmp_dt_et"), "normal", &out);
}

#[test]
fn cmp_dt_et_edge_counts() {
    let out = run("in/graph/darpa.csv", Mode::EdgeCounts).unwrap();
    save_txt(&format!("{ROOT}/cmp_dt_et"), "edge_counts", &out);
}

#[test]
fn cmp_dt_et_l1norm() {
    let out = run("in/graph/darpa.csv", Mode::L1Norm).unwrap();
    save_txt(&format!("{ROOT}/cmp_dt_et"), "l1norm", &out);
}
