use std::collections::HashMap;

use super::par_stream_sampler::ParStreamSampler;
use crate::{
    algorithm::{config::Config, forest::RSF, tree::RandShiftTree},
    metric::k_smallest,
};
use itertools::Itertools;
use ndarray::{prelude::*, Data};
use rand::{seq::SliceRandom, thread_rng};
use rand_distr::{Distribution, Uniform};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn retain(ps: &mut Vec<(usize, Array1<f32>)>, scores: &Array1<f32>, n1: usize) {
    let n = ps.len();
    let most_anomalous = k_smallest(scores, n1);
    let mut mask = Array1::from_elem(n, false);
    most_anomalous.into_iter().for_each(|i| mask[i] = true);
    let mut keep = mask.into_iter();
    ps.retain(|_| keep.next().unwrap());
}

pub trait DistributedIter<S: Data<Elem = f32>>: Iterator<Item = ArrayBase<S, Ix1>> + Sized {
    fn distribute_balanced(self, cfg: &Config) -> HashMap<usize, Vec<(usize, Array1<f32>)>> {
        let mut rng = thread_rng();
        let picks = Uniform::new(0, cfg.n_machines).sample_iter(&mut rng);
        let points = self.map(|p| p.to_owned()).enumerate();
        picks.zip(points).into_group_map()
    }

    fn distribute(self, cfg: &Config) -> Vec<Vec<(usize, Array1<f32>)>> {
        let points: Vec<_> = self.map(|p| p.to_owned()).enumerate().collect();
        let n = points.len();

        let mut splits: Vec<usize> = Uniform::new(0, n)
            .sample_iter(thread_rng())
            .take(cfg.n_machines - 1)
            .chain([0, n])
            .collect();
        splits.sort_unstable();

        splits
            .windows(2)
            .scan(points, |points, win| {
                let (lb, ub) = (win[0], win[1]);
                let at = points.len() - (ub - lb);
                Some(points.split_off(at))
            })
            .collect()
    }

    fn one_way_coordinator(self, cfg: &Config, n1: usize) -> (RSF, Vec<usize>) {
        let sample_size = cfg.n_points / cfg.n_machines;
        let distr = self.distribute(cfg);

        // partial forests
        let (sketches, candidates): (Vec<_>, Vec<_>) = distr
            .into_par_iter()
            .map(|mut points| {
                // construct forest
                let mut f = RSF::from_config(cfg);
                for tree in f.iter_trees_mut() {
                    for (_i, p) in points.choose_multiple(&mut thread_rng(), sample_size) {
                        tree.insert(p);
                    }
                }
                // score points
                let scores: Array1<_> = points.iter().map(|(_i, p)| f.score(p)).collect();
                retain(&mut points, &scores, n1);
                f.sketch(cfg.sketch_size);
                (f, points)
            })
            .unzip();
        let mut candidates: Vec<_> = candidates.into_iter().flatten().collect();

        // full forest
        let sketch_sum =
            sketches
                .into_iter()
                .fold(RSF::from_config(cfg), |mut sketch_sum, sketch| {
                    sketch_sum.extend(sketch);
                    sketch_sum
                });
        let scores: Array1<_> = candidates
            .iter()
            .map(|(_i, p)| sketch_sum.score(p))
            .collect();
        retain(&mut candidates, &scores, n1);
        let anomalies: Vec<_> = candidates.into_iter().map(|(i, _p)| i).collect();
        (sketch_sum, anomalies)
    }

    fn two_way_par_streams(self, cfg: &Config, n1: usize) -> (RSF, Vec<usize>) {
        let distr = self.distribute(cfg);

        // pass 1
        let mut sampler = ParStreamSampler::new(cfg);
        for (m, points) in distr.iter().enumerate() {
            for (_p, point) in points {
                sampler.insert(m, point);
            }
        }
        let sample = sampler.query(cfg);
        let mut f = RSF::from_config(cfg);
        f.batch_insert(&sample);
        f.sketch(cfg.sketch_size);

        // pass 2
        let mut candidates: Vec<_> = distr
            .into_par_iter()
            .flat_map(|points| {
                let mut scores = points
                    .into_iter()
                    .map(|(i, p)| (i, f.score(&p)))
                    .collect::<Vec<_>>();
                scores.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                scores.truncate(n1);
                scores
            })
            .collect();
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let anomalies = candidates.into_iter().take(n1).map(|c| c.0).collect();

        (f, anomalies)
    }
}

impl<S: Data<Elem = f32>, I: Iterator<Item = ArrayBase<S, Ix1>>> DistributedIter<S> for I {}
