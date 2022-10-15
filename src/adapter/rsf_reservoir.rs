use ndarray::{ArrayBase, Data, Ix1, RawDataClone};

use crate::algorithm::{config::Config, forest::RSF, tree::RandShiftTree};

use super::reservoir::{Reservoir, ReservoirUpdate};

pub struct RSFReservoir<I: Iterator, const M: bool> {
    iter: I,
    reservoirs: Vec<Reservoir<I::Item>>,
    f: RSF,
}

impl<S, I, const M: bool> RSFReservoir<I, M>
where
    S: Data<Elem = f32> + RawDataClone,
    I: Iterator<Item = ArrayBase<S, Ix1>>,
{
    fn new(iter: I, cfg: &Config) -> Self {
        let n_reservoirs = if M { cfg.n_trees } else { 1 };
        let r = cfg.n_points;
        let reservoirs = (0..n_reservoirs).map(|_| Reservoir::new(r)).collect();
        let f = RSF::from_config(cfg);
        let mut res = Self {
            iter,
            reservoirs,
            f,
        };
        res.by_ref().take(r).for_each(drop);
        res
    }

    fn handle_new(&mut self, item: ArrayBase<S, Ix1>) {
        if M {
            for (tree, res) in self.f.iter_trees_mut().zip(self.reservoirs.iter_mut()) {
                match res.insert(item.clone()) {
                    ReservoirUpdate::Skip(_p) => {}
                    ReservoirUpdate::Insert(p) => {
                        tree.insert(&p);
                    }
                    ReservoirUpdate::Replace(q, p) => {
                        tree.remove(&q);
                        tree.insert(&p);
                    }
                }
            }
        } else {
            match self.reservoirs[0].insert(item) {
                ReservoirUpdate::Skip(_p) => {}
                ReservoirUpdate::Insert(p) => {
                    self.f.insert(&p);
                }
                ReservoirUpdate::Replace(q, p) => {
                    self.f.remove(&q);
                    self.f.insert(&p);
                }
            }
        }
    }
}

impl<'a, S, I, const M: bool> Iterator for RSFReservoir<I, M>
where
    S: Data<Elem = f32> + RawDataClone + 'a,
    I: Iterator<Item = ArrayBase<S, Ix1>>,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| {
            let s = self.f.score(&item);
            self.handle_new(item);
            s
        })
    }
}

pub trait RSFReservoirIter<S: Data<Elem = f32> + RawDataClone>:
    Iterator<Item = ArrayBase<S, Ix1>> + Sized
{
    fn rsf_reservoir<const M: bool>(self, cfg: &Config) -> RSFReservoir<Self, M> {
        RSFReservoir::new(self, cfg)
    }
}

impl<S: Data<Elem = f32> + RawDataClone, I: Iterator<Item = ArrayBase<S, Ix1>>> RSFReservoirIter<S>
    for I
{
}
