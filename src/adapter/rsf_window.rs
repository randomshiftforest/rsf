use std::iter::Enumerate;

use ndarray::{ArrayBase, Data, Ix1, RawDataClone};

use crate::{
    adapter::window::{Window, WindowIterator, WindowUpdate},
    algorithm::{config::Config, forest::RSF, tree::RandShiftTree},
};

use super::hash_picker::HashPicker;

pub struct RSFWindow<I: Iterator, const M: bool> {
    iter: Window<Enumerate<I>>,
    f: RSF,
    pickers: Vec<HashPicker>,
}

impl<'a, S, I, const M: bool> RSFWindow<I, M>
where
    S: Data<Elem = f32> + 'a,
    I: Iterator<Item = ArrayBase<S, Ix1>>,
{
    fn new(items: I, cfg: &Config) -> Self {
        let n_pickers = if M { cfg.n_trees } else { 1 };
        let pickers = (0..n_pickers)
            .map(|_| HashPicker::from_frac(cfg.n_points, cfg.window))
            .collect();
        let iter = items.enumerate().window(cfg.window);
        let f = RSF::from_config(cfg);
        Self { iter, f, pickers }
    }

    fn handle_old(&mut self, item: (usize, I::Item)) {
        if M {
            for (tree, picker) in self.f.iter_trees_mut().zip(self.pickers.iter()) {
                if picker.picks(&item.0) {
                    tree.remove(&item.1);
                }
            }
        } else if self.pickers[0].picks(&item.0) {
            self.f.remove(&item.1);
        }
    }

    fn handle_new(&mut self, item: (usize, I::Item)) {
        if M {
            for (tree, picker) in self.f.iter_trees_mut().zip(self.pickers.iter()) {
                if picker.picks(&item.0) {
                    tree.insert(&item.1);
                }
            }
        } else if self.pickers[0].picks(&item.0) {
            self.f.insert(&item.1);
        }
    }
}

impl<'a, S, I, const M: bool> Iterator for RSFWindow<I, M>
where
    S: Data<Elem = f32> + RawDataClone + 'a,
    I: Iterator<Item = ArrayBase<S, Ix1>>,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                None => return None,
                Some(window_update) => match window_update {
                    WindowUpdate::Insert(new_item) => self.handle_new(new_item),
                    WindowUpdate::Replace(old_item, new_item) => {
                        let s = self.f.score(&new_item.1);
                        self.handle_old(old_item);
                        self.handle_new(new_item);
                        return Some(s);
                    }
                },
            }
        }
    }
}

pub trait RSFWindowIter<S: Data<Elem = f32>>: Iterator<Item = ArrayBase<S, Ix1>> + Sized {
    fn rsf_window<const M: bool>(self, cfg: &Config) -> RSFWindow<Self, M> {
        RSFWindow::new(self, cfg)
    }
}

impl<S: Data<Elem = f32>, I: Iterator<Item = ArrayBase<S, Ix1>>> RSFWindowIter<S> for I {}
