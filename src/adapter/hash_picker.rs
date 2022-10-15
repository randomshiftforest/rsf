use std::{
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hash, Hasher},
};

pub struct HashPicker {
    s: RandomState,
    num: u64,
    den: u64,
}

impl HashPicker {
    pub fn from_frac(num: usize, den: usize) -> Self {
        assert!(num <= den, "invalid picking fraction");
        Self {
            s: RandomState::new(),
            num: num as u64,
            den: den as u64,
        }
    }

    pub fn from_prob(p: f64) -> Self {
        let num = 1;
        let den = p.recip().floor() as usize;
        Self::from_frac(num, den)
    }

    pub fn picks<I: Hash>(&self, i: &I) -> bool {
        let mut hasher = self.s.build_hasher();
        i.hash(&mut hasher);
        let h = hasher.finish();
        (h % self.den) < self.num
    }
}
