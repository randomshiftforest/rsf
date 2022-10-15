use ndarray::ArrayViewMut1;

use crate::algorithm::bounding_box::BoundingBox;

pub trait NormaliseIter<'a>: Iterator<Item = ArrayViewMut1<'a, f32>> + Sized {
    fn normalise(self, bb: &BoundingBox) {
        self.for_each(|p| bb.normalise(p))
    }
}

impl<'a, I: Iterator<Item = ArrayViewMut1<'a, f32>>> NormaliseIter<'a> for I {}
