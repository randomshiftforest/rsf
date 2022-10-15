use ndarray::prelude::*;

pub fn const_copy<const D: usize>(x: ArrayView2<f32>) -> Vec<[f32; D]> {
    x.outer_iter()
        .map(|p| {
            let mut res = [0.; D];
            for i in 0..D {
                res[i] = p[i];
            }
            res
        })
        .collect()
}

pub fn const_copied<'a, const D: usize>(
    x: impl Iterator<Item = ArrayView1<'a, f32>> + 'a,
) -> impl Iterator<Item = [f32; D]> + 'a {
    x.map(|p| {
        let mut res = [0.; D];
        for i in 0..D {
            res[i] = p[i];
        }
        res
    })
}
