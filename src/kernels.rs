//! Kernels

use itertools::Itertools;
use nalgebra::{DMatrix, DVector, Scalar};

/// Radial-basis function kernel.
#[derive(Debug)]
pub struct RBF {
    pub length_scale: f64,
}

impl Default for RBF {
    fn default() -> Self {
        RBF { length_scale: 1.0 }
    }
}

impl RBF {
    /// Apply the kernel.
    ///
    /// # Arguments
    ///
    /// * `x` - Left argument for the kernel
    /// * `y` - Right argument for the kernel, if `None`, `RBF(x, x)` is evaluated instead
    pub fn apply(&self, x: &DMatrix<f64>, y: Option<&DMatrix<f64>>) -> DMatrix<f64> {
        let x = x / self.length_scale;
        let mut dist = match y {
            Some(y) => {
                let y = y / self.length_scale;
                cdist(&x, &y)
            }
            None => pdist(&x),
        };
        for i in dist.iter_mut() {
            *i = (-0.5 * i.inlined_clone()).exp();
        }
        dist.fill_diagonal(1.0);
        dist
    }
}

/// Compute the squared Euclidean distance between two vectors.
fn sqeuclidean_distance(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    (a - b).norm_squared()
}

/// Pairwise distances between observations in n-dimensional space using the squared Euclidean distance.
fn pdist(x: &DMatrix<f64>) -> DMatrix<f64> {
    if x.nrows() == 1 {
        return DMatrix::from_element(1, 1, 1.0);
    }

    let distances = (0..x.nrows())
        .combinations(2)
        .map(|idxs| sqeuclidean_distance(&x.row(idxs[0]).transpose(), &x.row(idxs[1]).transpose()))
        .collect::<Vec<_>>();

    squareform(DVector::from_vec(distances))
}

/// Compute distance between each pair of the two collections of inputs using the squared Euclidean distance.
fn cdist(x: &DMatrix<f64>, y: &DMatrix<f64>) -> DMatrix<f64> {
    let rows = x.nrows();
    let cols = y.nrows();
    let mut distances = Vec::with_capacity(rows * cols);
    for xrow in x.row_iter() {
        for yrow in y.row_iter() {
            let d = sqeuclidean_distance(&xrow.transpose(), &yrow.transpose());
            distances.push(d);
        }
    }
    DMatrix::from_vec(rows, cols, distances)
}

/// Convert a vector-form distance vector to a square-form distance matrix.
fn squareform(x: DVector<f64>) -> DMatrix<f64> {
    let n = (x.len() as f64 * 2.0).sqrt().ceil() as usize;
    let mut res = DMatrix::zeros(n, n);
    let mut idx = 0;
    for j in 0..n {
        for i in j + 1..n {
            unsafe {
                *res.get_unchecked_mut((i, j)) = x[idx];
            }
            idx += 1;
        }
    }
    res.fill_upper_triangle_with_lower_triangle();
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn sqeuclidean_distances() {
        let a = DVector::from_vec(vec![1.0, 0.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 0.0, 3.0]);
        assert_eq!(sqeuclidean_distance(&a, &b), 0.0);
        let b = DVector::from_vec(vec![0.0, 2.0, 0.0]);
        assert_eq!(sqeuclidean_distance(&a, &b), 14.0);
    }

    #[test]
    fn pdist_test() {
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3];
        let x = DMatrix::from_vec(3, 4, x);
        let res = pdist(&x);
        let expected = DMatrix::from_vec(
            3,
            3,
            vec![0.0, 0.04, 0.16, 0.04, 0.0, 0.04, 0.16, 0.04, 0.0],
        );
        assert_abs_diff_eq!(res, expected);
    }

    #[test]
    fn cdist_test() {
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3];
        let x = DMatrix::from_vec(3, 4, x);
        let y = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7];
        let y = DMatrix::from_vec(3, 4, y);
        let res = cdist(&x, &y);
        let expected = vec![1.48, 1.24, 1.08, 1.24, 1.08, 1.0, 1.08, 1.0, 1.0];
        let expected = DMatrix::from_vec(3, 3, expected);
        assert_abs_diff_eq!(res, expected);
        let y = vec![
            0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.0, 0.1, 0.0, 0.1,
        ];
        let y = DMatrix::from_vec(4, 4, y);
        let res = cdist(&x, &y);
        let expected = vec![
            1.02, 0.53, 0.39, 0.29, 1.02, 0.39, 0.33, 0.27, 1.1, 0.33, 0.35, 0.33,
        ];
        let expected = DMatrix::from_vec(3, 4, expected);
        assert_abs_diff_eq!(res, expected);
    }

    #[test]
    fn squareform_test() {
        let x = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let res = squareform(x);
        let expected = vec![0.0, 0.1, 0.2, 0.1, 0., 0.3, 0.2, 0.3, 0.0];
        let expected = DMatrix::from_vec(3, 3, expected);
        assert_abs_diff_eq!(res, expected);
        let x = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let res = squareform(x);
        let expected = vec![
            0.0, 0.1, 0.2, 0.3, 0.1, 0.0, 0.4, 0.5, 0.2, 0.4, 0.0, 0.6, 0.3, 0.5, 0.6, 0.0,
        ];
        let expected = DMatrix::from_vec(4, 4, expected);
        assert_abs_diff_eq!(res, expected);
    }

    #[test]
    fn rbf_test() {
        let kernel = RBF::default();
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3];
        let x = DMatrix::from_vec(3, 4, x);
        let res = kernel.apply(&x, None);
        let expected = vec![1.0, 0.98, 0.92, 0.98, 1.0, 0.98, 0.92, 0.98, 1.0];
        let expected = DMatrix::from_vec(3, 3, expected);
        assert_abs_diff_eq!(res, expected, epsilon = 0.01);
        let x = DMatrix::from_vec(1, 2, vec![2.6, 2.3]);
        let res = kernel.apply(&x, None);
        assert_eq!(res.shape(), (1, 1));
    }
}
