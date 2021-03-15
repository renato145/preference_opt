//! Posterior approximation strategies

use itertools::Itertools;
use nalgebra::{Cholesky, DMatrix, DVector, Dynamic, Matrix, MatrixXx2, SliceStorage, U1, U2};
use statrs::distribution::{Continuous, Normal, Univariate};

/// Laplace approximation of the posterior distribution P(f|M) in a probit-preference gaussian regression model.
#[derive(Debug)]
pub struct Laplace {
    /// Noise of the latent preference relations functions
    pub s_eval: f64,
    /// Maximum number of iterations of the Newton-Raphson recursion for the Laplace approximation of the posterior distribution P(f|M)
    pub max_iter: usize,
    /// Gradient descent step size
    pub eta: f64,
    /// Gradient descent convergence tolerance
    pub tol: f64,
}

impl Default for Laplace {
    fn default() -> Self {
        Laplace {
            s_eval: 1e-5,
            max_iter: 1000,
            eta: 0.01,
            tol: 1e-5,
        }
    }
}

pub type PreferenceSlice<'a, N, RStride = U1, CStride = Dynamic> =
    Matrix<N, U1, U2, SliceStorage<'a, N, U1, U2, RStride, CStride>>;

impl Laplace {
    /// Return the Laplace approximation of P(f|M).
    /// A Newton-Raphson descent is used to approximate f at the MAP.
    ///
    /// # Arguments
    ///
    /// * `f` - Gaussian process prior
    /// * `m` - Target choices (preferences)
    /// * `k` - Kernel decomposition
    pub fn apply(
        &self,
        f_prior: &DVector<f64>,
        m: &MatrixXx2<usize>,
        k: &Cholesky<f64, Dynamic>,
        distribution: &Normal,
    ) -> DVector<f64> {
        // Likelihood function of a preference relation
        let z_item = |f: &DVector<f64>, r, c| (f[r] - f[c]) / (2f64).sqrt() * self.s_eval;

        let z = |f: &DVector<f64>, column, idx| {
            let tst = m
                .column(column)
                .iter()
                .enumerate()
                .filter(|(_, &o)| o == idx)
                .map(|(i, _)| {
                    let row = m.row(i);
                    let (r, c) = (row[0], row[1]);
                    z_item(f, r, c)
                })
                .collect::<Vec<_>>();
            tst
        };

        let apply_distribution = |z: Vec<f64>| {
            z.into_iter()
                .map(|o| distribution.pdf(o) / distribution.cdf(o))
                .collect::<Vec<_>>()
        };

        // Root of the Taylor expansion derivative of log P(f|M)
        // with respect to the latent preference valuation functions
        let delta = |f: &DVector<f64>, m: &MatrixXx2<usize>| {
            let n = f.nrows();

            // Quantities of the first order derivative  of the loss function p(f|M) with respect to f
            let b = (0..n)
                .map(|i| {
                    let z_r = z(f, 0, i);
                    let z_c = z(f, 1, i);
                    let pos_r: f64 = apply_distribution(z_r).iter().sum();
                    let neg_c: f64 = apply_distribution(z_c).iter().sum();
                    (pos_r - neg_c) / (2f64).sqrt()
                })
                .collect::<Vec<f64>>();
            let b = DVector::from_vec(b);

            // Quantities of the second order derivative of the loss function p(f|M) with respect to f
            let mut c = DMatrix::zeros(n, n);
            let m_uni = m
                .row_iter()
                .map(|o| (o[0], o[1]))
                .unique()
                .collect::<Vec<_>>();
            for i in 0..m_uni.len() {
                let (m, n) = m_uni[i];
                let z_mn = z_item(f, m, n);
                let z_nm = -z_mn;
                let pdf_z = distribution.pdf(z_mn);
                let cdf_z_mn = distribution.cdf(z_mn);
                let cdf_z_nm = distribution.cdf(z_nm);
                let c_mn = (pdf_z / cdf_z_mn).powi(2) + pdf_z / cdf_z_mn * z_mn;
                let c_nm = (pdf_z / cdf_z_nm).powi(2) + pdf_z / cdf_z_nm * z_nm;
                c[(m, n)] = -(c_mn + c_nm) / 2f64 * self.s_eval;
                c[(n, m)] = -(c_mn + c_nm) / 2f64 * self.s_eval;
            }

            // Gradient
            let kf = k.solve(f);
            let g = kf - b;
            // Hessian
            let h = k.inverse() + c;
            (g, h)
        };

        let mut f = f_prior.clone();
        let mut eps = self.tol + 1.0;
        let mut i = 0;
        while (i < self.max_iter) & (eps > self.tol) {
            let (g, h) = delta(&f, m);
            let f_new = &f - self.eta * h.cholesky().unwrap().solve(&g);
            eps = (&f_new - f).norm();
            f = f_new;
            i += 1;
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::RBF, DataPreferences, DataSamples};
    use anyhow::Result;
    use approx::assert_abs_diff_eq;

    #[test]
    fn laplace_test() -> Result<()> {
        let x = DataSamples::new(vec![vec![1.0, 0.0], vec![2.0, 5.0], vec![3.5, 2.0]])?;
        let m = DataPreferences::new(vec![(1, 0), (2, 1)]);
        let kernel = RBF::default();
        let mut k = kernel.apply(&x.data, None);
        k.set_diagonal(&k.diagonal().add_scalar(1e-5));
        let f_prior = DVector::zeros(k.nrows());
        let k = k.cholesky().unwrap();
        let post_approx = Laplace::default();
        let distribution = Normal::new(0.0, 1.0)?;
        let res = post_approx.apply(&f_prior, &m.data, &k, &distribution);
        let expected = DVector::from_vec(vec![-0.56, 0.002, 0.56]);
        assert_abs_diff_eq!(res, expected, epsilon = 0.01);
        Ok(())
    }
}
