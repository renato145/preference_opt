//! Main optimization struct

use std::cmp::Ordering;

use crate::{
    acquisition::ExpectedImprovement, kernels::RBF, lbfgs_opt::minimize, posterior::Laplace,
    DataPreferences, DataSamples,
};
use anyhow::Result;
use itertools::Itertools;
use nalgebra::{Cholesky, DMatrix, DVector, Dynamic, RowDVector};
use rand::{distributions::Uniform, prelude::Distribution};
use statrs::distribution::Normal;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptError {
    #[error("Invalid preference index: {0}")]
    InvalidPreference(usize),
    #[error("The problem have {dims} dimensions but {n_bounds} bounds where given")]
    InvalidBounds { dims: usize, n_bounds: usize },
    #[error("The `low` bound ({low}) is higher than the `high` bound ({high})")]
    InvalidBoundLimits { low: f64, high: f64 },
    #[error("The kernel is not returning a positive-definite matrix. Try gradually increasing the `alpha` parameter on PreferenceOpt")]
    CholeskyNotFound,
}

#[derive(Debug)]
pub struct PreferenceOpt {
    pub x: DataSamples,
    pub m: DataPreferences,
    pub bounds: Option<Vec<(f64, f64)>>,
    pub kernel: RBF,
    /// Value added to the diagonal of the kernel matrix during fitting.
    /// Larger values correspond to increased noise level in the observations.
    pub alpha: f64,
    pub posterior: Option<DVector<f64>>,
    pub post_approx: Laplace,
    pub acquisition: ExpectedImprovement,
    pub distribution: Normal,
    l_: Option<Cholesky<f64, Dynamic>>,
    distributions: Option<Vec<Uniform<f64>>>,
}

impl PreferenceOpt {
    /// Creates a new optimization from data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use preference_opt::PreferenceOpt;
    /// let x = vec![vec![0.0, 1.0], vec![4.0, 3.0], vec![2.0, 3.0]];
    /// let m = vec![(0, 1), (2, 0)];
    /// let opt = PreferenceOpt::from_data(x, m).unwrap();
    /// ```
    pub fn from_data(samples: Vec<Vec<f64>>, preferences: Vec<(usize, usize)>) -> Result<Self> {
        let x = DataSamples::new(samples)?;
        let m = DataPreferences::new(preferences);
        if m.max() >= x.len() {
            return Err(OptError::InvalidPreference(m.max()).into());
        }
        Ok(Self {
            x,
            m,
            bounds: None,
            kernel: RBF::default(),
            alpha: 1e-5,
            posterior: None,
            post_approx: Laplace::default(),
            acquisition: ExpectedImprovement::default(),
            distribution: Normal::new(0.0, 1.0).unwrap(),
            l_: None,
            distributions: None,
        })
    }

    /// Get the number of dimensions to optimize.
    pub fn dims(&self) -> usize {
        self.x.data.ncols()
    }

    /// Define bounds for the optimization problem (inclusive bounds).
    ///
    /// # Examples
    ///
    /// ```
    /// # use preference_opt::PreferenceOpt;
    /// let x = vec![vec![0.0, 1.0], vec![4.0, 3.0], vec![2.0, 3.0]];
    /// let m = vec![(0, 1), (2, 0)];
    /// let opt = PreferenceOpt::from_data(x, m)?.with_bounds(vec![(0.0, 10.0), (0.0, 10.0)])?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Result<Self> {
        if bounds.len() != self.dims() {
            return Err(OptError::InvalidBounds {
                dims: self.dims(),
                n_bounds: bounds.len(),
            }
            .into());
        }
        for &(low, high) in bounds.iter() {
            if low >= high {
                return Err(OptError::InvalidBoundLimits { low, high }.into());
            }
        }

        self.distributions = Some(
            bounds
                .iter()
                .map(|(low, high)| Uniform::new_inclusive(low, high))
                .collect::<Vec<_>>(),
        );
        self.bounds = Some(bounds);
        Ok(self)
    }

    /// Define the same bounds for all dimensions of the optimization problem (inclusive bounds).
    ///
    /// # Examples
    ///
    /// ```
    /// # use preference_opt::PreferenceOpt;
    /// let x = vec![vec![0.0, 1.0], vec![4.0, 3.0], vec![2.0, 3.0]];
    /// let m = vec![(0, 1), (2, 0)];
    /// let opt = PreferenceOpt::from_data(x, m)?.with_same_bounds((0.0, 10.0))?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_same_bounds(self, bounds: (f64, f64)) -> Result<Self> {
        let bounds = vec![bounds; self.dims()];
        self.with_bounds(bounds)
    }

    /// Index of the last preference (m).
    fn m_last_idx(&self) -> usize {
        self.m.data.nrows() - 1
    }

    /// Optimizes the problem based on a function.
    ///
    /// # Arguments
    ///
    /// * `func` - Function to optimize
    /// * `max_iters` - Maximum number of iterations to be performed for the bayesian optimization
    /// * `f_prior` - Prior with mean zero is applied by default
    pub fn optimize_fn(
        &mut self,
        _func: fn(&[f64]) -> f64,
        max_iters: usize,
        f_prior: Option<Vec<f64>>,
        n_init: usize,
        n_solve: usize,
    ) -> Result<()> {
        let m_last_idx = self.m_last_idx();
        for m_ind_cpt in m_last_idx..(m_last_idx + max_iters) {
            self.fit(f_prior.as_ref())?;
            let x_optim = self.bayesopt(n_init, n_solve);
            // x_optim = self.bayesopt(bounds, method, n_init, n_solve)
        }
        // let row = self.x.data.row(0).iter().cloned().collect::<Vec<_>>();
        // let res = func(&row);
        // println!("{}", res);
        Ok(())
    }

    /// Fit a Gaussian process probit regression model.
    fn fit(&mut self, f_prior: Option<&Vec<f64>>) -> Result<()> {
        // compute quantities required for prediction
        let mut k = self.kernel.apply(&self.x.data, None);
        k.set_diagonal(&k.diagonal().add_scalar(self.alpha));
        let n = k.nrows();
        self.l_ = Some(k.cholesky().ok_or(OptError::CholeskyNotFound)?);
        let f_prior = f_prior.map(|o| o.clone()).unwrap_or(vec![0f64; n]);
        let f_prior = DVector::from_vec(f_prior);

        // compute the posterior distribution of f
        self.posterior = Some(self.post_approx.apply(
            &f_prior,
            &self.m.data,
            self.l_.as_ref().unwrap(),
            &self.distribution,
        ));
        Ok(())
    }

    /// Bayesian optimization based on the optimization of a
    /// utility function of the attributes of the posterior distribution.
    fn bayesopt(&self, n_init: usize, n_solve: usize) -> RowDVector<f64> {
        let y_max = self.posterior.as_ref().unwrap().max();
        let x_tries = self.random_sample(n_init);
        let aqc_optim = |x| {
            let (y_mean, std) = self.predict(&x, true);
            let std = std.unwrap();
            self.acquisition
                .apply(y_mean, std, y_max, &self.distribution)
        };

        let ys = aqc_optim(x_tries.clone_owned());
        let (x_argmax, mut max_acq) = ys.argmax();
        let mut x_max = x_tries.row(x_argmax).clone_owned();
        let x_seeds = ys
            .into_iter()
            .enumerate()
            .sorted_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Less))
            .map(|o| x_tries.row(o.0).clone_owned())
            .take(n_solve);
        for x_try in x_seeds {
            // Find the minimum of -1 * acquisition function
            let (best, best_value) =
                minimize(|o| -aqc_optim(o)[0], x_try, self.bounds.as_ref().unwrap());
            // Store it if better than previous minimum(maximum)
            if (-best_value) >= max_acq {
                x_max = best.row(0).clone_owned();
                max_acq = -best_value;
            }
        }

        x_max
    }

    /// Samples to warm up with random points.
    fn random_sample(&self, n: usize) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let x = self
            .distributions
            .as_ref()
            .unwrap()
            .iter()
            .map(|o| o.sample_iter(&mut rng).take(n).collect::<Vec<_>>())
            .flatten()
            .collect::<Vec<_>>();
        DMatrix::from_vec(n, self.dims(), x)
    }

    /// Predict using the Gaussian process regression model.
    /// Returns mean and standard deviation of predictive distribution at query points `x`.
    fn predict(&self, x: &DMatrix<f64>, return_std: bool) -> (DVector<f64>, Option<DVector<f64>>) {
        let l_ = self.l_.as_ref().unwrap();
        let k_trans = self.kernel.apply(&self.x.data, Some(x));
        let lk = l_.solve(&k_trans);
        let lf = l_.solve(self.posterior.as_ref().unwrap());
        let y_mean = lk.transpose() * lf;
        let std = if return_std {
            let y_var = self.kernel.apply(x, None).diagonal() - lk.map(|o| o.powi(2)).row_sum_tr();
            let std = y_var.map(|o| if o < 0.0 { 0.0 } else { o.sqrt() });
            Some(std)
        } else {
            None
        };
        (y_mean, std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic(expected = "Invalid preference index: 3")]
    fn from_data_fails() {
        let x = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        let m = vec![(0, 1), (2, 3)];
        PreferenceOpt::from_data(x, m).unwrap();
    }

    #[test]
    #[should_panic(expected = "The `low` bound (2) is higher than the `high` bound (1)")]
    fn with_bounds_fail() {
        let x = vec![vec![0.0, 1.0], vec![4.0, 3.0], vec![2.0, 3.0]];
        let m = vec![(0, 1)];
        PreferenceOpt::from_data(x, m)
            .unwrap()
            .with_bounds(vec![(2.0, 1.0), (0.0, 1.0)])
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "The problem have 2 dimensions but 1 bounds where given")]
    fn with_bounds_fail_dims() {
        let x = vec![vec![0.0, 1.0], vec![4.0, 3.0], vec![2.0, 3.0]];
        let m = vec![(0, 1)];
        PreferenceOpt::from_data(x, m)
            .unwrap()
            .with_bounds(vec![(0.0, 10.0)])
            .unwrap();
    }

    #[test]
    fn random_sample() -> Result<()> {
        let samples = vec![vec![1.0, 0.0]];
        let preferences = vec![];
        let opt = PreferenceOpt::from_data(samples, preferences)?
            .with_bounds(vec![(0.0, 10.0), (20.0, 40.0)])?;
        let res = opt.random_sample(10);
        assert_eq!(res.nrows(), 10);
        assert!(res.column(0).max() < 20.0);
        assert!(res.column(1).min() > 10.0);
        Ok(())
    }

    #[test]
    fn optimize_fn_test() -> Result<()> {
        let samples = vec![vec![1.0, 0.0], vec![2.0, 5.0]];
        let preferences = vec![(1, 0)];
        let mut opt =
            PreferenceOpt::from_data(samples, preferences)?.with_same_bounds((0.0, 10.0))?;
        let func = |o: &[f64]| o.iter().sum();
        opt.optimize_fn(func, 1, None, 10, 3)?;
        // println!("{}", opt.posterior.unwrap());
        Ok(())
    }
}
