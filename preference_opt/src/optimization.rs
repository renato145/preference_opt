//! Main optimization struct

use std::cmp::Ordering;

use crate::{
    acquisition::ExpectedImprovement, kernels::RBF, lbfgs_opt::minimize, posterior::Laplace,
    DataPreferences, DataSamples,
};
use anyhow::Result;
use dialoguer::{theme::ColorfulTheme, Select};
use itertools::Itertools;
use nalgebra::{Cholesky, DMatrix, DVector, Dynamic, MatrixXx2, RowDVector};
use rand::{distributions::Uniform, prelude::Distribution};
use statrs::distribution::Normal;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptError {
    #[error("No samples given, try using `PreferenceOpt::new`")]
    NoSamples,
    #[error("Invalid preference index: {0}")]
    InvalidPreference(usize),
    #[error("The problem have {dims} dimensions but {n_bounds} bounds where given")]
    InvalidBounds { dims: usize, n_bounds: usize },
    #[error("The `low` bound ({low}) is higher than the `high` bound ({high})")]
    InvalidBoundLimits { low: f64, high: f64 },
    #[error("The kernel is not returning a positive-definite matrix. Try gradually increasing the `alpha` parameter on PreferenceOpt")]
    CholeskyNotFound,
}

fn ask_user(
    current: Vec<f64>,
    current_idx: usize,
    proposal: Vec<f64>,
    proposal_idx: usize,
) -> Option<(usize, usize)> {
    let items = vec![
        format!("Current best: {:.2?}", current),
        format!("Proposal    : {:.2?}", proposal),
        "Quit".into(),
    ];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .items(&items)
        .default(0)
        .interact()
        .unwrap();

    match selection {
        0 => Some((current_idx, proposal_idx)),
        1 => Some((proposal_idx, current_idx)),
        _ => None,
    }
}

fn row2vec(x: &DMatrix<f64>, row: usize) -> Vec<f64> {
    x.row(row).iter().map(|&o| o).collect::<Vec<_>>()
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
    /// let opt = PreferenceOpt::new(5);
    /// ```
    pub fn new(dims: usize) -> Self {
        let x = DataSamples::new(dims);
        let m = DataPreferences::new(vec![]);
        Self {
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
        }
    }

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
        if samples.len() == 0 {
            return Err(OptError::NoSamples.into());
        }
        let x = DataSamples::from_data(samples)?;
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

    /// Prepare samples if the are less than
    pub fn init_samples(&mut self) {
        let n = self.x.len();
        if n < 2 {
            let rows2insert = 2 - n;
            let x_append = self.random_sample(rows2insert);
            let mut x = self.x.data.clone().insert_rows(n, rows2insert, 0.0);
            for (i, row) in x_append.row_iter().enumerate() {
                for (j, &o) in row.iter().enumerate() {
                    x[(n + i, j)] = o;
                }
            }
            self.x.data = x;
        }
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

    /// Get sample to test.
    /// This is useful for implementing a manual optimization in conjuction with `add_preference`, be sure to keep track of `f_prior`.
    /// Returns (sample1, index1, sample2, index2, f_prior)
    ///
    /// # Arguments
    ///
    /// * `f_prior` - Prior with mean zero is applied by default
    /// * `n_init` - Number of initialization points for the solver, obtained by randomly sampling the acquisition function
    /// * `n_solve` - The solver will be run n_solve times (cannot be superior to n_init)
    pub fn get_next_sample(
        &mut self,
        f_prior: Option<&Vec<f64>>,
        n_init: usize,
        n_solve: usize,
    ) -> Result<(Vec<f64>, usize, Vec<f64>, usize, Option<Vec<f64>>)> {
        self.init_samples();
        if self.m.data.len() == 0 {
            let row0 = row2vec(&self.x.data, 0);
            let row1 = row2vec(&self.x.data, 1);
            return Ok((row0, 0, row1, 1, None));
        }

        let mut x = self.x.data.clone();
        let m = self.m.data.clone();
        let n = x.nrows();
        let mut f_prior = match f_prior {
            Some(o) => DVector::from_vec(o.clone()),
            None => DVector::zeros(n),
        };
        let m_ind_cpt = if m.nrows() > 0 { m.nrows() - 1 } else { 0 };
        let (m_ind_current, m_ind_proposal) =
            self.get_next_pair(&mut x, &m, &mut f_prior, m_ind_cpt, n_init, n_solve)?;
        let current = row2vec(&x, m_ind_current);
        let proposal = row2vec(&x, m_ind_proposal);
        let f_prior = f_prior.iter().map(|&o| o).collect::<Vec<_>>();
        self.x.data = x;
        Ok((
            current,
            m_ind_current,
            proposal,
            m_ind_proposal,
            Some(f_prior),
        ))
    }

    /// Adds a preference to the data.
    /// This is useful for implementing a manual optimization in conjuction with `get_next_sample`.
    pub fn add_preference(&mut self, preference: usize, other: usize) {
        let mut m = self.m.data.clone();
        let n = m.nrows();
        m = m.insert_row(n, 0);
        m[(n, 0)] = preference;
        m[(n, 1)] = other;
        self.m.data = m;
    }

    /// Get the optimal set of values.
    pub fn get_optimal_values(&self) -> Vec<f64> {
        let idx = self.m.data[(self.m.len() - 1, 0)];
        row2vec(&self.x.data, idx)
    }

    /// Bayesian optimization via preferences inputs.
    /// Returns (optimal_values, f_posterior)
    ///
    /// # Arguments
    ///
    /// * `func` - Function to optimize
    /// * `max_iters` - Maximum number of iterations to be performed for the bayesian optimization
    /// * `f_prior` - Prior with mean zero is applied by default
    /// * `n_init` - Number of initialization points for the solver, obtained by randomly sampling the acquisition function
    /// * `n_solve` - The solver will be run n_solve times (cannot be superior to n_init)
    pub fn interactive_optimization(
        &mut self,
        max_iters: usize,
        f_prior: Option<DVector<f64>>,
        n_init: usize,
        n_solve: usize,
    ) -> Result<(RowDVector<f64>, DVector<f64>)> {
        self.init_samples();
        let mut x = self.x.data.clone();
        let mut m = if self.m.len() == 0 {
            let row0 = row2vec(&x, 0);
            let row1 = row2vec(&x, 1);
            let (a, b) = ask_user(row0, 0, row1, 1).unwrap();
            let m = MatrixXx2::from_vec(vec![a, b]);
            self.m.data = m.clone();
            m
        } else {
            self.m.data.clone()
        };

        let n = x.nrows();
        let mut f_prior = f_prior.map(|o| o.clone()).unwrap_or(DVector::zeros(n));
        let m_last_idx = if m.nrows() > 0 { m.nrows() - 1 } else { 0 };
        for m_ind_cpt in m_last_idx..(m_last_idx + max_iters) {
            let (m_ind_current, m_ind_proposal) =
                self.get_next_pair(&mut x, &m, &mut f_prior, m_ind_cpt, n_init, n_solve)?;
            let current = row2vec(&x, m_ind_current);
            let proposal = row2vec(&x, m_ind_proposal);
            match ask_user(current, m_ind_current, proposal, m_ind_proposal) {
                Some(new_pair) => {
                    let n = m.nrows();
                    m = m.insert_row(n, 0);
                    m[(n, 0)] = new_pair.0;
                    m[(n, 1)] = new_pair.1;
                }
                None => break,
            }
        }

        let idx = m[(m.nrows() - 1, 0)];
        let optimal_values = x.row(idx).clone_owned();
        let f_posterior = f_prior;
        self.x.data = x;
        self.m.data = m;

        Ok((optimal_values, f_posterior))
    }

    /// Optimizes the problem based on a function.
    /// Returns (optimal_values, f_posterior)
    ///
    /// # Arguments
    ///
    /// * `func` - Function to optimize
    /// * `max_iters` - Maximum number of iterations to be performed for the bayesian optimization
    /// * `f_prior` - Prior with mean zero is applied by default
    /// * `n_init` - Number of initialization points for the solver, obtained by randomly sampling the acquisition function
    /// * `n_solve` - The solver will be run n_solve times (cannot be superior to n_init)
    pub fn optimize_fn(
        &mut self,
        func: fn(&[f64]) -> f64,
        max_iters: usize,
        f_prior: Option<DVector<f64>>,
        n_init: usize,
        n_solve: usize,
    ) -> Result<(RowDVector<f64>, DVector<f64>)> {
        self.init_samples();
        let mut x = self.x.data.clone();
        let mut m = if self.m.len() == 0 {
            let row0 = func(&row2vec(&x, 0));
            let row1 = func(&row2vec(&x, 1));
            let m = if row0 < row1 {
                MatrixXx2::from_vec(vec![1, 0])
            } else {
                MatrixXx2::from_vec(vec![0, 1])
            };
            self.m.data = m.clone();
            m
        } else {
            self.m.data.clone()
        };

        let n = x.nrows();
        let mut f_prior = f_prior.map(|o| o.clone()).unwrap_or(DVector::zeros(n));
        let m_last_idx = if m.nrows() > 0 { m.nrows() - 1 } else { 0 };
        for m_ind_cpt in m_last_idx..(m_last_idx + max_iters) {
            let (m_ind_current, m_ind_proposal) =
                self.get_next_pair(&mut x, &m, &mut f_prior, m_ind_cpt, n_init, n_solve)?;
            let current = func(&row2vec(&x, m_ind_current));
            let proposal = func(&row2vec(&x, m_ind_proposal));
            let new_pair = if current < proposal {
                (m_ind_proposal, m_ind_current)
            } else {
                (m_ind_current, m_ind_proposal)
            };
            let n = m.nrows();
            m = m.insert_row(n, 0);
            m[(n, 0)] = new_pair.0;
            m[(n, 1)] = new_pair.1;
        }

        let idx = m[(m.nrows() - 1, 0)];
        let optimal_values = x.row(idx).clone_owned();
        let f_posterior = f_prior;
        self.x.data = x;
        self.m.data = m;

        Ok((optimal_values, f_posterior))
    }

    /// Outputs the index of the current best sample and a proposal sample.
    fn get_next_pair(
        &mut self,
        x: &mut DMatrix<f64>,
        m: &MatrixXx2<usize>,
        f_prior: &mut DVector<f64>,
        m_ind_cpt: usize,
        n_init: usize,
        n_solve: usize,
    ) -> Result<(usize, usize)> {
        self.x.data = x.clone();
        self.fit(x, f_prior)?;
        let x_optim = self.bayesopt(n_init, n_solve);
        let x_optim = DMatrix::from_rows(&[x_optim]);
        let f_optim = self.predict(&x_optim, false).0;
        let _f_prior = self.posterior.clone().unwrap();
        let n = _f_prior.nrows();
        *f_prior = _f_prior.insert_row(n, f_optim[0]);
        let n = x.nrows();
        *x = x.clone().insert_row(n, 0.0);
        x_optim
            .row(0)
            .iter()
            .enumerate()
            .for_each(|(i, &o)| x[(n, i)] = o);
        // current preference index in X
        let m_ind_current = m[(m.nrows() - 1, 0)];
        // suggestion index in X
        let m_ind_proposal = m_ind_cpt + 2;
        Ok((m_ind_current, m_ind_proposal))
    }

    /// Fit a Gaussian process probit regression model.
    fn fit(&mut self, x: &DMatrix<f64>, f_prior: &DVector<f64>) -> Result<()> {
        // compute quantities required for prediction
        let mut k = self.kernel.apply(x, None);
        k.set_diagonal(&k.diagonal().add_scalar(self.alpha));
        let kk = k.clone();
        self.l_ = Some(k.cholesky().ok_or(OptError::CholeskyNotFound)?);

        // compute the posterior distribution of f
        println!("---> x={:?} k={:?} prior={:?} m={:?}", x.shape(), kk.shape(), f_prior.shape(), self.m.data.shape());
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
    fn init_samples() -> Result<()> {
        let mut opt = PreferenceOpt::new(2).with_same_bounds((0.0, 10.0))?;
        opt.init_samples();
        println!("{}", opt.x.data);
        Ok(())
    }

    #[test]
    fn optimize_fn_test() -> Result<()> {
        let samples = vec![vec![1.0, 3.0], vec![2.0, 5.0]];
        let preferences = vec![(1, 0)];
        let mut opt =
            PreferenceOpt::from_data(samples, preferences)?.with_same_bounds((0.0, 10.0))?;
        let func = |o: &[f64]| o.iter().sum();
        let (optimal_values, f_posterior) = opt.optimize_fn(func, 1, None, 1, 1)?;
        println!("optimal_values -> {}", optimal_values);
        println!("f_posterior -> {}", f_posterior);
        opt.x.show();
        opt.m.show();
        let (optimal_values, f_posterior) = opt.optimize_fn(func, 1, Some(f_posterior), 1, 1)?;
        println!("optimal_values -> {}", optimal_values);
        println!("f_posterior -> {}", f_posterior);
        opt.x.show();
        opt.m.show();
        let samples = vec![vec![1.0, 0.0, 7.0], vec![2.0, 5.0, 3.0]];
        let preferences = vec![(1, 0)];
        let mut opt = PreferenceOpt::from_data(samples, preferences)?.with_bounds(vec![
            (0.0, 10.0),
            (0.0, 5.0),
            (0.0, 30.0),
        ])?;
        let func = |o: &[f64]| o.iter().sum();
        let (optimal_values, f_posterior) = opt.optimize_fn(func, 10, None, 10, 3)?;
        println!("optimal_values -> {}", optimal_values);
        println!("f_posterior -> {}", f_posterior);
        opt.x.show();
        opt.m.show();
        Ok(())
    }

    #[test]
    fn optimize_fn_empty_preferences() -> Result<()> {
        let samples = vec![vec![1.0, 3.0], vec![2.0, 5.0]];
        let preferences = vec![];
        let mut opt =
            PreferenceOpt::from_data(samples, preferences)?.with_same_bounds((0.0, 10.0))?;
        let func = |o: &[f64]| o.iter().sum();
        let (optimal_values, f_posterior) = opt.optimize_fn(func, 1, None, 1, 1)?;
        println!("optimal_values -> {}", optimal_values);
        println!("f_posterior -> {}", f_posterior);
        opt.x.show();
        opt.m.show();
        Ok(())
    }

    #[test]
    fn optimize_fn_empty_samples() -> Result<()> {
        let samples = vec![vec![1.0, 3.0]];
        let preferences = vec![];
        let mut opt =
            PreferenceOpt::from_data(samples, preferences)?.with_same_bounds((0.0, 10.0))?;
        let func = |o: &[f64]| o.iter().sum();
        let (optimal_values, f_posterior) = opt.optimize_fn(func, 1, None, 1, 1)?;
        println!("optimal_values -> {}", optimal_values);
        println!("f_posterior -> {}", f_posterior);
        opt.x.show();
        opt.m.show();
        let mut opt = PreferenceOpt::new(2).with_same_bounds((0.0, 10.0))?;
        let func = |o: &[f64]| o.iter().sum();
        let (optimal_values, f_posterior) = opt.optimize_fn(func, 1, None, 1, 1)?;
        println!("optimal_values -> {}", optimal_values);
        println!("f_posterior -> {}", f_posterior);
        opt.x.show();
        opt.m.show();
        Ok(())
    }

    #[test]
    fn manual_optimization() -> Result<()> {
        let mut opt = PreferenceOpt::new(3).with_same_bounds((0.0, 10.0))?;
        let (_, idx1, _, idx2, f_prior) = opt.get_next_sample(None, 500, 1)?;
        opt.add_preference(idx1, idx2);
        let (_, idx1, _, idx2, f_prior) = opt.get_next_sample(f_prior.as_ref(), 500, 1)?;
        opt.add_preference(idx1, idx2);
        let (_, idx1, _, idx2, _) = opt.get_next_sample(f_prior.as_ref(), 500, 1)?;
        opt.add_preference(idx1, idx2);
        opt.x.show();
        opt.m.show();
        println!("{:?}", opt.get_optimal_values());
        Ok(())
    }
}
