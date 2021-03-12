//! Main optimization struct

use crate::{DataPreferences, DataSamples};
use anyhow::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptError {
    #[error("Invalid preference index: {0}")]
    InvalidPreference(usize),
}

#[derive(Debug)]
pub struct PreferenceOpt {
    pub x: DataSamples,
    pub m: DataPreferences,
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
        Ok(Self { x, m })
    }
    pub fn optimize_fn(&self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic]
    fn from_data_fails() {
        let x = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        let m = vec![(0, 1), (2, 3)];
        PreferenceOpt::from_data(x, m).unwrap();
    }
}
