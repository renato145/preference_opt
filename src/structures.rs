//! Main data structures

use nalgebra::{DMatrix, MatrixXx2};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("Not all rows have size=={0}")]
    InvalidCols(usize),
}

#[derive(Debug)]
pub struct DataSamples {
    pub data: DMatrix<f64>,
}

impl DataSamples {
    /// Creates a new example data structure.
    ///
    /// # Examples
    ///
    /// ```
    /// # use preference_opt::DataSamples;
    /// let x = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
    /// let x = DataSamples::new(x).unwrap();
    /// ```
    pub fn new(x: Vec<Vec<f64>>) -> Result<Self, DataError> {
        let rows = x.len();
        let cols = x[0].len();
        if !x.iter().map(|row| row.len()).all(|o| o == cols) {
            return Err(DataError::InvalidCols(cols));
        }

        let x = x.into_iter().flatten().collect();
        let data = DMatrix::from_vec(rows, cols, x);
        Ok(Self { data })
    }

    /// Get the number of samples.
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Prints the samples.
    pub fn show(&self) {
        println!("Samples:");
        for (i, row) in self.data.row_iter().enumerate() {
            let row = row
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!("[{}]: {}", i, row);
        }
    }
}

#[derive(Debug)]
pub struct DataPreferences {
    pub data: MatrixXx2<usize>,
}

impl DataPreferences {
    /// Creates a new preference data structure.
    ///
    /// # Examples
    ///
    /// ```
    /// # use preference_opt::DataPreferences;
    /// let m = vec![(0, 1), (2, 3)];
    /// let m = DataPreferences::new(m);
    /// ```
    pub fn new(x: Vec<(usize, usize)>) -> Self {
        let x = x.into_iter().map(|(a, b)| vec![a, b]).flatten().collect();
        let data = MatrixXx2::from_vec(x);
        Self { data }
    }

    /// Get the highest index in the data.
    pub fn max(&self) -> usize {
        self.data.max()
    }

    /// Prints the preferences
    pub fn show(&self) {
        println!("Preferences:");
        for (i, row) in self.data.row_iter().enumerate() {
            println!("[{}]: {}, {}", i, row[0], row[1]);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic]
    fn samples_creation_fail() {
        let x = vec![vec![0.0, 1.0], vec![2.0, 3.0, 4.0]];
        DataSamples::new(x).unwrap();
    }

    #[test]
    fn samples_show() {
        let x = vec![vec![0.0, 1.2], vec![2.0, 3.0]];
        let x = DataSamples::new(x).unwrap();
        x.show();
    }

    #[test]
    fn preferences_show() {
        let x = vec![(1, 2), (2, 3)];
        let x = DataPreferences::new(x);
        x.show();
    }
}
