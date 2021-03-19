//! Acquisition functions

use nalgebra::DVector;
use statrs::distribution::{Continuous, Normal, Univariate};

/// Expected improvement utility function.
#[derive(Debug)]
pub struct ExpectedImprovement {
    /// Exploitation-exploration trade-off parameter
    pub xi: f64,
}

impl Default for ExpectedImprovement {
    fn default() -> Self {
        ExpectedImprovement { xi: 0f64 }
    }
}

impl ExpectedImprovement {
    pub fn apply(
        &self,
        y_mean: DVector<f64>,
        std: DVector<f64>,
        y_max: f64,
        distribution: &Normal,
    ) -> DVector<f64> {
        let n = y_mean.len();
        let y = y_mean.add_scalar(-y_max - self.xi);
        let z = y.component_div(&std);
        let cdf = DVector::from_iterator(n, z.iter().map(|&o| distribution.cdf(o)));
        let pdf = DVector::from_iterator(n, z.iter().map(|&o| distribution.pdf(o)));
        (y.component_mul(&cdf)) + (std.component_mul(&pdf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn expected_improvement_test() {
        let distribution = Normal::new(0.0, 1.0).unwrap();
        let ei = ExpectedImprovement::default();
        let y_mean = DVector::from_vec(vec![0.5]);
        let std = DVector::from_vec(vec![0.15]);
        let res = ei.apply(y_mean, std, 0.56, &distribution);
        let expected = DVector::from_vec(vec![0.034]);
        assert_abs_diff_eq!(res, expected, epsilon = 0.01);
    }
}
