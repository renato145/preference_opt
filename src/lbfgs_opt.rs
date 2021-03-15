//! L-BFGS optimization

use lbfgs::Lbfgs;
use nalgebra::{DMatrix, RowDVector};
use optimization::{Func, Function, Function1, NumericalDifferentiation};

/// Minimize vector in base of a function.
///
///
/// # Examples
///
/// ```
/// # use preference_opt::lbfgs_opt::minimize;
/// let x = nalgebra::RowDVector::from_vec(vec![0.1, 0.2, 0.3]);
/// let bounds = vec![(0.0, 10.0), (-5.0, 15.0), (0.0, 100.0)];
/// let (best_x, best_value) = minimize(|o| -o.sum(), x, &bounds);
/// assert_eq!(best_value, -125.0);
/// ```
pub fn minimize<F: Fn(DMatrix<f64>) -> f64>(
    func: F,
    x: RowDVector<f64>,
    bounds: &[(f64, f64)],
) -> (DMatrix<f64>, f64) {
    let total_size = x.ncols();
    let objective =
        NumericalDifferentiation::new(Func(|x| func(DMatrix::from_vec(1, total_size, x.to_vec()))));

    let tolerance = 1e-8;
    let lbfgs_memory = 5;
    let max_iter = 500;

    let mut lbfgs = Lbfgs::new(total_size, lbfgs_memory).with_sy_epsilon(1e-8);

    // Initialize hessian at the origin.
    lbfgs.update_hessian(
        &objective.gradient(&vec![0.0; total_size]),
        &vec![0.0; total_size],
    );

    let mut gradient: Vec<f64>;
    let mut coef = x.iter().cloned().collect::<Vec<_>>();
    for _ in 0..max_iter {
        // calculate gradient from coefficients
        gradient = objective.gradient(&coef);
        // update curvature information
        lbfgs.update_hessian(&gradient, &coef);
        // determine next direction
        lbfgs.apply_hessian(&mut gradient);

        let mut converged = true;
        for (i, (low, high)) in (0..gradient.len()).zip(bounds) {
            // Todo: We need to discover (and possibly mitigate?) divergence.
            if gradient[i].abs() > tolerance {
                converged = false;
            }
            coef[i] = (coef[i] - gradient[i]).clamp(*low, *high);
        }

        if converged {
            break;
        }
    }
    let best_value = objective.value(&coef);
    let best = DMatrix::from_vec(1, total_size, coef);
    (best, best_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimize_test() {
        let x = RowDVector::from_vec(vec![0.1, 0.2, 0.3]);
        let bounds = vec![(0.0, 10.0), (-5.0, 15.0), (0.0, 100.0)];
        let (best_x, best_value) = minimize(|o| -o.sum(), x, &bounds);
        assert_eq!(best_value, -125.0);
        println!("{:?} {}", best_x, best_value);
    }
}
