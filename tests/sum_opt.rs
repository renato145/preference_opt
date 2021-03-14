// use anyhow::Result;
// use preference_opt::*;

// bounds = {'x1': (0, 10), 'x2': (0, 10)}
// X = np.array([[1, 0], [2, 5]]).reshape(-1, 2)
// M = np.array([1, 0]).reshape(-1, 2)

// #[test]
// fn test_opt() -> Result<()> {
//     let samples = vec![vec![1.0, 0.0], vec![2.0, 5.0]];
//     let preferences = vec![(1, 0)];
//     let opt = PreferenceOpt::from_data(samples, preferences)?.with_same_bounds((0.0, 10.0));
//     // let func= |o: &[f64]| o.iter().sum();
//     // opt.optimize_fn(func, 1);
//     Ok(())
// }
