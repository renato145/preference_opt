use anyhow::Result;
use preference_opt::*;

fn main() -> Result<()> {
    let samples = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
    let preferences = vec![(1, 0)];
    let mut opt = PreferenceOpt::from_data(samples, preferences)?.with_same_bounds((0.0, 10.0))?;
    println!("Try to choose the examples with the higher values...");
    let (optimal_values, _f_posterior) = opt.interactive_optimization(100, None, 500, 1)?;
    println!("optimal_values -> {}", optimal_values);
    // println!("f_posterior -> {}", f_posterior);
    // opt.x.show();
    // opt.m.show();
    Ok(())
}
