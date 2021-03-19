use anyhow::Result;
use preference_opt::*;

fn main() -> Result<()> {
    let mut opt = PreferenceOpt::new(1).with_same_bounds((0.0, 100.0))?;
    println!("Try to figure some numbers in your mind and see how many tries does it take to achieve it:");
    println!("(numbers are between 0 and 10)");
    let (optimal_values, _f_posterior) = opt.interactive_optimization(100, None, 500, 1)?;
    println!("optimal_values -> {}", optimal_values);
    // println!("f_posterior -> {}", f_posterior);
    // opt.x.show();
    // opt.m.show();
    Ok(())
}
