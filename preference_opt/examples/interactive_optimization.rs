use anyhow::Result;
use clap::Clap;
use preference_opt::*;

#[derive(Clap)]
struct Opts {
    #[clap(short, long, default_value = "0")]
    lower_bound: f64,
    #[clap(short, long, default_value = "100")]
    upper_bound: f64,
    #[clap(long, default_value = "500")]
    n_init: usize,
    #[clap(long, default_value = "1")]
    n_solve: usize,
}

fn main() -> Result<()> {
    let opts: Opts = Opts::parse();
    let mut opt = PreferenceOpt::new(1).with_same_bounds((opts.lower_bound, opts.upper_bound))?;
    println!("Try to figure some numbers in your mind and see how many tries does it take to achieve it:");
    println!(
        "(numbers are between {:.2} and {:.2})",
        opts.lower_bound, opts.upper_bound
    );
    let (optimal_values, _f_posterior) =
        opt.interactive_optimization(100, None, opts.n_init, opts.n_solve)?;
    println!("optimal_values -> {}", optimal_values);
    // println!("f_posterior -> {}", f_posterior);
    // opt.x.show();
    // opt.m.show();
    Ok(())
}
