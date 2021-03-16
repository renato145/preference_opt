//! This repository closely follows the implementation on [GPro].
//!
//! [GPro]: https://github.com/chariff/GPro

pub mod optimization;
pub mod structures;
pub mod kernels;
pub mod posterior;
pub mod acquisition;
pub mod lbfgs_opt;

pub use crate::optimization::*;
pub use structures::*;