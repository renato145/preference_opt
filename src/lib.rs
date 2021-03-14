//! This repository will closely follow the implementation on [GPro].
//!
//! [GPro]: https://github.com/chariff/GPro
pub mod optimization;
pub mod structures;
pub mod kernels;
pub mod posterior;
pub mod acquisition;

pub use optimization::*;
pub use structures::*;