//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

use preference_opt_wasm::OptimizationEngine;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn get_engine() {
    let opt = OptimizationEngine::new(vec![1.0, 0.0, 2.0, 5.0], vec![1, 0], 2);
    assert_eq!(opt.dims(), 2);
}

#[wasm_bindgen_test]
fn optimization() {
    let mut opt = OptimizationEngine::new_empty(3).with_same_bounds(vec![0.0, 255.0]);
    let mut prior = None;
    for _ in 0..15 {
        let sample = opt.get_next_sample(1, 1, prior.clone());
        opt.add_preference(1, 0);
        prior = sample.f_prior();
    }
    println!("Optimal: {:?}", opt.get_optimal_values());
}
