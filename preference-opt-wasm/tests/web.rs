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
