//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use preference_opt_wasm::OptimizationEngine;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn pass() {
    assert_eq!(1 + 1, 2);
}

#[wasm_bindgen_test]
fn get_engine() {
    let opt = OptimizationEngine::new();
    assert_eq!(opt.tt(), 1e-5);
}