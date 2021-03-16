mod utils;

use preference_opt::PreferenceOpt;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct OptimizationEngine {
    engine: PreferenceOpt,
}

#[wasm_bindgen]
impl OptimizationEngine {
    pub fn new(// samples: Vec<Vec<f64>>,
        // preferences: Vec<(usize, usize)>,
    ) -> OptimizationEngine {
        let samples = vec![vec![1.0, 0.0], vec![2.0, 5.0]];
        let preferences = vec![(1, 0)];
        let engine = PreferenceOpt::from_data(samples, preferences).unwrap();
        Self { engine }
    }

    pub fn tt(&self) -> f64 {
        self.engine.alpha
    }
}

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, preference-opt-wasm!");
}
