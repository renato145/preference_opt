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
    pub fn new(
        samples: Vec<f64>,
        preferences: Vec<usize>,
        dimensions: usize,
    ) -> OptimizationEngine {
        let samples = samples
            .chunks(dimensions)
            .map(|o| o.to_vec())
            .collect::<Vec<_>>();
        let preferences = preferences
            .chunks(2)
            .map(|o| (o[0], o[1]))
            .collect::<Vec<_>>();
        let engine = PreferenceOpt::from_data(samples, preferences).unwrap();
        Self { engine }
    }

    pub fn with_bounds(mut self, bounds: Vec<f64>) -> Self {
        let bounds = bounds.chunks(2).map(|o| (o[0], o[1])).collect::<Vec<_>>();
        self.engine = self.engine.with_bounds(bounds).unwrap();
        self
    }

    pub fn with_same_bounds(mut self, bounds: Vec<f64>) -> Self {
        let bounds = (bounds[0], bounds[1]);
        self.engine = self.engine.with_same_bounds(bounds).unwrap();
        self
    }

    pub fn dims(&self) -> usize {
        self.engine.dims()
    }
}

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_engine() {
        let opt = OptimizationEngine::new(vec![1.0, 0.0, 2.0, 5.0], vec![1, 0], 2)
            .with_same_bounds(vec![0.0, 10.0]);
        opt.engine.x.show();
        opt.engine.m.show();
    }
}
