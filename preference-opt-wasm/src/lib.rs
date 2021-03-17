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
    pub fn new_empty(dims: usize) -> OptimizationEngine {
        let engine = PreferenceOpt::new(dims);
        Self { engine }
    }

    pub fn new(samples: Vec<f64>, preferences: Vec<usize>, dims: usize) -> OptimizationEngine {
        let samples = samples.chunks(dims).map(|o| o.to_vec()).collect::<Vec<_>>();
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

    pub fn get_next_sample(
        &mut self,
        n_init: usize,
        n_solve: usize,
        f_prior: Option<Vec<f64>>,
    ) -> SampleData {
        let (sample1, idx1, sample2, idx2, f_prior) = self
            .engine
            .get_next_sample(f_prior.as_ref(), n_init, n_solve)
            .unwrap();
        SampleData {
            sample1,
            idx1,
            sample2,
            idx2,
            f_prior,
        }
    }

    pub fn add_preference(&mut self, preference: usize, other: usize) {
        self.engine.add_preference(preference, other);
    }

    pub fn get_optimal_values(&self) -> Vec<f64> {
        self.engine.get_optimal_values()
    }
}

#[wasm_bindgen]
pub struct SampleData {
    sample1: Vec<f64>,
    pub idx1: usize,
    sample2: Vec<f64>,
    pub idx2: usize,
    f_prior: Option<Vec<f64>>,
}

#[wasm_bindgen]
impl SampleData {
    pub fn sample1(&self) -> Vec<f64> {
        self.sample1.clone()
    }

    pub fn sample2(&self) -> Vec<f64> {
        self.sample2.clone()
    }

    pub fn f_prior(&self) -> Option<Vec<f64>> {
        self.f_prior.clone()
    }
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

    #[test]
    fn optimization() {
        let mut opt = OptimizationEngine::new_empty(2).with_same_bounds(vec![0.0, 10.0]);
        opt.engine.x.show();
        opt.engine.m.show();
        let sample = opt.get_next_sample(500, 1, None);
        opt.add_preference(1, 0);
        let _sample = opt.get_next_sample(500, 1, sample.f_prior());
        opt.engine.x.show();
        opt.engine.m.show();
        println!("Optimal: {:?}", opt.get_optimal_values());
    }
}
