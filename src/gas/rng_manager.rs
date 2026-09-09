use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

pub struct RngManager {
    rng: StdRng,
}
impl RngManager {
    pub fn init(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn get_f64(&mut self, min: f64, max: f64) -> f64 {
        let res = self.rng.random_range(min..max);
        res
    }
    pub fn get_usize(&mut self, min: usize, max: usize) -> usize {
        let res = self.rng.gen_range(min..max);
        res
    }
    pub fn get_rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}
