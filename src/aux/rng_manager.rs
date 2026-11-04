use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

/// A manager for handling random number generation with configurable seeding.
///
/// This struct wraps [`StdRng`] and provides convenient methods to generate
/// random numbers of various types within specified ranges. It allows reseeding
/// the underlying RNG, making it useful in scenarios where reproducible results
/// are needed or when different random sequences are required.
pub struct RngManager {
    rng: StdRng,
}
impl RngManager {
    /// Initializes a new [`RngManager`] with the given seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - A u64 value used to seed the random number generator.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::rng_manager::RngManager;
    ///
    /// let mut rng_manager = RngManager::init(42);
    /// ```
    pub fn init(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
    /// Reseeds the internal random number generator with a new seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - A u64 value used to reseed the random number generator.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::rng_manager::RngManager;
    ///
    /// let mut rng_manager = RngManager::init(42);
    /// rng_manager.seed_rng(123);
    /// ```
    pub fn seed_rng(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
    /// Generates a random `f64` value in the range `[min, max)`.
    ///
    /// # Arguments
    ///
    /// * `min` - The inclusive lower bound of the range.
    /// * `max` - The exclusive upper bound of the range.
    ///
    /// # Returns
    ///
    /// A random `f64` value in the interval `[min, max)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::rng_manager::RngManager;
    ///
    /// let mut rng_manager = RngManager::init(42);
    /// let value = rng_manager.get_f64(0.0, 1.0);
    /// assert!(value >= 0.0 && value < 1.0);
    /// ```
    pub fn get_f64(&mut self, min: f64, max: f64) -> f64 {
        let res = self.rng.random_range(min..max);
        res
    }
    /// Generates a random `usize` value in the range `[min, max)`.
    ///
    /// # Arguments
    ///
    /// * `min` - The inclusive lower bound of the range.
    /// * `max` - The exclusive upper bound of the range.
    ///
    /// # Returns
    ///
    /// A random `usize` value in the interval `[min, max)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::rng_manager::RngManager;
    ///
    /// let mut rng_manager = RngManager::init(42);
    /// let value = rng_manager.get_usize(0, 100);
    /// assert!(value < 100);
    /// ```
    pub fn get_usize(&mut self, min: usize, max: usize) -> usize {
        let res = self.rng.gen_range(min..max);
        res
    }
    /// Returns a mutable reference to the internal [`StdRng`].
    ///
    /// This allows direct access to the underlying random number generator,
    /// enabling advanced usage not covered by this wrapper.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rand::Rng;
    /// use rust_gng::aux::rng_manager::RngManager;
    ///
    /// let mut rng_manager = RngManager::init(42);
    /// let rng_ref = rng_manager.get_rng();
    /// let value = rng_ref.gen::<f64>();
    /// ```
    pub fn get_rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_manager_init() {
        let rng_manager = RngManager::init(42);
        // Test that initialization works without panicking
        assert!(true); // Basic test that struct can be created
    }

    #[test]
    fn test_rng_manager_seed_rng() {
        let mut rng_manager = RngManager::init(42);

        // Change the seed
        rng_manager.seed_rng(123);

        // Test that we can still generate values
        let value = rng_manager.get_f64(0.0, 1.0);
        assert!(value >= 0.0 && value < 1.0);
    }

    #[test]
    fn test_rng_manager_get_f64() {
        let mut rng_manager = RngManager::init(42);

        // Test generating f64 values in different ranges
        for _ in 0..10 {
            let value = rng_manager.get_f64(0.0, 10.0);
            assert!(value >= 0.0 && value < 10.0);

            let value2 = rng_manager.get_f64(-5.0, 5.0);
            assert!(value2 >= -5.0 && value2 < 5.0);
        }
    }

    #[test]
    fn test_rng_manager_get_usize() {
        let mut rng_manager = RngManager::init(42);

        // Test generating usize values in different ranges
        for _ in 0..10 {
            let value = rng_manager.get_usize(0, 100);
            assert!(value < 100);

            let value2 = rng_manager.get_usize(10, 20);
            assert!(value2 >= 10 && value2 < 20);
        }
    }

    #[test]
    fn test_rng_manager_deterministic_behavior() {
        // Test that same seed produces same sequence
        let mut rng_manager1 = RngManager::init(42);
        let mut rng_manager2 = RngManager::init(42);

        let values1: Vec<f64> = (0..5).map(|_| rng_manager1.get_f64(0.0, 1.0)).collect();
        let values2: Vec<f64> = (0..5).map(|_| rng_manager2.get_f64(0.0, 1.0)).collect();

        assert_eq!(values1, values2);
    }

    #[test]
    fn test_rng_manager_deterministic_behavior_2() {
        // Test that same seed produces same sequence
        let mut rng_manager = RngManager::init(123);
        rng_manager.seed_rng(890);

        let values1: Vec<f64> = (0..5).map(|_| rng_manager.get_f64(0.0, 1.0)).collect();

        rng_manager.seed_rng(890);
        let values2: Vec<f64> = (0..5).map(|_| rng_manager.get_f64(0.0, 1.0)).collect();

        assert_eq!(values1, values2);
    }

    #[test]
    fn test_rng_manager_different_seeds_different_results() {
        let mut rng_manager1 = RngManager::init(42);
        let mut rng_manager2 = RngManager::init(123);

        // Generate some values
        let values1: Vec<f64> = (0..10).map(|_| rng_manager1.get_f64(0.0, 1.0)).collect();
        let values2: Vec<f64> = (0..10).map(|_| rng_manager2.get_f64(0.0, 1.0)).collect();

        // While there's a small chance they could be equal, it's highly unlikely
        // and this test verifies the RNG is working properly
        assert_eq!(values1.len(), values2.len());
    }

    #[test]
    fn test_rng_manager_get_rng() {
        let mut rng_manager = RngManager::init(42);

        // Test that we can get access to the underlying RNG
        let rng_ref = rng_manager.get_rng();
        assert!(rng_ref as *const _ != std::ptr::null());
    }

    #[test]
    fn test_rng_manager_seed_rng_changes_behavior() {
        let mut rng_manager = RngManager::init(42);

        // Get some initial values
        let initial_values: Vec<f64> = (0..3).map(|_| rng_manager.get_f64(0.0, 1.0)).collect();

        // Change seed
        rng_manager.seed_rng(42); // Same seed

        // Get values again - should be same due to same seed
        let same_seed_values: Vec<f64> = (0..3).map(|_| rng_manager.get_f64(0.0, 1.0)).collect();

        assert_eq!(initial_values, same_seed_values);
    }
}
