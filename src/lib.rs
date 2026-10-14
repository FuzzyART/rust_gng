#![allow(warnings)]
pub mod aux;
pub mod components;
pub mod ecs;
pub mod gas;
pub mod handlers;

pub mod internal {
    // Re-export the original types and functions from core
    pub use crate::gas::core::{
        fit as core_fit, fit_step as core_fit_step, get_edges as core_get_edges,
        get_model_string as core_get_model_string, get_neurons as core_get_neurons,
        init_dataset as core_init_dataset, init_dataset_vec as core_init_dataset_vec,
        init_step as core_init_step, load_config as core_load_config,
        save_model_json as core_save_model_json, set_input_width as core_set_input_width,
        set_parameters as core_set_parameters,
    };
    pub use crate::gas::gng_system::Gng_System;
}

pub struct Gng {
    gng_system: internal::Gng_System,
}

impl Default for Gng {
    fn default() -> Self {
        let mut gng_system = internal::Gng_System::init();
        gng_system.create_system();
        Self { gng_system }
    }
}

impl Gng {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_config(&mut self, filename_config: &str) {
        internal::core_load_config(&mut self.gng_system, &filename_config.to_string());
    }

    pub fn init_dataset_vec(&mut self, data: &Vec<f64>) {
        internal::core_init_dataset_vec(&mut self.gng_system, data);
    }

    pub fn init_dataset(&mut self, filename_dataset: &str) {
        internal::core_init_dataset(&mut self.gng_system, &filename_dataset.to_string());
    }

    pub fn fit(&mut self) {
        internal::core_fit(&mut self.gng_system);
    }
    pub fn init_step(&mut self) {
        internal::core_init_step(&mut self.gng_system);
    }
    pub fn fit_step(&mut self) {
        internal::core_fit_step(&mut self.gng_system);
    }
    pub fn save_model_json(&mut self, filename_output: &str) {
        internal::core_save_model_json(&mut self.gng_system, filename_output.to_string());
    }
    pub fn get_model_string(&mut self) -> String {
        internal::core_get_model_string(&mut self.gng_system)
    }

    pub fn get_neurons(&mut self) -> Vec<(usize, Vec<f64>)> {
        internal::core_get_neurons(&mut self.gng_system)
    }

    pub fn get_edges(&mut self) -> Vec<(usize, usize)> {
        internal::core_get_edges(&mut self.gng_system)
    }

    pub fn set_parameters(
        &mut self,
        input_width: usize,
        weight_rng_min: f64,
        weight_rng_max: f64,
        edge_removal_age: usize,
        neuron_creation_interval: usize,
        max_epochs: usize,
        max_neurons: usize,
        target_error: f64,
        epsilon_w: f64,
        epsilon_n: f64,
        alpha: f64,
        beta: f64,
        rng_seed: u64,
    ) {
        internal::core_set_parameters(
            &mut self.gng_system,
            input_width,
            weight_rng_min,
            weight_rng_max,
            edge_removal_age,
            neuron_creation_interval,
            max_epochs,
            max_neurons,
            target_error,
            epsilon_w,
            epsilon_n,
            alpha,
            beta,
            rng_seed,
        );
        self.gng_system.rng_manager.seed_rng(rng_seed);
    }
}
