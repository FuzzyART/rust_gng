#![allow(warnings)]
pub mod data_structures;
pub mod ecs;
pub mod gas;
pub mod handlers;

pub mod internal {
    // Re-export the original types and functions from core
    pub use crate::gas::core::{
        fit as core_fit, get_model_string as core_get_model_string,
        init_dataset as core_init_dataset, init_dataset_vec as core_init_dataset_vec,
        load_config as core_load_config, save_model_json as core_save_model_json,
        set_input_width as core_set_input_width, set_parameters as core_set_parameters,
         get_neurons as core_get_neurons, get_edges as core_get_edges,
        Handler,
    };
}

pub struct Gng {
    cont_params: internal::Handler,
}

impl Default for Gng {
    fn default() -> Self {
        let mut cont_params = internal::Handler::init();
        cont_params.create_system();
        Self { cont_params }
    }
}

impl Gng {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_config(&mut self, filename_config: &str) {
        internal::core_load_config(&mut self.cont_params, &filename_config.to_string());
    }

    pub fn init_dataset_vec(&mut self, data: &Vec<f64>) {
        internal::core_init_dataset_vec(&mut self.cont_params, data);
    }

    pub fn init_dataset(&mut self, filename_dataset: &str) {
        internal::core_init_dataset(&mut self.cont_params, &filename_dataset.to_string());
    }

    pub fn fit(&mut self) {
        internal::core_fit(&mut self.cont_params);
    }
    pub fn save_model_json(&mut self, filename_output: &str) {
        internal::core_save_model_json(&mut self.cont_params, filename_output.to_string());
    }
    pub fn get_model_string(&mut self) -> String {
        internal::core_get_model_string(&mut self.cont_params)
    }

//    pub fn get_neurons(&mut self) -> Vec<Neuron> {
//         internal::core_get_neurons(&mut self.cont_params)
//     }
//
     pub fn get_neurons(&mut self) -> Vec<(usize,Vec<f64>)> {
             internal::core_get_neurons(&mut self.cont_params)
         }

         pub fn get_edges(&mut self) -> Vec<(usize,usize)> {
             internal::core_get_edges(&mut self.cont_params)
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
    ) {
        internal::core_set_parameters(
            &mut self.cont_params,
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
        );
    }
}
