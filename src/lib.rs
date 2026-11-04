//! A Rust implementation of the Growing Neural Gas (GNG) algorithm for unsupervised learning.
//!
//! This module provides a complete implementation of the GNG algorithm with support for
//! training, model persistence, and various utility functions for managing neural networks.

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

/// A wrapper struct for the GNG system that provides a clean API for users.
///
/// This struct encapsulates the internal [`Gng_System`] and provides a simplified
/// interface for common operations like training, configuration, and model management.
pub struct Gng {
    gng_system: internal::Gng_System,
}

impl Default for Gng {
    /// Creates a new GNG system with default parameters.
    ///
    /// Initializes the system with default configuration values and prepares it for training.
    fn default() -> Self {
        let mut gng_system = internal::Gng_System::init();
        gng_system.create_system();
        Self { gng_system }
    }
}

impl Gng {
    /// Creates a new GNG instance with default settings.
    ///
    /// # Returns
    ///
    /// A new [`Gng`] instance initialized with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads configuration parameters from a JSON file.
    ///
    /// This function reads configuration values such as input width, weight range,
    /// and training parameters from a JSON file to initialize the GNG system.
    ///
    /// # Arguments
    ///
    /// * `filename_config` - The path to the JSON configuration file
    pub fn load_config(&mut self, filename_config: &str) {
        internal::core_load_config(&mut self.gng_system, &filename_config.to_string());
    }

    /// Initializes the dataset from a vector of f64 values.
    ///
    /// This function loads sample data directly from a vector and prepares it for training,
    /// using the configured input width to properly parse each sample.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f64 values representing the dataset samples
    pub fn init_dataset_vec(&mut self, data: &Vec<f64>) {
        internal::core_init_dataset_vec(&mut self.gng_system, data);
    }

    /// Initializes the dataset from a CSV file.
    ///
    /// This function loads sample data from a CSV file and prepares it for training,
    /// using the configured input width to properly parse each sample.
    ///
    /// # Arguments
    ///
    /// * `filename_dataset` - The path to the CSV dataset file
    pub fn init_dataset(&mut self, filename_dataset: &str) {
        internal::core_init_dataset(&mut self.gng_system, &filename_dataset.to_string());
    }

    /// Runs the complete training process until maximum epochs are reached.
    ///
    /// This function manages the full training loop by iterating through epochs and samples
    /// until the maximum number of epochs is reached or the target error is achieved.
    pub fn fit(&mut self) {
        internal::core_fit(&mut self.gng_system);
    }

    /// Initializes the training process for a single epoch.
    ///
    /// This function prepares the system for training by setting up initial state and
    /// preparing for the first epoch of training.
    pub fn init_step(&mut self) {
        internal::core_init_step(&mut self.gng_system);
    }

    /// Executes a single fitting iteration (one epoch).
    ///
    /// This function performs one complete training epoch by processing all samples in
    /// the dataset and updating the network structure accordingly.
    pub fn fit_step(&mut self) {
        internal::core_fit_step(&mut self.gng_system);
    }

    /// Exports the current model to a JSON file.
    ///
    /// This function serializes all neurons and edges in the network to a structured JSON format
    /// that can be saved to disk for later loading or analysis.
    ///
    /// # Arguments
    ///
    /// * `filename_output` - The path where the JSON file will be written
    pub fn save_model_json(&mut self, filename_output: &str) {
        internal::core_save_model_json(&mut self.gng_system, filename_output.to_string());
    }

    /// Returns the current model as a JSON string.
    ///
    /// This function serializes all neurons and edges in the network to a JSON format
    /// without writing to a file, useful for immediate processing or transmission.
    ///
    /// # Returns
    ///
    /// A String containing the serialized model in JSON format
    pub fn get_model_string(&mut self) -> String {
        internal::core_get_model_string(&mut self.gng_system)
    }

    /// Returns the current neurons as (id, position) pairs.
    ///
    /// The positions are vectors stored as f64 arrays.
    ///
    /// # Returns
    ///
    /// A vector of tuples where each tuple contains (neuron_id, neuron_position)
    pub fn get_neurons(&mut self) -> Vec<(usize, Vec<f64>)> {
        internal::core_get_neurons(&mut self.gng_system)
    }

    /// Returns the current edges as (from, to) pairs.
    ///
    /// Each tuple represents a directed edge between two neurons.
    ///
    /// # Returns
    ///
    /// A vector of tuples where each tuple contains (from_neuron_id, to_neuron_id)
    pub fn get_edges(&mut self) -> Vec<(usize, usize)> {
        internal::core_get_edges(&mut self.gng_system)
    }

    /// Sets all model parameters directly.
    ///
    /// This function provides a programmatic way to configure all GNG system parameters
    /// without requiring a configuration file.
    ///
    /// # Arguments
    ///
    /// * `input_width` - The dimensionality of input samples
    /// * `weight_rng_min` - Minimum value for initial weight randomization
    /// * `weight_rng_max` - Maximum value for initial weight randomization
    /// * `edge_removal_age` - Age threshold for edge removal
    /// * `neuron_creation_interval` - Iteration interval between neuron creation
    /// * `max_epochs` - Maximum number of training epochs
    /// * `max_neurons` - Maximum number of neurons allowed in the network
    /// * `target_error` - Target error level for termination
    /// * `epsilon_w` - Weight update parameter for winner neuron
    /// * `epsilon_n` - Weight update parameter for neighbor neurons
    /// * `alpha` - Error decay factor for newly created neurons
    /// * `beta` - Global error decay factor
    /// * `rng_seed` - Random number generator seed for reproducibility
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
