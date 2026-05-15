use crate::ecs::manager;
use crate::gas::json_reader;

//==================================================================================================
// Config Struct
#[derive(Debug)]
pub struct Config {
    pub config_filename: String,
    pub input_set_filename: String,
    pub input_width: usize,
    pub weight_rng_min: f64,
    pub weight_rng_max: f64,
    pub edge_removal_age: usize,
    pub neuron_creation_interval: usize,
    pub max_epochs: usize,
    pub max_neurons: usize,
    pub target_error: f64,
    pub epsilon_w: f64,
    pub epsilon_n: f64,
    pub alpha: f64,
    pub beta: f64,
}
impl Config {
    pub fn init() -> Self {
        Self {
            config_filename: "".to_string(),
            input_set_filename: "".to_string(),
            input_width: 0,

            weight_rng_min: 0.0,
            weight_rng_max: 0.0,
            edge_removal_age: 0,
            neuron_creation_interval: 0,
            max_epochs: 0,
            max_neurons: 0,
            target_error: 0.0,
            epsilon_w: 0.0,
            epsilon_n: 0.0,
            alpha: 0.0,
            beta: 0.0,
        }
    }
    //    pub fn load_config(&mut self,filename_config: &String) {
    //        let reader = json_reader::read_file(&filename_config).unwrap();
    //
    //        self.set_input_width(json_reader::read_val_usize(
    //            &reader,
    //            "config",
    //            "input_width",
    //        ));
    //}

    //----------------------------------------------------------
    // --- Setters ---

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
        self.input_width = input_width;
        self.weight_rng_min = weight_rng_min;
        self.weight_rng_max = weight_rng_max;
        self.edge_removal_age = edge_removal_age;
        self.neuron_creation_interval = neuron_creation_interval;
        self.max_epochs = max_epochs;
        self.max_neurons = max_neurons;
        self.target_error = target_error;
        self.epsilon_w = epsilon_w;
        self.epsilon_n = epsilon_n;
        self.alpha = alpha;
        self.beta = beta;
    }

    //    pub fn set_config_filename(&mut self, filename: String) {
    //        self.config_filename = filename;
    //    }

    //    pub fn set_input_set_filename(&mut self, filename: String) {
    //        self.input_set_filename = filename;
    //    }

    //    pub fn set_input_width(&mut self, val: usize) {
    //        self.input_width = val;
    //    }
    //
    //    pub fn set_weight_rng_min(&mut self, val: f64) {
    //        self.weight_rng_min = val;
    //    }
    //
    //    pub fn set_weight_rng_max(&mut self, val: f64) {
    //        self.weight_rng_max = val;
    //    }
    //
    //    pub fn set_edge_removal_age(&mut self, val: usize) {
    //        self.edge_removal_age = val;
    //    }
    //
    //    pub fn set_neuron_creation_interval(&mut self, val: usize) {
    //        self.neuron_creation_interval = val;
    //    }
    //
    //    pub fn set_max_epochs(&mut self, val: usize) {
    //        self.max_epochs = val;
    //    }
    //    pub fn set_max_neurons(&mut self, val: usize) {
    //        self.max_neurons = val;
    //    }
    //
    //    pub fn set_target_error(&mut self, val: f64) {
    //        self.target_error = val;
    //    }
    //
    //    pub fn set_epsilon_w(&mut self, val: f64) {
    //        self.epsilon_w = val;
    //    }
    //
    //    pub fn set_epsilon_n(&mut self, val: f64) {
    //        self.epsilon_n = val;
    //    }
    //
    //    pub fn set_alpha(&mut self, val: f64) {
    //        self.alpha = val;
    //    }
    //
    //    pub fn set_beta(&mut self, val: f64) {
    //        self.beta = val;
    //    }

    // --- Getters (return Option<&T> for safety) ---

    //    pub fn get_config_filename(&self) -> &String {
    //        &self.config_filename
    //    }
    //
    //    pub fn get_input_set_filename(&self) -> &String {
    //        &self.input_set_filename
    //    }
    //
    //    pub fn get_input_width(&self) -> usize {
    //        self.input_width
    //    }
    //
    //    pub fn get_weight_rng_min(&self) -> f64 {
    //        self.weight_rng_min
    //    }
    //
    //    pub fn get_weight_rng_max(&self) -> f64 {
    //        self.weight_rng_max
    //    }
    //
    //    pub fn get_edge_removal_age(&self) -> usize {
    //        self.edge_removal_age
    //    }
    //
    //    pub fn get_neuron_creation_interval(&self) -> usize {
    //        self.neuron_creation_interval
    //    }
    //
    //    pub fn get_max_epochs(&self) -> usize {
    //        self.max_epochs
    //    }
    //
    //    pub fn get_max_neurons(&self) -> usize {
    //        self.max_neurons
    //    }
    //
    //    pub fn get_target_error(&self) -> f64 {
    //        self.target_error
    //    }
    //
    //    pub fn get_epsilon_w(&self) -> f64 {
    //        self.epsilon_w
    //    }
    //
    //    pub fn get_epsilon_n(&self) -> f64 {
    //        self.epsilon_n
    //    }
    //
    //    pub fn get_alpha(&self) -> f64 {
    //        self.alpha
    //    }
    //
    //    pub fn get_beta(&self) -> f64 {
    //        self.beta
    //    }

    // --- Reader (loads from JSON file) ---

    pub fn read_config(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();

        self.input_width = json_reader::read_val_usize(&reader, "config", "input_width");
        self.weight_rng_min = json_reader::read_val_f64(&reader, "config", "weight_rng_min");
        self.weight_rng_max = json_reader::read_val_f64(&reader, "config", "weight_rng_max");
        self.edge_removal_age = json_reader::read_val_usize(&reader, "config", "edge_removal_age");
        self.neuron_creation_interval =
            json_reader::read_val_usize(&reader, "config", "neuron_creation_interval");
        self.max_epochs = json_reader::read_val_usize(&reader, "config", "max_epochs");
        self.target_error = json_reader::read_val_f64(&reader, "config", "target_error");
        self.epsilon_w = json_reader::read_val_f64(&reader, "config", "epsilon_w");
        self.epsilon_n = json_reader::read_val_f64(&reader, "config", "epsilon_n");
    }

    // --- Convenience: load_config (renamed/updated from original) ---

    pub fn load_config(&mut self, filename_config: &String) {
        self.read_config(filename_config);
    }
}
