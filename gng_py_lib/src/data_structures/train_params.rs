pub struct FileNames {
    pub input_set_filename: String,
    pub config_filename: String,
}

impl FileNames {
    pub fn init() -> Self {
        Self {
            input_set_filename: "NONE".to_string(),
            config_filename: "".to_string(),
        }
    }
}
#[derive(Debug)]
pub struct AlgorithmState {
    pub train_initiated: bool,
    pub dataset_initiated: bool,
    pub train_completed: bool,
    pub reshuffle_required: bool,
    pub normal_iteration: bool,
    pub create_neuron_scheduled: bool,
    pub iteration_complete: bool,
    pub debug_mode: bool,

    pub sample_id_position: usize,
    pub curr_iteration: usize,
    pub curr_sample_pos: usize,
}
impl AlgorithmState {
    pub fn init() -> Self {
        Self {
            train_initiated: false,
            dataset_initiated: false,
            train_completed: false,
            reshuffle_required: false,
            normal_iteration: false,
            create_neuron_scheduled: false,
            iteration_complete: false,
            debug_mode: false,

            sample_id_position: 0,

            curr_iteration: 0,
            curr_sample_pos: 0,
        }
    }
}
//--------------------------------------------------------------------------------------------------
pub struct TrainParams {
    pub weight_rng_min: f64,
    pub weight_rng_max: f64,

    //pub t: usize,
    pub epsilon_w: f64,
    pub epsilon_n: f64,
    pub lambda_start: f64,
    pub lambda_end: f64,
    pub train_iterations: usize,
    pub input_set_filename: String,
    pub d: f64,
    pub alpha: f64,
    pub edge_removal_age: usize,
    pub neuron_creation_interval: usize,
    pub max_train_iterations: usize,
    pub target_error: f64,
}

impl TrainParams {
    pub fn init() -> Self {
        Self {
            weight_rng_min: 0.0,
            weight_rng_max: 0.0,
            epsilon_w: 0.0,
            epsilon_n: 0.0,
            lambda_start: 0.0,
            lambda_end: 0.0,
            train_iterations: 0,
            input_set_filename: "".to_string(),
            d: 0.0,
            alpha: 0.0,
            edge_removal_age: 0,
            neuron_creation_interval: 0,
            max_train_iterations: 0,
            target_error: 0.0,
        }
    }
}
