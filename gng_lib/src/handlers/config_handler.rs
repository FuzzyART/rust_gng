use crate::ecs::manager;
use crate::gas::json_reader;
//use crate::ecs::DataStructures::Config;

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
    pub max_train_iterations: usize,
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
            max_train_iterations: 0,
            target_error: 0.0,
            epsilon_w: 0.0,
            epsilon_n: 0.0,
            alpha: 0.0,
            beta: 0.0,
        }
    }
}
// Config Struct
//==================================================================================================
// Config Handler
pub struct ConfigHandler {
    pub config_man: manager::EntityManager<Config>,
    pub filename_config: String,
    pub filename_input_set: String,
    pub input_width: usize,
}
impl ConfigHandler {
    pub fn init() -> Self {
        Self {
            config_man: manager::EntityManager::new(),

            filename_config: "".to_string(),
            filename_input_set: "".to_string(),
            input_width: 0,
        }
    }
    pub fn create_config(&mut self) {
        self.config_man.create(Config::init());
    }

    pub fn load_config(&mut self, filename_config: &String) {
        let reader = json_reader::read_file(&filename_config).unwrap();

        self.set_input_width(json_reader::read_val_usize(
            &reader,
            "config",
            "input_width",
        ));

        self.set_weight_rng_min(json_reader::read_val_f64(
            &reader,
            "config",
            "weight_rng_min",
        ));
        self.set_weight_rng_max(json_reader::read_val_f64(
            &reader,
            "config",
            "weight_rng_max",
        ));
        self.set_edge_removal_age(json_reader::read_val_usize(
            &reader,
            "config",
            "edge_removal_age",
        ));
        self.set_neuron_creation_interval(json_reader::read_val_usize(
            &reader,
            "config",
            "neuron_creation_interval",
        ));
        self.set_max_train_iterations(json_reader::read_val_usize(
            &reader,
            "config",
            "max_train_iterations",
        ));
        self.set_target_error(json_reader::read_val_f64(&reader, "config", "target_error"));
        self.set_epsilon_w(json_reader::read_val_f64(&reader, "config", "epsilon_w"));
        self.set_epsilon_n(json_reader::read_val_f64(&reader, "config", "epsilon_n"));

        //----------------------------------------
    }

    //----------------------------------------------------------
    pub fn set_config_filename(&mut self, filename_config: String) {
        if let Some(val) = self.config_man.get_mut(0) {
            val.config_filename = filename_config;
        }
    }
    pub fn get_config_filename(&self) -> Option<&String> {
        self.config_man.get(0).map(|val| &val.config_filename)
    }
    //----------------------------------------------------------
    pub fn set_input_set_filename(&mut self, filename: String) {
        if let Some(val) = self.config_man.get_mut(0) {
            val.input_set_filename = filename;
        }
    }
    pub fn get_input_set_filename(&self) -> Option<&String> {
        self.config_man.get(0).map(|val| &val.input_set_filename)
    }
    //----------------------------------------------------------
    pub fn set_input_width(&mut self, val: usize) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.input_width = val;
        }
    }
    pub fn read_input_width(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_usize(&reader, "config", "input_width");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.input_width = val;
        }
    }
    pub fn get_input_width(&self) -> &usize {
        self.config_man
            .get(0)
            .map(|val| &val.input_width)
            .expect("config not initiated")
    }
    //----------------------------------------------------------

    //----------------------------------------------------------
    pub fn set_weight_rng_min(&mut self, val: f64) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.weight_rng_min = val;
        }
    }
    pub fn read_weight_rng_min(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_f64(&reader, "config", "weight_rng_min");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.weight_rng_min = val;
        }
    }
    pub fn get_weight_rng_min(&self) -> &f64 {
        self.config_man
            .get(0)
            .map(|val| &val.weight_rng_min)
            .expect("config not initiated")
    }

    //----------------------------------------------------------
    pub fn set_weight_rng_max(&mut self, val: f64) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.weight_rng_max = val;
        }
    }
    pub fn read_weight_rng_max(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_f64(&reader, "config", "weight_rng_max");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.weight_rng_max = val;
        }
    }
    pub fn get_weight_rng_max(&self) -> &f64 {
        self.config_man
            .get(0)
            .map(|val| &val.weight_rng_max)
            .expect("config not initiated")
    }

    //----------------------------------------------------------
    pub fn set_edge_removal_age(&mut self, val: usize) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.edge_removal_age = val;
        }
    }
    pub fn read_edge_removal_age(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_usize(&reader, "config", "edge_removal_age");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.edge_removal_age = val;
        }
    }
    pub fn get_edge_removal_age(&self) -> Option<&usize> {
        self.config_man.get(0).map(|val| &val.edge_removal_age)
    }

    //----------------------------------------------------------
    pub fn set_neuron_creation_interval(&mut self, val: usize) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.neuron_creation_interval = val;
        }
    }
    pub fn read_neuron_creation_interval(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_usize(&reader, "config", "neuron_creation_interval");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.neuron_creation_interval = val;
        }
    }
    pub fn get_neuron_creation_interval(&self) -> &usize {
        self.config_man
            .get(0)
            .map(|val| &val.neuron_creation_interval)
            .expect("neuron creation intervall not found")
    }

    //----------------------------------------------------------
    pub fn set_max_train_iterations(&mut self, val: usize) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.max_train_iterations = val;
        }
    }
    pub fn get_max_train_iterations(&self) -> &usize {
        self.config_man
            .get(0)
            .map(|val| &val.max_train_iterations)
            .expect("max train iteration not found")
    }
    pub fn read_max_train_iterations(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_usize(&reader, "config", "max_train_iterations");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.max_train_iterations = val;
        }
    }

    //----------------------------------------------------------
    pub fn set_target_error(&mut self, val: f64) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.target_error = val;
        }
    }
    pub fn read_target_error(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_f64(&reader, "config", "target_error");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.target_error = val;
        }
    }
    pub fn get_target_error(&self) -> Option<&f64> {
        self.config_man.get(0).map(|val| &val.target_error)
    }

    //----------------------------------------------------------
    pub fn set_epsilon_w(&mut self, val: f64) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.epsilon_w = val;
        }
    }
    pub fn read_epsilon_w(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_f64(&reader, "config", "epsilon_w");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.epsilon_w = val;
        }
    }
    pub fn get_epsilon_w(&self) -> &f64 {
        self.config_man
            .get(0)
            .map(|val| &val.epsilon_w)
            .expect("epsilon_w not found")
    }
    pub fn set_alpha(&mut self, val: f64) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.alpha = val;
        }
    }
    pub fn get_alpha(&self) -> &f64 {
        self.config_man
            .get(0)
            .map(|val| &val.alpha)
            .expect("alpha not found")
    }

    pub fn set_beta(&mut self, val: f64) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.beta = val;
        }
    }
    pub fn get_beta(&self) -> &f64 {
        self.config_man
            .get(0)
            .map(|val| &val.beta)
            .expect("d not found")
    }

    //----------------------------------------------------------
    pub fn set_epsilon_n(&mut self, val: f64) {
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.epsilon_n = val;
        }
    }
    pub fn read_epsilon_n(&mut self, filename: &str) {
        let reader = json_reader::read_file(filename).unwrap();
        let val = json_reader::read_val_f64(&reader, "config", "epsilon_n");
        if let Some(obj) = self.config_man.get_mut(0) {
            obj.epsilon_n = val;
        }
    }
    pub fn get_epsilon_n(&self) -> &f64 {
        self.config_man
            .get(0)
            .map(|val| &val.epsilon_n)
            .expect("epsilon_n not found")
    }
}
