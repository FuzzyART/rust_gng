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
){
    self.input_width = input_width;
    self.weight_rng_min = weight_rng_min;
    self.weight_rng_max = weight_rng_max;
    self.edge_removal_age = edge_removal_age;
    self.neuron_creation_interval = neuron_creation_interval;
    self.max_epochs = max_epochs;
    self.max_neurons= max_neurons;
    self.target_error = target_error;
    self.epsilon_w = epsilon_w;
    self.epsilon_n = epsilon_n;
    self.alpha = alpha;
    self.beta = beta;

}

   pub fn set_config_filename(&mut self, filename: String) {
       self.config_filename = filename;
   }

   pub fn set_input_set_filename(&mut self, filename: String) {
       self.input_set_filename = filename;
   }

   pub fn set_input_width(&mut self, val: usize) {
       self.input_width = val;
   }

   pub fn set_weight_rng_min(&mut self, val: f64) {
       self.weight_rng_min = val;
   }

   pub fn set_weight_rng_max(&mut self, val: f64) {
       self.weight_rng_max = val;
   }

   pub fn set_edge_removal_age(&mut self, val: usize) {
       self.edge_removal_age = val;
   }

   pub fn set_neuron_creation_interval(&mut self, val: usize) {
       self.neuron_creation_interval = val;
   }

   pub fn set_max_epochs(&mut self, val: usize) {
       self.max_epochs = val;
   }
   pub fn set_max_neurons(&mut self, val:usize){

       self.max_neurons = val;
   }

   pub fn set_target_error(&mut self, val: f64) {
       self.target_error = val;
   }

   pub fn set_epsilon_w(&mut self, val: f64) {
       self.epsilon_w = val;
   }

   pub fn set_epsilon_n(&mut self, val: f64) {
       self.epsilon_n = val;
   }

   pub fn set_alpha(&mut self, val: f64) {
       self.alpha = val;
   }

   pub fn set_beta(&mut self, val: f64) {
       self.beta = val;
   }

   // --- Getters (return Option<&T> for safety) ---

   pub fn get_config_filename(&self) -> &String {
       &self.config_filename
   }

   pub fn get_input_set_filename(&self) -> &String {
       &self.input_set_filename
   }

   pub fn get_input_width(&self) -> usize {
       self.input_width
   }

   pub fn get_weight_rng_min(&self) -> f64 {
       self.weight_rng_min
   }

   pub fn get_weight_rng_max(&self) -> f64 {
       self.weight_rng_max
   }

   pub fn get_edge_removal_age(&self) -> usize {
       self.edge_removal_age
   }

   pub fn get_neuron_creation_interval(&self) -> usize {
       self.neuron_creation_interval
   }

   pub fn get_max_epochs(&self) -> usize {
       self.max_epochs
   }

   pub fn get_max_neurons(&self) -> usize {
       self.max_neurons
   }

   pub fn get_target_error(&self) -> f64 {
       self.target_error
   }

   pub fn get_epsilon_w(&self) -> f64 {
       self.epsilon_w
   }

   pub fn get_epsilon_n(&self) -> f64 {
       self.epsilon_n
   }

   pub fn get_alpha(&self) -> f64 {
       self.alpha
   }

   pub fn get_beta(&self) -> f64 {
       self.beta
   }

   // --- Reader (loads from JSON file) ---

   pub fn read_config(&mut self, filename: &str) {
       let reader = json_reader::read_file(filename).unwrap();

       self.set_input_width(json_reader::read_val_usize(&reader, "config", "input_width"));
       self.set_weight_rng_min(json_reader::read_val_f64(&reader, "config", "weight_rng_min"));
       self.set_weight_rng_max(json_reader::read_val_f64(&reader, "config", "weight_rng_max"));
       self.set_edge_removal_age(json_reader::read_val_usize(&reader, "config", "edge_removal_age"));
       self.set_neuron_creation_interval(json_reader::read_val_usize(&reader, "config", "neuron_creation_interval"));
       self.set_max_epochs(json_reader::read_val_usize(&reader, "config", "max_epochs"));
       self.set_target_error(json_reader::read_val_f64(&reader, "config", "target_error"));
       self.set_epsilon_w(json_reader::read_val_f64(&reader, "config", "epsilon_w"));
       self.set_epsilon_n(json_reader::read_val_f64(&reader, "config", "epsilon_n"));
       // Note: alpha and beta are not in the original JSON reader calls in ConfigHandler
       // If they are in the JSON, add them here:
       // self.set_alpha(json_reader::read_val_f64(&reader, "config", "alpha"));
       // self.set_beta(json_reader::read_val_f64(&reader, "config", "beta"));
   }

   // --- Convenience: load_config (renamed/updated from original) ---

   pub fn load_config(&mut self, filename_config: &String) {
       self.read_config(filename_config);
   }












}
// Config Struct
//==================================================================================================
// Config Handler
//pub struct ConfigHandler {
//    pub config_man: manager::EntityManager<Config>,
//    pub filename_config: String,
//    pub filename_input_set: String,
//    pub input_width: usize,
//}
//impl ConfigHandler {
//    pub fn init() -> Self {
//        Self {
//            config_man: manager::EntityManager::new(),
//
//            filename_config: "".to_string(),
//            filename_input_set: "".to_string(),
//            input_width: 0,
//        }
//    }
//    pub fn create_config(&mut self) {
//        self.config_man.create(Config::init());
//    }
//
//    pub fn load_config(&mut self, filename_config: &String) {
//        let reader = json_reader::read_file(&filename_config).unwrap();
//
//        self.set_input_width(json_reader::read_val_usize(
//            &reader,
//            "config",
//            "input_width",
//        ));
//
//        self.set_weight_rng_min(json_reader::read_val_f64(
//            &reader,
//            "config",
//            "weight_rng_min",
//        ));
//        self.set_weight_rng_max(json_reader::read_val_f64(
//            &reader,
//            "config",
//            "weight_rng_max",
//        ));
//        self.set_edge_removal_age(json_reader::read_val_usize(
//            &reader,
//            "config",
//            "edge_removal_age",
//        ));
//        self.set_neuron_creation_interval(json_reader::read_val_usize(
//            &reader,
//            "config",
//            "neuron_creation_interval",
//        ));
//        self.set_max_epochs(json_reader::read_val_usize(
//            &reader,
//            "config",
//            "max_epochs",
//        ));
//        self.set_target_error(json_reader::read_val_f64(&reader, "config", "target_error"));
//        self.set_epsilon_w(json_reader::read_val_f64(&reader, "config", "epsilon_w"));
//        self.set_epsilon_n(json_reader::read_val_f64(&reader, "config", "epsilon_n"));
//
//        //----------------------------------------
//    }
//
//    //----------------------------------------------------------
//    pub fn set_parameters(
//        &mut self,
//        input_width: usize,
//        weight_rng_min: f64,
//        weight_rng_max: f64,
//        edge_removal_age: usize,
//        neuron_creation_interval: usize,
//        max_epochs: usize,
//        target_error: f64,
//        epsilon_w: f64,
//        epsilon_n: f64,
//        alpha: f64,
//        beta: f64,
//    ) {
//        if let Some(val) = self.config_man.get_mut(0) {
//            val.input_width = input_width;
//            val.weight_rng_min = weight_rng_min;
//            val.weight_rng_max = weight_rng_max;
//            val.edge_removal_age = edge_removal_age;
//            val.neuron_creation_interval = neuron_creation_interval;
//            val.max_epochs = max_epochs;
//            val.target_error = target_error;
//            val.epsilon_w = epsilon_w;
//            val.epsilon_n = epsilon_n;
//            val.alpha = alpha;
//            val.beta = beta;
//        }
//    }
//
//    //----------------------------------------------------------
//    pub fn set_config_filename(&mut self, filename_config: String) {
//        if let Some(val) = self.config_man.get_mut(0) {
//            val.config_filename = filename_config;
//        }
//    }
//    pub fn get_config_filename(&self) -> Option<&String> {
//        self.config_man.get(0).map(|val| &val.config_filename)
//    }
//    //----------------------------------------------------------
//    pub fn set_input_set_filename(&mut self, filename: String) {
//        if let Some(val) = self.config_man.get_mut(0) {
//            val.input_set_filename = filename;
//        }
//    }
//    pub fn get_input_set_filename(&self) -> Option<&String> {
//        self.config_man.get(0).map(|val| &val.input_set_filename)
//    }
//    //----------------------------------------------------------
//    pub fn set_input_width(&mut self, val: usize) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.input_width = val;
//        }
//    }
//    pub fn read_input_width(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_usize(&reader, "config", "input_width");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.input_width = val;
//        }
//    }
//    pub fn get_input_width(&self) -> &usize {
//        self.config_man
//            .get(0)
//            .map(|val| &val.input_width)
//            .expect("input width: config not initiated")
//    }
//    //----------------------------------------------------------
//
//    //----------------------------------------------------------
//    pub fn set_weight_rng_min(&mut self, val: f64) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.weight_rng_min = val;
//        }
//    }
//    pub fn read_weight_rng_min(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_f64(&reader, "config", "weight_rng_min");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.weight_rng_min = val;
//        }
//    }
//    pub fn get_weight_rng_min(&self) -> &f64 {
//        self.config_man
//            .get(0)
//            .map(|val| &val.weight_rng_min)
//            .expect("weight rng min: config not initiated")
//    }
//
//    //----------------------------------------------------------
//    pub fn set_weight_rng_max(&mut self, val: f64) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.weight_rng_max = val;
//        }
//    }
//    pub fn read_weight_rng_max(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_f64(&reader, "config", "weight_rng_max");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.weight_rng_max = val;
//        }
//    }
//    pub fn get_weight_rng_max(&self) -> &f64 {
//        self.config_man
//            .get(0)
//            .map(|val| &val.weight_rng_max)
//            .expect("weight rng max: config not initiated")
//    }
//
//    //----------------------------------------------------------
//    pub fn set_edge_removal_age(&mut self, val: usize) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.edge_removal_age = val;
//        }
//    }
//    pub fn read_edge_removal_age(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_usize(&reader, "config", "edge_removal_age");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.edge_removal_age = val;
//        }
//    }
//    pub fn get_edge_removal_age(&self) -> Option<&usize> {
//        self.config_man.get(0).map(|val| &val.edge_removal_age)
//    }
//
//    //----------------------------------------------------------
//    pub fn set_neuron_creation_interval(&mut self, val: usize) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.neuron_creation_interval = val;
//        }
//    }
//    pub fn read_neuron_creation_interval(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_usize(&reader, "config", "neuron_creation_interval");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.neuron_creation_interval = val;
//        }
//    }
//    pub fn get_neuron_creation_interval(&self) -> &usize {
//        self.config_man
//            .get(0)
//            .map(|val| &val.neuron_creation_interval)
//            .expect("neuron creation intervall not found")
//    }
//
//    //----------------------------------------------------------
//    pub fn set_max_epochs(&mut self, val: usize) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.max_epochs = val;
//        }
//    }
//    pub fn get_max_epochs(&self) -> &usize {
//        self.config_man
//            .get(0)
//            .map(|val| &val.max_epochs)
//            .expect("max train iteration not found")
//    }
//    pub fn read_max_epochs(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_usize(&reader, "config", "max_epochs");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.max_epochs = val;
//        }
//    }
//
//    //----------------------------------------------------------
//    pub fn set_target_error(&mut self, val: f64) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.target_error = val;
//        }
//    }
//    pub fn read_target_error(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_f64(&reader, "config", "target_error");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.target_error = val;
//        }
//    }
//    pub fn get_target_error(&self) -> Option<&f64> {
//        self.config_man.get(0).map(|val| &val.target_error)
//    }
//
//    //----------------------------------------------------------
//    pub fn set_epsilon_w(&mut self, val: f64) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.epsilon_w = val;
//        }
//    }
//    pub fn read_epsilon_w(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_f64(&reader, "config", "epsilon_w");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.epsilon_w = val;
//        }
//    }
//    pub fn get_epsilon_w(&self) -> &f64 {
//        self.config_man
//            .get(0)
//            .map(|val| &val.epsilon_w)
//            .expect("epsilon_w not found")
//    }
//    pub fn set_alpha(&mut self, val: f64) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.alpha = val;
//        }
//    }
//    pub fn get_alpha(&self) -> &f64 {
//        self.config_man
//            .get(0)
//            .map(|val| &val.alpha)
//            .expect("alpha not found")
//    }
//
//    pub fn set_beta(&mut self, val: f64) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.beta = val;
//        }
//    }
//    pub fn get_beta(&self) -> &f64 {
//        self.config_man
//            .get(0)
//            .map(|val| &val.beta)
//            .expect("d not found")
//    }
//
//    //----------------------------------------------------------
//    pub fn set_epsilon_n(&mut self, val: f64) {
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.epsilon_n = val;
//        }
//    }
//    pub fn read_epsilon_n(&mut self, filename: &str) {
//        let reader = json_reader::read_file(filename).unwrap();
//        let val = json_reader::read_val_f64(&reader, "config", "epsilon_n");
//        if let Some(obj) = self.config_man.get_mut(0) {
//            obj.epsilon_n = val;
//        }
//    }
//    pub fn get_epsilon_n(&self) -> &f64 {
//        self.config_man
//            .get(0)
//            .map(|val| &val.epsilon_n)
//            .expect("epsilon_n not found")
//    }
//}
