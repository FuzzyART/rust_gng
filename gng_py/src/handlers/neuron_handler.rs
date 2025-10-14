use crate::ecs::manager;
#[derive(Debug)]
pub struct Neuron {
    pub w: Vec<f64>,

    pub error: f64,

    pub distance: f64,

    pub neuron_dependency: usize, // 1: winner neuron,
                                  // 2: neighbour neuron
                                  // 0: not connected to winner neuron
}
impl Neuron {
    pub fn init(weights: Vec<f64>) -> Self {
        Self {
            //w:Vec::new(),
            w: weights.clone(),
            error: 0.0,

            distance: 0.0,
            neuron_dependency: 0,
        }
    }
}
//--------------------------------------------------------------------------------------------------
pub struct NeuronHandler {
    pub neuron_man: manager::EntityManager<Neuron>,
}
impl NeuronHandler {
    pub fn init() -> Self {
        Self {
            neuron_man: manager::EntityManager::new(),
        }
    }
    pub fn create_neuron(&mut self, weights: Vec<f64>) -> usize {
        self.neuron_man.create(Neuron::init(weights))
    }
    pub fn remove_neuron(&mut self, key: usize) {
        self.neuron_man.remove(key);
    }

    pub fn get_weights(&self, num_neuron: usize) -> &Vec<f64> {
        self.neuron_man
            .get(num_neuron)
            .map(|val| &val.w)
            .expect("neuron not found")
    }

    pub fn set_weights(&mut self, num_neuron: usize, weights: Vec<f64>) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.w = weights;
        }
    }

    pub fn set_error(&mut self, num_neuron: usize, val: f64) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.error = val;
        }
    }
    pub fn get_error(&self, num_neuron: usize) -> &f64 {
        return self
            .neuron_man
            .get(num_neuron)
            .map(|val| &val.error)
            .expect("err not found");
    }

    pub fn get_error2(&self, num_neuron: usize) -> Option<f64> {
        self.neuron_man.get(num_neuron).map(|val| val.error)
    }

    pub fn set_distance(&mut self, num_neuron: usize, val: f64) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.distance = val;
        }
    }

    pub fn get_distance(&self, num_neuron: usize) -> &f64 {
        return self
            .neuron_man
            .get(num_neuron)
            .map(|val| &val.distance)
            .expect("distance not found");
    }

    pub fn set_neuron_dependency(&mut self, num_neuron: usize, val: usize) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.neuron_dependency = val;
        }
    }
    pub fn get_neuron_dependency(&self, num_neuron: usize) -> &usize {
        return self
            .neuron_man
            .get(num_neuron)
            .map(|val| &val.neuron_dependency)
            .expect("neuron dependency not found");
    }

    pub fn get_num_neurons(&self) -> usize {
        let temp = self.neuron_man.len();
        temp
    }

    //----------------------------------------------------------
    // Debugging helpers
    pub fn init_neurons_debug(&mut self, num_neurons: usize) {
        for a in 0..num_neurons {
            let temp: f64 = a as f64;
            let w: Vec<f64> = vec![(temp * 0.1) + 1.0, (temp * 0.1) + 2.0];
            self.create_neuron(w);
        }
    }
    pub fn set_distances_debug(&mut self, distances: Vec<f64>) {
        for a in 0..distances.len() {
            self.set_distance(a, distances[a]);
        }
    }
    pub fn set_errors_debug(&mut self, errors: Vec<f64>) {
        for a in 0..errors.len() {
            self.set_error(a, errors[a]);
        }
    }

    pub fn set_neuron_dependencies_debug(&mut self, dependencies: Vec<usize>) {
        for a in 0..dependencies.len() {
            self.set_neuron_dependency(a, dependencies[a]);
        }
    }

    pub fn print_neurons(&mut self) {
        println!("----------------------------------------");
        for (id, val) in self.neuron_man.iter() {
            println!(
                "ID: {} W: {:?} dist: {:.2}, error: {:.2},depend: {}",
                id, val.w, val.distance, val.error, val.neuron_dependency
            );
        }
        println!("----------------------------------------");
    }
    pub fn get_weight_vec(&self, width: &usize) -> Vec<f64> {
        let len = self.neuron_man.len();
        let mut res_vec: Vec<f64> = vec![0.0; len * (*width)];

        for (id, line) in self.neuron_man.iter() {
            for a in 0..(*width) {
                res_vec[(*id * (*width)) + a] = line.w[a];
            }
        }
        res_vec
    }
    pub fn get_all_errors(&self) -> Vec<f64> {
        self.neuron_man
            .iter()
            .map(|(_, neuron)| neuron.error)
            .collect()
    }
    pub fn get_all_neuron_ids(&self) -> Vec<usize> {
        self.neuron_man.keys().copied().collect()
    }
    pub fn get_all_neuron_ids_sorted(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.neuron_man.keys().copied().collect();
        ids.sort();
        ids
    }

    pub fn get_mut(&mut self, key: usize) -> Option<&mut Neuron> {
        self.neuron_man.get_mut(key)
    }

    pub fn get_keys(&self) -> impl Iterator<Item = &usize> {
        self.neuron_man.keys()
    }
}
