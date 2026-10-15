use crate::ecs::manager;

/// A neuron in the neural network.
///
/// This struct represents a single neuron with its weights, error value,
/// distance, and dependency information used in the SOM (Self-Organizing Map) algorithm.
#[derive(Debug)]
pub struct Neuron {
    /// The weights of the neuron.
    pub w: Vec<f64>,

    /// The error associated with this neuron.
    pub error: f64,

    /// The distance from the winning neuron.
    pub distance: f64,

    /// The dependency of this neuron to the winner neuron:
    /// - `1`: Winner neuron
    /// - `2`: Neighbour neuron
    /// - `0`: Not connected to winner neuron
    pub neuron_dependency: usize,
}

impl Neuron {
    /// Initializes a new neuron with the given weights.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of initial weights for the neuron.
    ///
    /// # Returns
    ///
    /// A new `Neuron` instance initialized with the provided weights and default values
    /// for error, distance, and neuron dependency.
    pub fn init(weights: Vec<f64>) -> Self {
        Self {
            w: weights.clone(),
            error: 0.0,
            distance: 0.0,
            neuron_dependency: 0,
        }
    }
}

/// A handler for managing neurons in the neural network.
///
/// This struct provides methods to create, modify, and retrieve information
/// about neurons stored in an entity manager.
pub struct NeuronHandler {
    /// The entity manager that stores the neurons.
    pub neuron_man: manager::EntityManager<Neuron>,
}

impl NeuronHandler {
    /// Creates a new `NeuronHandler` with an empty entity manager.
    ///
    /// # Returns
    ///
    /// A new `NeuronHandler` instance.
    pub fn init() -> Self {
        Self {
            neuron_man: manager::EntityManager::new(),
        }
    }

    /// Creates a new neuron and adds it to the manager.
    ///
    /// # Arguments
    ///
    /// * `weights` - The initial weights for the new neuron.
    ///
    /// # Returns
    ///
    /// The unique identifier (key) assigned to the newly created neuron.
    pub fn create_neuron(&mut self, weights: Vec<f64>) -> usize {
        self.neuron_man.create(Neuron::init(weights))
    }

    /// Removes a neuron from the manager by its key.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique identifier of the neuron to remove.
    pub fn remove_neuron(&mut self, key: usize) {
        self.neuron_man.remove(key);
    }

    /// Gets immutable references to the weights of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose weights are requested.
    ///
    /// # Returns
    ///
    /// A reference to the weights vector of the specified neuron.
    ///
    /// # Panics
    ///
    /// Panics if the neuron with the given key does not exist.
    pub fn get_weights(&self, num_neuron: usize) -> &Vec<f64> {
        self.neuron_man
            .get(num_neuron)
            .map(|val| &val.w)
            .expect("neuron not found")
    }

    /// Sets the weights of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose weights are to be set.
    /// * `weights` - The new weights for the neuron.
    pub fn set_weights(&mut self, num_neuron: usize, weights: Vec<f64>) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.w = weights;
        }
    }

    /// Sets the error value of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose error is to be set.
    /// * `val` - The new error value for the neuron.
    pub fn set_error(&mut self, num_neuron: usize, val: f64) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.error = val;
        }
    }

    /// Gets immutable reference to the error of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose error is requested.
    ///
    /// # Returns
    ///
    /// A reference to the error value of the specified neuron.
    ///
    /// # Panics
    ///
    /// Panics if the neuron with the given key does not exist.
    pub fn get_error(&self, num_neuron: usize) -> &f64 {
        return self
            .neuron_man
            .get(num_neuron)
            .map(|val| &val.error)
            .expect("err not found");
    }

    /// Gets the error value of a neuron as an `Option`.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose error is requested.
    ///
    /// # Returns
    ///
    /// An `Option<f64>` containing the error value if the neuron exists, or `None` otherwise.
    pub fn get_error2(&self, num_neuron: usize) -> Option<f64> {
        self.neuron_man.get(num_neuron).map(|val| val.error)
    }

    /// Sets the distance of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose distance is to be set.
    /// * `val` - The new distance value for the neuron.
    pub fn set_distance(&mut self, num_neuron: usize, val: f64) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.distance = val;
        }
    }

    /// Gets immutable reference to the distance of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose distance is requested.
    ///
    /// # Returns
    ///
    /// A reference to the distance value of the specified neuron.
    ///
    /// # Panics
    ///
    /// Panics if the neuron with the given key does not exist.
    pub fn get_distance(&self, num_neuron: usize) -> &f64 {
        return self
            .neuron_man
            .get(num_neuron)
            .map(|val| &val.distance)
            .expect("distance not found");
    }

    /// Sets the neuron dependency of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose dependency is to be set.
    /// * `val` - The new dependency value for the neuron (1 = winner, 2 = neighbor, 0 = not connected).
    pub fn set_neuron_dependency(&mut self, num_neuron: usize, val: usize) {
        if let Some(obj) = self.neuron_man.get_mut(num_neuron) {
            obj.neuron_dependency = val;
        }
    }

    /// Gets immutable reference to the neuron dependency of a neuron.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The identifier of the neuron whose dependency is requested.
    ///
    /// # Returns
    ///
    /// A reference to the neuron dependency value of the specified neuron.
    ///
    /// # Panics
    ///
    /// Panics if the neuron with the given key does not exist.
    pub fn get_neuron_dependency(&self, num_neuron: usize) -> &usize {
        return self
            .neuron_man
            .get(num_neuron)
            .map(|val| &val.neuron_dependency)
            .expect("neuron dependency not found");
    }

    /// Gets the total number of neurons managed.
    ///
    /// # Returns
    ///
    /// The count of neurons in the manager.
    pub fn get_num_neurons(&self) -> usize {
        let temp = self.neuron_man.len();
        temp
    }

    /// Gets the total number of neurons managed (alias for `get_num_neurons`).
    ///
    /// # Returns
    ///
    /// The count of neurons in the manager.
    pub fn len(&self) -> usize {
        self.neuron_man.len()
    }

    /// Initializes neurons for debugging purposes.
    ///
    /// Creates a specified number of neurons with initialized weights based on their index.
    ///
    /// # Arguments
    ///
    /// * `num_neurons` - The number of neurons to create.
    pub fn init_neurons_debug(&mut self, num_neurons: usize) {
        for a in 0..num_neurons {
            let temp: f64 = a as f64;
            let w: Vec<f64> = vec![(temp * 0.1) + 1.0, (temp * 0.1) + 2.0];
            self.create_neuron(w);
        }
    }

    /// Sets distances for neurons for debugging purposes.
    ///
    /// # Arguments
    ///
    /// * `distances` - A vector of distance values to assign to neurons.
    pub fn set_distances_debug(&mut self, distances: Vec<f64>) {
        for a in 0..distances.len() {
            self.set_distance(a, distances[a]);
        }
    }

    /// Sets errors for neurons for debugging purposes.
    ///
    /// # Arguments
    ///
    /// * `errors` - A vector of error values to assign to neurons.
    pub fn set_errors_debug(&mut self, errors: Vec<f64>) {
        for a in 0..errors.len() {
            self.set_error(a, errors[a]);
        }
    }

    /// Sets neuron dependencies for debugging purposes.
    ///
    /// # Arguments
    ///
    /// * `dependencies` - A vector of dependency values to assign to neurons.
    pub fn set_neuron_dependencies_debug(&mut self, dependencies: Vec<usize>) {
        for a in 0..dependencies.len() {
            self.set_neuron_dependency(a, dependencies[a]);
        }
    }

    /// Prints all neurons with their properties for debugging purposes.
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

    /// Gets all weights of neurons arranged in a flattened vector.
    ///
    /// # Arguments
    ///
    /// * `width` - The width (number of elements per neuron) to use for the result vector.
    ///
    /// # Returns
    ///
    /// A flattened vector containing all neuron weights.
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

    /// Gets all error values of neurons.
    ///
    /// # Returns
    ///
    /// A vector containing the error value of each neuron in order.
    pub fn get_all_errors(&self) -> Vec<f64> {
        self.neuron_man
            .iter()
            .map(|(_, neuron)| neuron.error)
            .collect()
    }

    /// Gets all neuron identifiers.
    ///
    /// # Returns
    ///
    /// A vector containing the identifiers of all neurons.
    pub fn get_all_neuron_ids(&self) -> Vec<usize> {
        self.neuron_man.keys().copied().collect()
    }

    /// Gets all neuron identifiers sorted in ascending order.
    ///
    /// # Returns
    ///
    /// A vector containing the identifiers of all neurons, sorted in ascending order.
    pub fn get_all_neuron_ids_sorted(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.neuron_man.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Gets a mutable reference to a neuron by its key.
    ///
    /// # Arguments
    ///
    /// * `key` - The identifier of the neuron to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option<&mut Neuron>` containing a mutable reference to the neuron if found,
    /// or `None` if no neuron exists with that key.
    pub fn get_mut(&mut self, key: usize) -> Option<&mut Neuron> {
        self.neuron_man.get_mut(key)
    }

    /// Gets an iterator over all neuron keys.
    ///
    /// # Returns
    ///
    /// An iterator over references to the identifiers of all neurons.
    pub fn get_keys(&self) -> impl Iterator<Item = &usize> {
        self.neuron_man.keys()
    }
}
