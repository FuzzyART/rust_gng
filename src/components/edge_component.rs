//! A module for managing edge components in the ECS.
//!
//! This module provides functionality for creating, managing, and querying edges
//! within an entity-component system. Each edge represents a connection between
//! two entities (neurons) with associated metadata such as age.

use crate::ecs::manager;

//==================================================================================================
/// Represents an edge connecting two neurons.
///
/// An `Edge` stores the IDs of the start and end neurons, along with the age of the edge.
/// The age can be used to track how long an edge has existed or for decay mechanisms.
#[derive(Debug)]
pub struct Edge {
    /// The ID of the starting neuron.
    pub start: usize,
    /// The ID of the ending neuron.
    pub end: usize,
    /// The age of the edge, typically used to track lifetime or decay.
    pub age: usize,
}

impl Edge {
    /// Creates a new `Edge` with the specified start, end, and age.
    ///
    /// # Arguments
    ///
    /// * `from` - The ID of the starting neuron.
    /// * `to` - The ID of the ending neuron.
    /// * `age` - The initial age of the edge.
    ///
    /// # Returns
    ///
    /// Returns a new `Edge` instance with the provided parameters.
    pub fn init(from: usize, to: usize, age: usize) -> Self {
        Self {
            start: from,
            end: to,
            age: age,
        }
    }
}

//==================================================================================================
/// A handler for managing edges within an ECS.
///
/// The `EdgeHandler` uses an [`EntityManager`] to store and manage edge entities.
/// It provides methods for creating, removing, and querying edges, as well as
/// accessing their properties.
///
/// # Examples
///
/// ```
/// use gng::components::edge_component::EdgeHandler;
///
/// let mut handler = EdgeHandler::init();
/// let edge_id = handler.create_edge(0, 1, 0);
/// assert_eq!(handler.get_num_edges(), 1);
/// ```
pub struct EdgeHandler {
    /// The entity manager responsible for storing edge entities.
    pub edge_man: manager::EntityManager<Edge>,
}

impl EdgeHandler {
    /// Creates a new `EdgeHandler` with an empty entity manager.
    ///
    /// # Returns
    ///
    /// Returns a new `EdgeHandler` instance.
    pub fn init() -> Self {
        Self {
            edge_man: manager::EntityManager::new(),
        }
    }

    /// Creates a new edge and returns its unique ID.
    ///
    /// This method creates an edge connecting two neurons with the specified age.
    /// The returned ID can be used to reference this edge in subsequent operations.
    ///
    /// # Arguments
    ///
    /// * `from` - The ID of the starting neuron.
    /// * `to` - The ID of the ending neuron.
    /// * `age` - The initial age of the edge.
    ///
    /// # Returns
    ///
    /// Returns the unique ID assigned to the newly created edge.
    pub fn create_edge(&mut self, from: usize, to: usize, age: usize) {
        self.edge_man.create(Edge::init(from, to, age));
    }

    /// Removes an edge with the given ID.
    ///
    /// If an edge with the specified ID exists, it is removed and its ID becomes
    /// available for reuse. If no such edge exists, this operation has no effect.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique ID of the edge to remove.
    pub fn remove_edge(&mut self, key: usize) {
        self.edge_man.remove(key);
    }

    /// Prints all edges in a human-readable format.
    ///
    /// This method iterates through all stored edges and prints their IDs,
    /// start neuron IDs, and end neuron IDs.
    pub fn print_edges(&mut self) {
        for (id, val) in self.edge_man.iter() {
            println!("ID: {} From: {:?} To: {:?}", id, val.start, val.end);
        }
    }

    /// Gets all edge IDs connected to a specific neuron.
    ///
    /// This method returns a vector of edge IDs where the specified neuron
    /// is either the start or end of the edge.
    ///
    /// # Arguments
    ///
    /// * `neuron_num` - The ID of the neuron to search for connections.
    ///
    /// # Returns
    ///
    /// Returns a vector containing the IDs of all edges connected to the specified neuron.
    pub fn get_connected_edges(&self, neuron_num: usize) -> Vec<usize> {
        let mut res: Vec<usize> = Vec::new();
        let keys = self.get_keys();
        for a in keys {
            let start_edge: usize = *self
                .edge_man
                .get(*a)
                .map(|val| &val.start)
                .expect("edge not found 1");
            let end_edge: usize = *self
                .edge_man
                .get(*a)
                .map(|val| &val.end)
                .expect("edge not found 2");
            if (start_edge == neuron_num) || (end_edge == neuron_num) {
                res.push(*a);
            }
        }
        res
    }

    /// Gets the start neuron ID of an edge.
    ///
    /// # Arguments
    ///
    /// * `num_neuron` - The ID of the edge to query.
    ///
    /// # Returns
    ///
    /// Returns a reference to the start neuron ID of the specified edge.
    ///
    /// # Panics
    ///
    /// Panics if no edge exists with the given ID.
    pub fn get_edge_start(&self, num_neuron: usize) -> &usize {
        let edge_start = self
            .edge_man
            .get(num_neuron)
            .map(|val| &val.start)
            .expect("edge start not found 1");
        edge_start
    }

    /// Gets the end neuron ID of an edge.
    ///
    /// # Arguments
    ///
    /// * `num` - The ID of the edge to query.
    ///
    /// # Returns
    ///
    /// Returns a reference to the end neuron ID of the specified edge.
    ///
    /// # Panics
    ///
    /// Panics if no edge exists with the given ID.
    pub fn get_edge_end(&self, num: usize) -> &usize {
        let edge_start = self
            .edge_man
            .get(num)
            .map(|val| &val.end)
            .expect("edge start not found 2");
        edge_start
    }

    /// Gets the age of an edge.
    ///
    /// # Arguments
    ///
    /// * `num` - The ID of the edge to query.
    ///
    /// # Returns
    ///
    /// Returns a reference to the age of the specified edge.
    ///
    /// # Panics
    ///
    /// Panics if no edge exists with the given ID.
    pub fn get_edge_age(&self, num: usize) -> &usize {
        let edge_age = self
            .edge_man
            .get(num)
            .map(|val| &val.age)
            .expect("edge age not found 3");
        edge_age
    }

    /// Sets the age of an edge.
    ///
    /// # Arguments
    ///
    /// * `num` - The ID of the edge to modify.
    /// * `val` - The new age value for the edge.
    pub fn set_edge_age(&mut self, num: usize, val: usize) {
        if let Some(obj) = self.edge_man.get_mut(num) {
            obj.age = val;
        }
    }

    /// Increments the age of an edge by one.
    ///
    /// # Arguments
    ///
    /// * `num` - The ID of the edge to increment.
    pub fn increase_edge_age(&mut self, num: usize) {
        if let Some(obj) = self.edge_man.get_mut(num) {
            obj.age = obj.age + 1;
        }
    }

    /// Gets the total number of edges stored.
    ///
    /// # Returns
    ///
    /// Returns the count of all edges managed by this handler.
    pub fn get_num_edges(&self) -> usize {
        self.edge_man.len()
    }

    /// Gets the total number of edges stored (alias for `get_num_edges`).
    ///
    /// # Returns
    ///
    /// Returns the count of all edges managed by this handler.
    pub fn len(&self) -> usize {
        self.edge_man.len()
    }

    //==================================================================================================
    // Debug functions

    /// Gets a vector of start neuron IDs for all edges.
    ///
    /// This method returns a vector where each index corresponds to an edge ID,
    /// and the value at that index is the start neuron ID of the edge.
    ///
    /// # Returns
    ///
    /// Returns a vector containing the start neuron IDs of all edges.
    pub fn get_edge_start_vec(&self) -> Vec<usize> {
        let len = self.edge_man.len();
        let mut res_vec: Vec<usize> = vec![0; len];

        for (id, line) in self.edge_man.iter() {
            res_vec[*id] = line.start;
        }
        res_vec
    }

    /// Gets a vector of end neuron IDs for all edges.
    ///
    /// This method returns a vector where each index corresponds to an edge ID,
    /// and the value at that index is the end neuron ID of the edge.
    ///
    /// # Returns
    ///
    /// Returns a vector containing the end neuron IDs of all edges.
    pub fn get_edge_end_vec(&self) -> Vec<usize> {
        let len = self.edge_man.len();
        let mut res_vec: Vec<usize> = vec![0; len];

        for (id, line) in self.edge_man.iter() {
            res_vec[*id] = line.end;
        }
        res_vec
    }

    /// Gets a vector of ages for all edges.
    ///
    /// This method returns a vector where each index corresponds to an edge ID,
    /// and the value at that index is the age of the edge.
    ///
    /// # Returns
    ///
    /// Returns a vector containing the ages of all edges.
    pub fn get_edge_age_vec(&self) -> Vec<usize> {
        let len = self.edge_man.len();
        let mut res_vec: Vec<usize> = vec![0; len];

        for (id, line) in self.edge_man.iter() {
            res_vec[*id] = line.age;
        }
        res_vec
    }

    /// Gets all edge IDs.
    ///
    /// # Returns
    ///
    /// Returns a vector containing the IDs of all edges.
    pub fn get_all_edge_ids(&self) -> Vec<usize> {
        self.edge_man.keys().copied().collect()
    }

    /// Gets an iterator over all edge keys (IDs).
    ///
    /// # Returns
    ///
    /// Returns an iterator over references to all edge IDs.
    pub fn get_keys(&self) -> impl Iterator<Item = &usize> {
        self.edge_man.keys()
    }
}
