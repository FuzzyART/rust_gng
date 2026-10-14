use crate::handlers::system_handler::Phase;
use std::collections::HashSet;

use crate::aux::{
    csv_reader, json_reader, json_writer::write_json_to_file, json_writer::write_value_to_block,
    rng_manager::RngManager,
};

use rand::seq::SliceRandom;
use serde_json::{json, Value};

use crate::gas::gng_system::Gng_System;
//--------------------------------------------------------------------------------------------------
pub fn init_step(params: &mut Gng_System) {
    //   let mut curr_phase = Phase::StartNewEpoch;
    init_training(params);
    shuffle_dataset(params);
    params.system_handler.train_initiated = true;
    params.system_handler.curr_phase = Phase::StartNewEpoch;
}

//--------------------------------------------------------------------------------------------------
pub fn fit_step(params: &mut Gng_System) {
    shuffle_dataset(params);
    let curr_epoch = params.system_handler.curr_epoch;
    params.system_handler.curr_epoch = curr_epoch + 1;

    params.system_handler.curr_phase = Phase::NormalIteration;
    while params.system_handler.curr_phase != Phase::StartNewEpoch {
        select_sample(params);
        calc_neuron_distances(params);
        calc_nearest_neurons(params);
        calc_neuron_dependencies(params);
        increase_edge_age(params);
        add_error_to_winner_neuron(params);

        update_weights(params);

        create_edge(params);
        delete_old_edges(params);
        remove_unconnected_neurons(params);

        if params.system_handler.curr_iteration % params.config_handler.neuron_creation_interval
            == 0
        {
            create_neuron(params);
        }

        decrease_error_global(params);

        params.system_handler.curr_iteration += 1;

        update_phase(params);
    }
}
//--------------------------------------------------------------------------------------------------
pub fn fit(params: &mut Gng_System) {
    // init_run
    params.system_handler.curr_phase = Phase::StartNewEpoch;
    init_training(params);
    shuffle_dataset(params);
    params.system_handler.train_initiated = true;
    // init_run
    //========================================
    while params.system_handler.train_completed == false {
        if params.system_handler.curr_phase == Phase::StartNewEpoch {
            shuffle_dataset(params);
            let curr_epoch = params.system_handler.curr_epoch;
            params.system_handler.curr_epoch = curr_epoch + 1;
        }

        if params.system_handler.curr_phase == Phase::NormalIteration {
            select_sample(params);
            calc_neuron_distances(params);
            calc_nearest_neurons(params);
            calc_neuron_dependencies(params);
            increase_edge_age(params);
            add_error_to_winner_neuron(params);
            update_weights(params);
            create_edge(params);
            delete_old_edges(params);
            remove_unconnected_neurons(params);

            if params.system_handler.curr_iteration % params.config_handler.neuron_creation_interval
                == 0
            {
                create_neuron(params);
            }

            decrease_error_global(params);
        }

        params.system_handler.curr_iteration += 1;
        update_phase(params);
        if params.system_handler.curr_epoch >= params.config_handler.max_epochs {
            params.system_handler.train_completed = true;
        }
    }
}

//--------------------------------------------------------------------------------------------------
pub fn update_phase(params: &mut Gng_System) {
    if params.system_handler.train_initiated == true {
        params.system_handler.curr_phase = Phase::NormalIteration;
    };
    if params.system_handler.last_sample_reached == true {
        params.system_handler.last_sample_reached = false;
        params.system_handler.curr_phase = Phase::StartNewEpoch;
    }
}

pub fn start_new_epoch(params: &mut Gng_System) {
    shuffle_dataset(params);
    let curr_epoch = params.system_handler.curr_epoch;
    params.system_handler.curr_epoch = curr_epoch + 1;
}

//--------------------------------------------------------------------------------------------------

pub fn load_model(params: &mut Gng_System, filename_model: String) {
    let reader_input = json_reader::read_file(&filename_model).unwrap();

    let weights = json_reader::read_array_f64(&reader_input, "gng_model", "weights");
    let edge_start = json_reader::read_array_usize(&reader_input, "gng_model", "edge_start");
    let edge_end = json_reader::read_array_usize(&reader_input, "gng_model", "edge_end");
    let edge_age = json_reader::read_array_usize(&reader_input, "gng_model", "edge_age");
    let input_width = json_reader::read_val_usize(&reader_input, "gng_model", "input_width");

    params.config_handler.input_width = input_width;
    let num_neurons = weights.len() / input_width;
    let num_edges = edge_start.len();

    let mut neuron_ids: Vec<usize> = Vec::new();
    for n in 0..num_neurons {
        let mut w_curr: Vec<f64> = Vec::new();
        for w in 0..input_width {
            w_curr.push(weights[(n * input_width) + w]);
        }
        let id = params.neuron_handler.create_neuron(w_curr);
        neuron_ids.push(id);
    }

    for e in 0..num_edges {
        params
            .edge_handler
            .create_edge(edge_start[e], edge_end[e], edge_age[e]);
    }
}
//--------------------------------------------------------------------------------------------------
pub fn configure_model(filename_input: String, gng_params: &mut Gng_System) {
    let reader_input = json_reader::read_file(&filename_input).unwrap();
    gng_params.config_handler.input_width =
        json_reader::read_val_usize(&reader_input, "config", "input_width");
    gng_params.config_handler.weight_rng_min =
        json_reader::read_val_f64(&reader_input, "config", "weight_rng_min");
    gng_params.config_handler.weight_rng_max =
        json_reader::read_val_f64(&reader_input, "config", "weight_rng_max");
}

pub fn set_parameters(
    gng_params: &mut Gng_System,
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
    gng_params.config_handler.set_parameters(
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
}
pub fn set_input_width(gng_params: &mut Gng_System, input_width: usize) {
    gng_params.config_handler.input_width = input_width;
}
//--------------------------------------------------------------------------------------------------

pub fn init_model(params: &mut Gng_System) {
    let width = params.config_handler.input_width;
    let rng_min = params.config_handler.weight_rng_min;
    let rng_max = params.config_handler.weight_rng_max;
    let mut weights_n1: Vec<f64> = Vec::new();
    let mut weights_n2: Vec<f64> = Vec::new();

    for _a in 0..width {
        weights_n1.push(params.rng_manager.get_f64(rng_min, rng_max));
        weights_n2.push(params.rng_manager.get_f64(rng_min, rng_max));
    }
    let n1: usize = params.neuron_handler.create_neuron(weights_n1);
    let n2 = params.neuron_handler.create_neuron(weights_n2);
    params.edge_handler.create_edge(n1, n2, 0);
}

//--------------------------------------------------------------------------------------------------
pub fn calc_neuron_distances(params: &mut Gng_System) {
    // input: config: input_width
    //                weight
    //        system: curr_sample_pos
    //        sample: sample
    // output: neuron: distance
    //----------------------------------------
    // input
    let input_width = params.config_handler.input_width;

    let sample_pos = params.system_handler.curr_sample_pos;
    // output
    //----------------------------------------
    let input = params.sample_handler.get_sample(sample_pos);

    let mut distance_temp: Vec<(usize, f64)> = Vec::new();
    let keys = params.neuron_handler.get_keys();
    for n in keys {
        let mut dist = 0.0;
        let w = params.neuron_handler.get_weights(*n); //[n * input_width + i];
        for i in 0..input_width {
            let curr_w = w[i];
            let curr_input = input[i];
            dist += (curr_w - curr_input) * (curr_w - curr_input);
        }
        let dist_res = f64::sqrt(dist);
        distance_temp.push((*n, dist_res));
    }

    for (n, val) in distance_temp {
        params.neuron_handler.set_distance(n, val);
    }
}

//--------------------------------------------------------------------------------------------------
/// Calculates the 2 neurons closest to the input
pub fn calc_nearest_neurons(params: &mut Gng_System) {
    let keys = params.neuron_handler.get_keys();

    let mut first_min = f64::INFINITY;
    let mut first_key = None;
    let mut second_min = f64::INFINITY;
    let mut second_key = None;

    for &key in keys {
        let distance = *params.neuron_handler.get_distance(key);
        if distance < first_min {
            second_min = first_min;
            second_key = first_key;
            first_min = distance;
            first_key = Some(key);
        } else if distance < second_min {
            second_min = distance;
            second_key = Some(key);
        }
    }

    if let Some(winner) = first_key {
        params.system_handler.winner_neuron = winner;
        if let Some(second) = second_key {
            params.system_handler.second_neuron = second;
        } else {
            // Handle case where there's only one neuron
            params.system_handler.second_neuron = winner; // Or some default
        }
    } else {
        // Handle case where there are no neurons
    }
}

//--------------------------------------------------------------------------------------------------
//Calculates and populates the neuron dependency array
//
// the neuron dependency array identifies the winner neuron and its neighboring neurons within the network. This information is used
//to determine the influence scope of the winning neuron during subsequent processing steps.
//
//## Function Signature
//```rust
//pub fn calc_neuron_dependencies(params: &mut Gng_System)
//
//## Parameters
//
//### `params: &mut Gng_System`
//A mutable reference to a `Gng_System` struct containing the system state and all necessary handlers. This parameter is modified in place to update neuron dependency flags.
//
//## Function Behavior
//
//### Process
//1. **Neuron Identification**: Retrieves the total number of neurons and identifies the current winner neuron
//2. **Edge Analysis**: Obtains all edges connected to the winner neuron and determines neighboring neurons through edge connections
//3. **Dependency Assignment**: Sets dependency flags for each neuron in the network:
//   - **Flag 0**: Irrelevant neuron (not part of winner's influence scope)
//   - **Flag 1**: Winner neuron (primary neuron of interest)
//   - **Flag 2**: Neighbor neuron (directly connected to winner neuron)
//
//### Algorithm Steps
//1. Collect all edges connected to the winner neuron
//2. For each edge, identify the neighboring neuron (either start or end point)
//3. Initialize all neurons with dependency flag 0
//4. Set the winner neuron's dependency flag to 1
//5. Set all neighbor neurons' dependency flags to 2
//
//## Requirements
//- `model_params.winner_edges` must be properly initialized and populated
//- The system must have a valid winner neuron identified in `system_handler.winner_neuron`
//- Edge handler must contain valid edge connections for the winner neuron
//
//## Notes
//- This function assumes that `winner_edges` contains valid edge references
//- The dependency array is reset and rebuilt entirely on each call
//- Neighbors are determined based on direct edge connections from the winner neuron
//- All neurons in the system are processed, regardless of their current dependency state

pub fn calc_neuron_dependencies(params: &mut Gng_System) {
    // params
    let num_neurons = params.neuron_handler.get_num_neurons();

    // input
    let winner_neuron = params.system_handler.winner_neuron;
    let winner_edges: Vec<usize> = params.edge_handler.get_connected_edges(winner_neuron);

    //----------------------------------------

    let mut neighbor_neur: Vec<usize> = Vec::new();
    for a in 0..winner_edges.len() {
        let curr_edge = winner_edges[a];
        let curr_edge_start: usize = *params.edge_handler.get_edge_start(curr_edge);
        if curr_edge_start != winner_neuron {
            neighbor_neur.push(curr_edge_start);
        } else {
            let curr_edge_end: usize = *params.edge_handler.get_edge_end(curr_edge);
            neighbor_neur.push(curr_edge_end);
        }
    }
    // fill neuron dependency array
    let keys: Vec<usize> = params.neuron_handler.get_keys().cloned().collect();
    for a in &keys {
        params.neuron_handler.set_neuron_dependency(*a, 0);
    }
    params
        .neuron_handler
        .set_neuron_dependency(winner_neuron, 1);
    for a in 0..neighbor_neur.len() {
        params
            .neuron_handler
            .set_neuron_dependency(neighbor_neur[a], 2);
    }
}

//--------------------------------------------------------------------------------------------------
pub fn update_weights(params: &mut Gng_System) {
    // params
    let input_width = params.config_handler.input_width;
    let epsilon_w: f64 = params.config_handler.epsilon_w;
    let epsilon_n: f64 = params.config_handler.epsilon_n;

    // input
    let sample_pos = params.system_handler.curr_sample_pos;
    //----------------------------------------
    let mut w_temp: Vec<(usize, Vec<f64>)> = Vec::new();

    let keys = params.neuron_handler.get_keys();
    for n in keys {
        let dependency = *params.neuron_handler.get_neuron_dependency(*n);
        if dependency == 0 {
        } else if dependency == 1 {
            let in_vec_temp = params.sample_handler.get_sample(sample_pos);
            let weight_vec_temp = params.neuron_handler.get_weights(*n);
            let mut weight_res: Vec<f64> = vec![0.0; input_width];
            for i in 0..input_width {
                let in_temp = in_vec_temp[i];
                let w_temp = weight_vec_temp[i];
                let w_delta: f64 = in_temp - w_temp;
                weight_res[i] = w_temp + (w_delta * epsilon_w);
            }
            w_temp.push((*n, weight_res));
        } else if dependency == 2 {
            let in_vec_temp = params.sample_handler.get_sample(sample_pos);
            let weight_vec_temp = params.neuron_handler.get_weights(*n);
            let mut weight_res: Vec<f64> = vec![0.0; input_width];
            for i in 0..input_width {
                let in_temp = in_vec_temp[i];
                let w_temp = weight_vec_temp[i];
                let w_delta: f64 = in_temp - w_temp;
                weight_res[i] = w_temp + (w_delta * epsilon_n);
            }
            w_temp.push((*n, weight_res));
        } else {
        }
    }
    for (id, curr_weight) in &w_temp {
        params
            .neuron_handler
            .set_weights(*id, (*curr_weight.clone()).to_vec());
    }
}

//--------------------------------------------------------------------------------------------------
pub fn get_winner_edges(params: &mut Gng_System) -> Vec<usize> {
    let winner_neuron = params.system_handler.winner_neuron;

    return params.edge_handler.get_connected_edges(winner_neuron);
}
//--------------------------------------------------------------------------------------------------

pub fn calc_neighbor_neuron_vec_max_err(params: &mut Gng_System) {
    // input
    //  edge: edge start
    //  edge: edge end
    //  system: winner_neuron
    // output
    //  system: neighbor_neurons

    let mut res_vec: Vec<usize> = Vec::new();

    let target_neuron = params.system_handler.neuron_max_err;

    let keys = params.edge_handler.get_keys();
    for a in keys {
        let &edge_start = params.edge_handler.get_edge_start(*a);
        let &edge_end = params.edge_handler.get_edge_end(*a);
        if edge_start == target_neuron {
            res_vec.push(edge_end);
        }
        if edge_end == target_neuron {
            res_vec.push(edge_start);
        }
    }
    // output
    params.system_handler.neighbor_neuron_vec_max_err = res_vec;
}
//--------------------------------------------------------------------------------------------------
/// Selects the next sample from the sample order and updates the system state.
///
/// This function retrieves the current sample order position and the sample
/// order, then assigns the sample at the current position to `curr_sample_pos`.
/// It increments `sample_order_position` to prepare for the next sample.
///
/// If the current position is the last position in the sample order,
/// `last_sample_reached` is set to `true`.
///
/// # Arguments
///
/// * `params` - A mutable reference to the GNG system containing the sample
///   order, current sample position, and related state.
///
/// # Panics
///
/// Panics if `sample_order_position` is out of bounds for `sample_order`.
pub fn select_sample(params: &mut Gng_System) {
    let sample_order_position = params.system_handler.sample_order_position.clone();
    let sample_order = params.system_handler.sample_order.clone();
    let len: usize = sample_order.len();

    params.system_handler.curr_sample_pos = sample_order[sample_order_position];
    params.system_handler.sample_order_position = sample_order_position + 1;

    if sample_order_position >= len - 1 {
        params.system_handler.last_sample_reached = true;
    }
}
//--------------------------------------------------------------------------------------------------
//Randomizes the order of samples in the dataset.
//
// This function prepares the sample order for sequential sampling by creating a
//shuffled permutation of all available sample IDs.
//
//## Function Signature
//```rust
//pub fn shuffle_dataset(params: &mut Gng_System)
//```
//
//## Parameters
//
//### `params: &mut Gng_System`
//A mutable reference to a `Gng_System` struct containing the system state and handlers. This parameter is modified in place to update the sample order and reset the sample position counter.
//
//## Function Behavior
//
//### Process
//1. **Collect Sample Keys**: Retrieves all sample IDs from the sample handler and collects them into a vector
//2. **Normalize Order**: Sorts the sample IDs in ascending order to ensure consistent iteration
//3. **Apply Randomization**: Uses the system's seeded random number generator to shuffle the sorted sample IDs
//4. **Update System State**: Stores the shuffled order in the system handler and resets the sample position counter
//
//### State Updates
//- `params.system_handler.sample_order`: Updated to contain the randomly shuffled sample IDs
//- `params.system_handler.sample_order_position`: Reset to 0, starting the sampling sequence from the beginning
//
//## Notes
//- The function uses a seeded RNG for reproducible shuffling results
//- Sample IDs are first sorted then shuffled to ensure deterministic behavior while maintaining randomness
//- This function should be called before starting a new sampling iteration to randomize the order of samples

pub fn shuffle_dataset(params: &mut Gng_System) {
    let mut res: Vec<usize> = params.sample_handler.get_keys().copied().collect();

    // Normalize the hash-table iteration order.
    res.sort_unstable();

    // Apply the seeded RNG.
    let rng = params.rng_manager.get_rng();
    res.shuffle(rng);

    params.system_handler.sample_order = res;
    params.system_handler.sample_order_position = 0;
}

//--------------------------------------------------------------------------------------------------

pub fn decrease_error_global(params: &mut Gng_System) {
    let keys = params.neuron_handler.get_keys();
    let d = params.config_handler.beta;

    let mut res: Vec<(usize, f64)> = Vec::new();

    for a in keys {
        //let mut err = 0.0;
        if let Some(val) = params.neuron_handler.get_error2(*a) {
            res.push((*a, val * d));
        } else {
        }
    }

    for (a, err) in res {
        let new_err = err;

        params.neuron_handler.set_error(a, new_err);
    }
}
//--------------------------------------------------------------------------------------------------
pub fn increase_edge_age(params: &mut Gng_System) {
    let winner_neuron = params.system_handler.winner_neuron;

    let keys = params.edge_handler.get_keys();
    let mut marked_edges: Vec<usize> = Vec::new();
    for a in keys {
        let edge_start = params.edge_handler.get_edge_start(*a);
        let edge_end = params.edge_handler.get_edge_end(*a);
        let edge_age = params.edge_handler.get_edge_age(*a);
        if (*edge_start == winner_neuron) || (*edge_end == winner_neuron) {
            marked_edges.push(*a);
        }
    }
    for i in marked_edges {
        let age = params.edge_handler.get_edge_age(i);
        params.edge_handler.set_edge_age(i, age + 1);
    }
}
//--------------------------------------------------------------------------------------------------
pub fn add_error_to_winner_neuron(params: &mut Gng_System) {
    let sample_pos = params.system_handler.curr_sample_pos;
    let input_width = params.config_handler.input_width;

    // get winner neuron
    let winner_neuron = params.system_handler.winner_neuron;

    let mut sum: f64 = 0.0;

    let val1_vec = params.neuron_handler.get_weights(winner_neuron);
    let val2_vec = params.sample_handler.get_sample(sample_pos);
    for a in 0..input_width {
        let val1 = val1_vec[a];

        let val2 = val2_vec[a];
        sum += (val1 - val2) * (val1 - val2);
    }

    let curr_err = params.neuron_handler.get_error(winner_neuron);
    params
        .neuron_handler
        .set_error(winner_neuron, curr_err + sum);
}
//--------------------------------------------------------------------------------------------------
// create_neuron and sub gas
pub fn create_neuron(params: &mut Gng_System) {
    if params.neuron_handler.get_num_neurons() < params.config_handler.max_neurons {
        //------------------------------------------------------------------------------
        // get neuron with biggest error
        calc_max_error_neuron(params);
        // get best neuron amongst neighbors of winner
        calc_neighbor_neuron_vec_max_err(params);

        calc_neighbor_neuron_max_err(params);

        let max_err_neuron: usize = params.system_handler.neuron_max_err;
        let best_neighbor_neuron: usize = params.system_handler.neighbor_neuron_max_err;

        remove_edge(params);

        insert_new_neuron(params);
        let neuron_1 = params.system_handler.neuron_max_err;
        let neuron_2 = params.system_handler.neighbor_neuron_max_err;
        let err_1 = params.neuron_handler.get_error(neuron_1) * params.config_handler.alpha;
        let err_2 = params.neuron_handler.get_error(neuron_2) * params.config_handler.alpha;

        params.neuron_handler.set_error(neuron_1, err_1);
        params.neuron_handler.set_error(neuron_2, err_2);

        let neuron_1 = params.system_handler.neuron_max_err;
        let neuron_2 = params.system_handler.neighbor_neuron_max_err;
        let neuron_new = params.system_handler.newest_neuron_id;
        params.edge_handler.create_edge(neuron_new, neuron_1, 0);
        params.edge_handler.create_edge(neuron_new, neuron_2, 0);
    }
}
//---------------------------------------

pub fn calc_neighbor_neuron_max_err(params: &mut Gng_System) {
    let neighbor_neurons = &params.system_handler.neighbor_neuron_vec_max_err;
    let mut max_err_neighbor: f64 = *params.neuron_handler.get_error(neighbor_neurons[0]);
    let mut max_error_neighbor_pos = neighbor_neurons[0];
    for a in neighbor_neurons {
        let curr_err = *params.neuron_handler.get_error(*a);
        if max_err_neighbor < curr_err {
            max_err_neighbor = curr_err;
            max_error_neighbor_pos = *a;
        }
    }

    params.system_handler.neighbor_neuron_max_err = max_error_neighbor_pos;
}
pub fn calc_max_error_neuron(params: &mut Gng_System) {
    let keys = params.neuron_handler.get_all_neuron_ids();
    // Handle empty collection
    if keys.is_empty() {
        return;
    }

    let mut max_error_pos: usize = keys[0];
    let mut max_err: f64 = *params.neuron_handler.get_error(keys[0]);

    for a in 0..keys.len() {
        let curr_err = *params.neuron_handler.get_error(keys[a]);
        if max_err < curr_err {
            max_err = curr_err;
            max_error_pos = keys[a];
        }
    }
    params.system_handler.neuron_max_err = max_error_pos;
}

//--------------------------------------------------------------------------------------------------
pub fn delete_old_edges(params: &mut Gng_System) {
    let edges = delete_old_edges_mark(params);

    for a in edges {
        params.edge_handler.remove_edge(a);
    }
}
fn delete_old_edges_mark(params: &mut Gng_System) -> Vec<usize> {
    let keys = params.edge_handler.get_keys();
    let max_age = params.config_handler.edge_removal_age;
    let mut marked_edges: Vec<usize> = Vec::new();
    for &a in keys {
        if *params.edge_handler.get_edge_age(a) > max_age {
            marked_edges.push(a);
        }
    }
    marked_edges
}
//--------------------------------------------------------------------------------------------------
pub fn remove_unconnected_neurons(params: &mut Gng_System) {
    // Step 1: Get all edge keys
    let edge_keys = params.edge_handler.get_keys();

    // Step 2: Create a set to track connected neuron keys
    let mut connected_neurons = HashSet::new();

    // Step 3: For each edge, add its start and end neurons to the set
    for &edge_key in edge_keys {
        let start_key = params.edge_handler.get_edge_start(edge_key);
        let end_key = params.edge_handler.get_edge_end(edge_key);
        connected_neurons.insert(start_key);
        connected_neurons.insert(end_key);
    }

    // Step 4: Get all neuron keys
    let neuron_keys = params.neuron_handler.get_keys();

    // Step 5: Remove any neurons that are not in the connected set
    let mut marked_neurons: Vec<usize> = Vec::new();
    for neuron_key in neuron_keys {
        if !connected_neurons.contains(&neuron_key) {
            marked_neurons.push(*neuron_key);
        }
    }
    for neuron_key in marked_neurons {
        params.neuron_handler.remove_neuron(neuron_key);
    }
}

//--------------------------------------------------------------------------------------------------
pub fn create_edge(params: &mut Gng_System) {
    // examine, if winner and second neuron are connected
    let winner_neuron: usize = params.system_handler.winner_neuron;
    let second_neuron: usize = params.system_handler.second_neuron;

    let winner_edges = get_winner_edges(params);

    let mut connected = false;

    for a in 0..winner_edges.len() {
        let start = params.edge_handler.get_edge_start(winner_edges[a]);
        let end = params.edge_handler.get_edge_end(winner_edges[a]);
        if (*start == second_neuron) || (*end == second_neuron) {
            connected = true;
            params.edge_handler.set_edge_age(winner_edges[a], 0);
            break;
        }
    }
    // connect winner and second, because they are not connected
    if connected == false {
        params
            .edge_handler
            .create_edge(winner_neuron, second_neuron, 0);
    }
}
//--------------------------------------
pub fn remove_edge(params: &mut Gng_System) {
    let neuron_1 = params.system_handler.neuron_max_err;
    let neuron_2 = params.system_handler.neighbor_neuron_max_err;
    let keys: Vec<usize> = params.edge_handler.get_keys().copied().collect();

    for a in keys {
        let start = params.edge_handler.get_edge_start(a);
        let end = params.edge_handler.get_edge_end(a);
        if (*start == neuron_1 && *end == neuron_2) || (*end == neuron_1 && *start == neuron_2) {
            params.edge_handler.remove_edge(a);
            break;
        }
    }
}
pub fn insert_new_neuron(params: &mut Gng_System) {
    let neuron_1 = params.system_handler.neuron_max_err;
    let neuron_2 = params.system_handler.neighbor_neuron_max_err;
    let input_width = params.config_handler.input_width;

    // process weight between neurons and push them on W Vector
    let w_1 = params.neuron_handler.get_weights(neuron_1);
    let w_2 = params.neuron_handler.get_weights(neuron_2);
    let mut w_new: Vec<f64> = Vec::new();
    for a in 0..input_width {
        w_new.push((w_1[a] + w_2[a]) / 2.0);
    }
    // process errors
    let err_1 = params.neuron_handler.get_error(neuron_1) * params.config_handler.alpha;
    let err_2 = params.neuron_handler.get_error(neuron_2) * params.config_handler.alpha;
    let err_new = (err_1 + err_2) / 2.0;

    let new_id = params.neuron_handler.create_neuron(w_new);
    params.system_handler.newest_neuron_id = new_id;

    params.neuron_handler.set_error(new_id, err_new);
}

pub fn init_dataset(params: &mut Gng_System, filename_dataset: &String) {
    let input_width = params.config_handler.input_width;
    params
        .sample_handler
        .init_data_set(&filename_dataset, input_width);
}

pub fn init_dataset_vec(params: &mut Gng_System, data_vec: &Vec<f64>) {
    let input_width = params.config_handler.input_width;
    params.sample_handler.init_input_vec(&data_vec, input_width);
}

//--------------------------------------------------------------------------------------------------
pub fn load_config(params: &mut Gng_System, filename_config: &String) {
    params.config_handler.load_config(&filename_config);
}

//--------------------------------------------------------------------------------------------------
pub fn init_training(params: &mut Gng_System) {
    params.system_handler.train_completed = false;

    params.system_handler.reshuffle_required = true;

    init_model(params);

    params.system_handler.train_initiated = true;
}

//--------------------------------------------------------------------------------------------------
// Export Model
pub fn save_model_json(
    params: &mut Gng_System,
    output_file: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = json!({});
    let keys = params.neuron_handler.get_keys();
    let mut neuron_array: Vec<Value> = Vec::new();
    for a in keys {
        let neuron = json!({
            "id": *a,
            "position":  params.neuron_handler.get_weights(*a),
        });
        neuron_array.push(neuron);
    }
    let mut edge_array: Vec<Value> = Vec::new();
    let keys_edges = params.edge_handler.get_keys();
    for a in keys_edges {
        let edge = json!({
            "from": params.edge_handler.get_edge_start(*a),
            "to": params.edge_handler.get_edge_end(*a),
        });
        edge_array.push(edge);
    }

    //--------------------------------------------------------------------------------------------------

    // Step 1: Create an empty JSON object (Value)
    let mut data = json!({});
    // Step 2: Use write_value_to_block to add values to specific blocks
    write_value_to_block(&mut data, "model", "neurons", neuron_array);
    write_value_to_block(&mut data, "model", "edges", edge_array);

    // Step 3: Serialize and write this JSON object to a file
    write_json_to_file(&output_file.to_string(), &data)?;

    println!("JSON successfully written to {}", output_file);
    Ok(())
}

pub fn get_model_string(params: &mut Gng_System) -> String {
    let mut data = json!({});
    let keys = params.neuron_handler.get_keys();
    let mut neuron_array: Vec<Value> = Vec::new();
    let mut edge_array: Vec<Value> = Vec::new();
    for a in keys {
        let neuron = json!({
            "id": *a,
            "position":  params.neuron_handler.get_weights(*a),
        });
        neuron_array.push(neuron);
    }

    //--------------------------------------------------------------------------------------------------

    // Step 1: Create an empty JSON object (Value)
    let mut data = json!({});

    // Step 2: Use write_value_to_block to add values to specific blocks
    //let mut edge_array: Vec<Value> = Vec::new();
    write_value_to_block(&mut data, "model", "neurons", neuron_array);
    write_value_to_block(&mut data, "model", "edges", edge_array);

    // Step 3: Serialize and write this JSON object to a file
    let res_str = serde_json::to_string(&data).unwrap();
    res_str
}

// Export Model
//--------------------------------------------------------------------------------------------------
// get neurons and edges

/// Returns the current neurons as (id, position) pairs.
///
/// The positions are 3D vectors stored as (x, y, z) tuples.
pub fn get_neurons(params: &mut Gng_System) -> Vec<(usize, Vec<f64>)> {
    let mut result = Vec::with_capacity(params.neuron_handler.get_num_neurons());
    let keys = params.neuron_handler.get_keys();
    let mut neuron_array: Vec<Value> = Vec::new();
    for a in keys {
        result.push((*a, params.neuron_handler.get_weights(*a).clone()));
    }
    result
}

pub fn get_edges(params: &mut Gng_System) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(params.neuron_handler.get_num_neurons());
    let keys = params.edge_handler.get_keys();
    let mut neuron_array: Vec<Value> = Vec::new();
    for a in keys {
        result.push((
            params.edge_handler.get_edge_start(*a).clone(),
            params.edge_handler.get_edge_end(*a).clone(),
        ));
    }
    result
}

/// Returns the current edges as (from, to) pairs.
///
/// Each tuple represents a directed edge between two neurons.
//pub fn get_edges(params: &mut Gng_System) -> Vec<(usize, usize)> {
//    self.edge_handler
//        .get()
//        .iter()
//        .filter_map(|edge| {
//            if edge.active {
//                Some((edge.from, edge.to))
//            } else {
//                None
//            }
//        })
//        .collect()
//}

mod core_tests {
    use std::assert_eq;

    use crate::gas::gng_system;

    use super::*;
    #[test]
    fn test_init_dataset_t1() {
        let filename_input = "test_data/growing_neural_gas/init_dataset/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/init_dataset/target.json".to_string();
        let filename_dataset = "test_data/growing_neural_gas/init_dataset/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        //------------------------------------------------------------------------------

        let mut gng_params: Gng_System = Gng_System::init();

        // input
        //----------------------------------------------------------
        // input

        //----------------------------------------------------------
        // output
        let samples_target = json_reader::read_array_f64(&reader_target, "target", "samples");
        let num_samples_target =
            json_reader::read_val_usize(&reader_target, "target", "num_samples");
        // output
        //----------------------------------------------------------
        // function call
        configure_model(filename_input, &mut gng_params);
        let input_width = gng_params.config_handler.input_width;
        init_dataset(&mut gng_params, &filename_dataset);
        let samples_res = gng_params.sample_handler.get_samples_vec();
        let num_samples_res = gng_params.sample_handler.get_num_samples();
        // function call
        //----------------------------------------------------------
        // validation

        assert_eq!(num_samples_res, num_samples_target);

        for a in 0..samples_res.len() {
            assert!((samples_res[a] - samples_target[a]).abs() < 0.0001);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    pub fn load_model_t1() {
        let filename_input = "test_data/growing_neural_gas/load_model/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/load_model/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();

        // init
        //----------------------------------------------------------
        // target
        let weights_target = json_reader::read_array_f64(&reader_target, "target", "weights");
        let edge_start_target =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let edge_end_target = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let edge_age_target = json_reader::read_array_usize(&reader_target, "target", "edge_age");
        // target
        //----------------------------------------------------------
        // function call

        load_model(&mut params, filename_input);

        // function call
        //----------------------------------------------------------
        // validation
        let width = params.config_handler.input_width;
        let weights_res = params.neuron_handler.get_weight_vec(&width);
        let edge_start_res = params.edge_handler.get_edge_start_vec();
        let edge_end_res = params.edge_handler.get_edge_end_vec();
        let edge_age_res = params.edge_handler.get_edge_age_vec();

        for a in 0..weights_res.len() {
            assert!((weights_res[a] - weights_target[a]).abs() < 0.0001);
        }
        for a in 0..edge_start_res.len() {
            assert_eq!(edge_start_res[a], edge_start_target[a]);
        }
        for a in 0..edge_end_res.len() {
            assert_eq!(edge_end_res[a], edge_end_target[a]);
        }
        for a in 0..edge_age_res.len() {
            assert_eq!(edge_age_res[a], edge_age_target[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn test_shuffle_dataset_deterministic() {
        let rng_seed = 12345;
        let mut samples = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let sample_width = 2;

        // First call
        let mut params: Gng_System = Gng_System::init();
        params.rng_manager.seed_rng(rng_seed);
        params
            .sample_handler
            .init_input_vec(&mut samples, sample_width);
        let mut res_pre_shuffle_1: Vec<usize> = Vec::new();

        shuffle_dataset(&mut params);

        let sample_order_1 = params.system_handler.sample_order;
        //========================================
        // Second call with same data
        let mut params_2: Gng_System = Gng_System::init();
        params_2.rng_manager.seed_rng(rng_seed);
        //let foo2 = params_2.rng_manager.get_f64(-1.0, 1.0);
        params_2
            .sample_handler
            .init_input_vec(&mut samples, sample_width);

        shuffle_dataset(&mut params_2);

        let sample_order_2 = params_2.system_handler.sample_order;
        assert_eq!(sample_order_1, sample_order_2);
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn test_init_model_t1() {
        let filename_input = "test_data/growing_neural_gas/init_model/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/init_model/target.json".to_string();

        let reader_input = json_reader::read_file(&filename_input).unwrap();
        let reader_target = json_reader::read_file(&filename_target).unwrap();

        //------------------------------------------------------------------------------

        let mut comps: Gng_System = Gng_System::init();
        comps.create_system();

        // input
        //----------------------------------------------------------
        // input

        comps.config_handler.input_width =
            json_reader::read_val_usize(&reader_input, "config", "input_width");
        comps.config_handler.weight_rng_min =
            json_reader::read_val_f64(&reader_input, "config", "weight_rng_min");
        comps.config_handler.weight_rng_max =
            json_reader::read_val_f64(&reader_input, "config", "weight_rng_max");

        //----------------------------------------------------------
        // output
        let w_target = json_reader::read_array_f64(&reader_target, "target", "w");
        let edge_start_target =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let edge_end_target = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        // output
        //----------------------------------------------------------
        // function call

        init_model(&mut comps);

        // function call
        //----------------------------------------------------------
        // validation

        let input_width = comps.config_handler.input_width;
        let weights = comps.neuron_handler.get_weight_vec(&input_width);

        let edges_start = comps.edge_handler.get_edge_start_vec();
        let edges_end = comps.edge_handler.get_edge_end_vec();

        assert_eq!(weights.len(), w_target.len());
        assert_eq!(edges_start.len(), edge_start_target.len());
        assert_eq!(edges_end.len(), edge_end_target.len());

        for a in 0..weights.len() {
            assert!((weights[a] - w_target[a]).abs() < 0.0001);
        }
        for a in 0..edges_start.len() {
            assert_eq!(edges_start[a], edge_start_target[a]);
        }
        for a in 0..edges_end.len() {
            assert_eq!(edges_end[a], edge_end_target[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------

    #[test]
    fn test_init_model_t1_deterministic() {
        let filename_input = "test_data/growing_neural_gas/init_model/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/init_model/target.json".to_string();

        let reader_input = json_reader::read_file(&filename_input).unwrap();
        let reader_target = json_reader::read_file(&filename_target).unwrap();

        // Setup
        let mut comps1: Gng_System = Gng_System::init();
        let mut comps2: Gng_System = Gng_System::init();

        comps1.create_system();
        comps2.create_system();

        // Input configuration
        comps1.config_handler.input_width =
            json_reader::read_val_usize(&reader_input, "config", "input_width");
        comps1.config_handler.weight_rng_min =
            json_reader::read_val_f64(&reader_input, "config", "weight_rng_min");
        comps1.config_handler.weight_rng_max =
            json_reader::read_val_f64(&reader_input, "config", "weight_rng_max");

        comps2.config_handler.input_width =
            json_reader::read_val_usize(&reader_input, "config", "input_width");
        comps2.config_handler.weight_rng_min =
            json_reader::read_val_f64(&reader_input, "config", "weight_rng_min");
        comps2.config_handler.weight_rng_max =
            json_reader::read_val_f64(&reader_input, "config", "weight_rng_max");

        // Run with same seed (e.g., 42)
        // let seed = 42u64;
        // comps1.set_rng_seed(seed); // Assume this method exists to set the RNG seed
        // comps2.set_rng_seed(seed);

        init_model(&mut comps1);
        init_model(&mut comps2);

        // Validation: compare results
        let input_width = comps1.config_handler.input_width;
        let weights1 = comps1.neuron_handler.get_weight_vec(&input_width);
        let edges_start1 = comps1.edge_handler.get_edge_start_vec();
        let edges_end1 = comps1.edge_handler.get_edge_end_vec();

        let weights2 = comps2.neuron_handler.get_weight_vec(&input_width);
        let edges_start2 = comps2.edge_handler.get_edge_start_vec();
        let edges_end2 = comps2.edge_handler.get_edge_end_vec();

        assert_eq!(weights1, weights2);
        assert_eq!(edges_start1, edges_start2);
        assert_eq!(edges_end1, edges_end2);
    }
    //--------------------------------------------------------------------------------------------------

    #[test]
    pub fn calc_neuron_distances_t1() {
        let filename_input =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();

        params.system_handler.curr_sample_pos = 1;
        // init
        //----------------------------------------------------------
        // target
        let distance_target = json_reader::read_array_f64(&reader_target, "target", "distance");
        // target
        //----------------------------------------------------------
        // function call

        load_model(&mut params, filename_input);
        let input_width = params.config_handler.input_width;
        init_dataset(&mut params, &filename_dataset);
        calc_neuron_distances(&mut params);

        // function call
        //----------------------------------------------------------
        // validation
        for a in 0..distance_target.len() {
            assert!((params.neuron_handler.get_distance(a) - distance_target[a]).abs() < 0.0001);
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    pub fn calc_neuron_distances_determinism() {
        let filename_input =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();

        // First execution
        let mut params1: Gng_System = Gng_System::init();
        params1.create_system();

        params1.system_handler.curr_sample_pos = 1;

        // Load and initialize
        load_model(&mut params1, filename_input.clone());
        let input_width = params1.config_handler.input_width;
        init_dataset(&mut params1, &filename_dataset);

        // First call
        calc_neuron_distances(&mut params1);

        // Second execution with same setup
        let mut params2: Gng_System = Gng_System::init();
        params2.create_system();

        params2.system_handler.curr_sample_pos = 1;

        // Load and initialize (same data)
        load_model(&mut params2, filename_input);
        init_dataset(&mut params2, &filename_dataset);

        // Second call
        calc_neuron_distances(&mut params2);

        // Get target distances
        let distance_target = json_reader::read_array_f64(&reader_target, "target", "distance");

        // Validate both executions produce identical results
        for a in 0..distance_target.len() {
            let dist1 = params1.neuron_handler.get_distance(a);
            let dist2 = params2.neuron_handler.get_distance(a);
            let target = distance_target[a];

            // Check that both executions match target
            assert!(
                (dist1 - target).abs() < 0.0001,
                "First execution mismatch at index {}: {} vs {}",
                a,
                dist1,
                target
            );
            assert!(
                (dist2 - target).abs() < 0.0001,
                "Second execution mismatch at index {}: {} vs {}",
                a,
                dist2,
                target
            );

            // Check that both executions are identical to each other
            assert!(
                (dist1 - dist2).abs() < 0.0001,
                "Non-deterministic result: execution 1 = {}, execution 2 = {} at index {}",
                dist1,
                dist2,
                a
            );
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn calc_nearest_neurons_t1() {
        let filename_input =
            "test_data/growing_neural_gas/calc_nearest_neurons_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_nearest_neurons_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();
        //------------------------------------------------------------------------------
        // input
        let distance_input =
            json_reader::read_array_f64(&reader_input, "config", "neuron_distance");
        let num_neurons = json_reader::read_val_usize(&reader_input, "config", "num_neurons");
        // input
        //------------------------------------------------------------------------------
        // output
        let winner_neuron_target =
            json_reader::read_val_usize(&reader_target, "target", "winner_neuron");
        let second_neuron_target =
            json_reader::read_val_usize(&reader_target, "target", "second_neuron");

        // output
        //------------------------------------------------------------------------------
        // function call
        params.neuron_handler.init_neurons_debug(num_neurons);
        params.neuron_handler.set_distances_debug(distance_input);
        calc_nearest_neurons(&mut params);

        // function call
        //------------------------------------------------------------------------------
        // validation
        //
        assert_eq!(params.system_handler.winner_neuron, winner_neuron_target);

        assert_eq!(params.system_handler.second_neuron, second_neuron_target);
    }
    //--------------------------------------------------------------------------------------------------

    #[test]
    fn calc_nearest_neurons_t2() {
        let filename_input =
            "test_data/growing_neural_gas/calc_nearest_neurons_t2/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_nearest_neurons_t2/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();
        //------------------------------------------------------------------------------
        // input
        let distance_input =
            json_reader::read_array_f64(&reader_input, "config", "neuron_distance");
        let num_neurons = json_reader::read_val_usize(&reader_input, "config", "num_neurons");
        // input
        //------------------------------------------------------------------------------
        // output
        let winner_neuron_target =
            json_reader::read_val_usize(&reader_target, "target", "winner_neuron");
        let second_neuron_target =
            json_reader::read_val_usize(&reader_target, "target", "second_neuron");

        // output
        //------------------------------------------------------------------------------
        // function call
        params.neuron_handler.init_neurons_debug(num_neurons);
        params.neuron_handler.set_distances_debug(distance_input);
        calc_nearest_neurons(&mut params);

        params.neuron_handler.print_neurons();
        // function call
        //------------------------------------------------------------------------------
        // validation
        //
        assert_eq!(params.system_handler.winner_neuron, winner_neuron_target);

        assert_eq!(params.system_handler.second_neuron, second_neuron_target);
    }

    #[test]
    fn calc_nearest_neurons_determinism() {
        let filename_input =
            "test_data/growing_neural_gas/calc_nearest_neurons_t2/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_nearest_neurons_t2/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        // Read target values once
        let winner_neuron_target =
            json_reader::read_val_usize(&reader_target, "target", "winner_neuron");
        let second_neuron_target =
            json_reader::read_val_usize(&reader_target, "target", "second_neuron");

        // First run
        let mut params1: Gng_System = Gng_System::init();
        params1.create_system();

        let distance_input =
            json_reader::read_array_f64(&reader_input, "config", "neuron_distance");
        let num_neurons = json_reader::read_val_usize(&reader_input, "config", "num_neurons");

        params1.neuron_handler.init_neurons_debug(num_neurons);
        params1
            .neuron_handler
            .set_distances_debug(distance_input.clone());
        calc_nearest_neurons(&mut params1);

        // Second run with identical inputs
        let mut params2: Gng_System = Gng_System::init();
        params2.create_system();

        params2.neuron_handler.init_neurons_debug(num_neurons);
        params2
            .neuron_handler
            .set_distances_debug(distance_input.clone());
        calc_nearest_neurons(&mut params2);

        // Validate both runs produce identical results
        assert_eq!(params1.system_handler.winner_neuron, winner_neuron_target);
        assert_eq!(params2.system_handler.winner_neuron, winner_neuron_target);

        assert_eq!(params1.system_handler.second_neuron, second_neuron_target);
        assert_eq!(params2.system_handler.second_neuron, second_neuron_target);

        // Additional determinism check: ensure both runs produce identical system states
        assert_eq!(
            params1.system_handler.winner_neuron,
            params2.system_handler.winner_neuron
        );
        assert_eq!(
            params1.system_handler.second_neuron,
            params2.system_handler.second_neuron
        );

        // Optional: check that neuron positions are identical (if applicable)
        // This would depend on your specific implementation details
        // assert_eq!(params1.neuron_handler.neurons, params2.neuron_handler.neurons);
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn calc_neuron_dependencies_t1() {
        let filename_input =
            "test_data/growing_neural_gas/calc_neuron_dependencies_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_neuron_dependencies_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();
        //------------------------------------------------------------------------------
        // input
        let winner_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");
        let second_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "second_neuron");
        // input
        //------------------------------------------------------------------------------
        // output
        let target_neuron_dependencies =
            json_reader::read_array_usize(&reader_target, "target", "neuron_dependencies");

        // output
        //------------------------------------------------------------------------------
        // preparation
        load_model(&mut params, filename_input);
        params.system_handler.winner_neuron = winner_neuron_input;
        params.system_handler.second_neuron = second_neuron_input;
        // preparation
        //------------------------------------------------------------------------------
        // function call
        calc_neuron_dependencies(&mut params);

        // function call
        //------------------------------------------------------------------------------
        // validation

        for a in 0..target_neuron_dependencies.len() {
            assert_eq!(
                *params.neuron_handler.get_neuron_dependency(a) as usize,
                target_neuron_dependencies[a] as usize
            );
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn calc_neuron_dependencies_determinism() {
        let filename_input =
            "test_data/growing_neural_gas/calc_neuron_dependencies_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_neuron_dependencies_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        // Read target data once
        let target_neuron_dependencies =
            json_reader::read_array_usize(&reader_target, "target", "neuron_dependencies");

        // First run
        let mut params1: Gng_System = Gng_System::init();
        params1.create_system();

        // Preparation for first run
        load_model(&mut params1, filename_input.clone());
        let winner_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");
        let second_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "second_neuron");
        params1.system_handler.winner_neuron = winner_neuron_input;
        params1.system_handler.second_neuron = second_neuron_input;

        // First function call
        calc_neuron_dependencies(&mut params1);

        // Second run with fresh system
        let mut params2: Gng_System = Gng_System::init();
        params2.create_system();

        // Preparation for second run (same inputs)
        load_model(&mut params2, filename_input.clone());
        params2.system_handler.winner_neuron = winner_neuron_input;
        params2.system_handler.second_neuron = second_neuron_input;

        // Second function call
        calc_neuron_dependencies(&mut params2);

        // Validate both runs produce identical results
        for a in 0..target_neuron_dependencies.len() {
            assert_eq!(
                *params1.neuron_handler.get_neuron_dependency(a) as usize,
                *params2.neuron_handler.get_neuron_dependency(a) as usize,
                "Determinism check failed: different results between runs at index {}",
                a
            );
            assert_eq!(
                *params1.neuron_handler.get_neuron_dependency(a) as usize,
                target_neuron_dependencies[a] as usize,
                "First run result doesn't match target at index {}",
                a
            );
        }

        // Additional check: ensure both results are identical to target
        for a in 0..target_neuron_dependencies.len() {
            assert_eq!(
                *params1.neuron_handler.get_neuron_dependency(a) as usize,
                target_neuron_dependencies[a] as usize,
                "First run doesn't match target at index {}",
                a
            );
            assert_eq!(
                *params2.neuron_handler.get_neuron_dependency(a) as usize,
                target_neuron_dependencies[a] as usize,
                "Second run doesn't match target at index {}",
                a
            );
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn add_error_to_winner_neuron_t1() {
        let filename_input =
            "test_data/growing_neural_gas/add_error_to_winner_neuron_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/add_error_to_winner_neuron_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/add_error_to_winner_neuron_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();
        //------------------------------------------------------------------------------
        // input

        let sample_pos = json_reader::read_val_usize(&reader_input, "config", "curr_sample_pos");
        let neuron_err_input =
            json_reader::read_array_f64(&reader_input, "gng_model", "neuron_err");
        let winner_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");
        // input
        //------------------------------------------------------------------------------
        // output
        let neuron_err_target = json_reader::read_array_f64(&reader_target, "target", "neuron_err");

        // output
        //------------------------------------------------------------------------------
        // function call

        load_model(&mut params, filename_input);
        let input_width = params.config_handler.input_width;
        params
            .sample_handler
            .init_data_set(&filename_dataset, input_width);
        params.system_handler.curr_sample_pos = sample_pos;
        params
            .neuron_handler
            .set_errors_debug(neuron_err_input.clone());
        params.system_handler.winner_neuron = winner_neuron_input;
        add_error_to_winner_neuron(&mut params);

        // function call
        //------------------------------------------------------------------------------
        // validation
        for a in 0..neuron_err_target.len() {
            assert!((neuron_err_target[a] - params.neuron_handler.get_error(a)).abs() < 0.001);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn add_error_to_winner_neuron_determinism() {
        let filename_input =
            "test_data/growing_neural_gas/add_error_to_winner_neuron_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/add_error_to_winner_neuron_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/add_error_to_winner_neuron_t1/dataset.csv".to_string();

        // First run
        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params1: Gng_System = Gng_System::init();
        params1.create_system();

        // Input setup for first run
        let sample_pos = json_reader::read_val_usize(&reader_input, "config", "curr_sample_pos");
        let neuron_err_input =
            json_reader::read_array_f64(&reader_input, "gng_model", "neuron_err");
        let winner_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");

        load_model(&mut params1, filename_input.clone());
        let input_width = params1.config_handler.input_width;
        params1
            .sample_handler
            .init_data_set(&filename_dataset, input_width);
        params1.system_handler.curr_sample_pos = sample_pos;
        params1
            .neuron_handler
            .set_errors_debug(neuron_err_input.clone());
        params1.system_handler.winner_neuron = winner_neuron_input;
        add_error_to_winner_neuron(&mut params1);

        // Second run with identical inputs
        let mut params2: Gng_System = Gng_System::init();
        params2.create_system();

        // Same input setup for second run
        load_model(&mut params2, filename_input);
        params2
            .sample_handler
            .init_data_set(&filename_dataset, input_width);
        params2.system_handler.curr_sample_pos = sample_pos;
        params2
            .neuron_handler
            .set_errors_debug(neuron_err_input.clone());
        params2.system_handler.winner_neuron = winner_neuron_input;
        add_error_to_winner_neuron(&mut params2);

        // Validation: Compare all neuron errors between runs
        let num_neurons = params1.neuron_handler.get_num_neurons();
        assert_eq!(num_neurons, params2.neuron_handler.get_num_neurons());

        for i in 0..num_neurons {
            let error1 = params1.neuron_handler.get_error(i);
            let error2 = params2.neuron_handler.get_error(i);
            assert!(
                (error1 - error2).abs() < 0.001,
                "Determinism failed: neuron {} has different errors between runs ({}, {})",
                i,
                error1,
                error2
            );
        }

        // Also validate against target values to ensure both runs are correct
        let neuron_err_target = json_reader::read_array_f64(&reader_target, "target", "neuron_err");
        for a in 0..neuron_err_target.len() {
            assert!((neuron_err_target[a] - params1.neuron_handler.get_error(a)).abs() < 0.001);
            assert!((neuron_err_target[a] - params2.neuron_handler.get_error(a)).abs() < 0.001);
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn update_weights_t1() {
        let filename_input =
            "test_data/growing_neural_gas/update_weights_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/update_weights_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/update_weights_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();

        //------------------------------------------------------------------------------
        // input

        let epsilon_w: f64 = json_reader::read_val_f64(&reader_input, "config", "epsilon_w");
        let epsilon_n: f64 = json_reader::read_val_f64(&reader_input, "config", "epsilon_n");

        let sample_pos = json_reader::read_val_usize(&reader_input, "state", "curr_sample_pos");
        let neuron_dependencies =
            json_reader::read_array_usize(&reader_input, "state", "neuron_dependencies");

        // input
        //------------------------------------------------------------------------------
        // output
        let weights_target = json_reader::read_array_f64(&reader_target, "target", "weights");

        // output
        //------------------------------------------------------------------------------
        // function call
        load_model(&mut params, filename_input);
        let input_width = params.config_handler.input_width;
        init_dataset(&mut params, &filename_dataset);
        params.config_handler.epsilon_w = epsilon_w;
        params.config_handler.epsilon_n = epsilon_n;
        params
            .neuron_handler
            .set_neuron_dependencies_debug(neuron_dependencies);
        params.system_handler.curr_sample_pos = sample_pos;
        update_weights(&mut params);

        let weights_res = params.neuron_handler.get_weight_vec(&input_width);
        for a in 0..weights_target.len() {
            assert!((weights_res[a] - weights_target[a]).abs() < 0.001);
        }
    }

    //--------------------------------------------------------------------------------------------------

    #[test]
    fn update_weights_determinism() {
        let filename_input =
            "test_data/growing_neural_gas/update_weights_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/update_weights_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/update_weights_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        // First run
        let mut params1: Gng_System = Gng_System::init();
        params1.create_system();

        let epsilon_w: f64 = json_reader::read_val_f64(&reader_input, "config", "epsilon_w");
        let epsilon_n: f64 = json_reader::read_val_f64(&reader_input, "config", "epsilon_n");
        let sample_pos = json_reader::read_val_usize(&reader_input, "state", "curr_sample_pos");
        let neuron_dependencies =
            json_reader::read_array_usize(&reader_input, "state", "neuron_dependencies");

        load_model(&mut params1, filename_input.clone());
        let input_width = params1.config_handler.input_width;
        init_dataset(&mut params1, &filename_dataset);
        params1.config_handler.epsilon_w = epsilon_w;
        params1.config_handler.epsilon_n = epsilon_n;
        params1
            .neuron_handler
            .set_neuron_dependencies_debug(neuron_dependencies.clone());
        params1.system_handler.curr_sample_pos = sample_pos;
        update_weights(&mut params1);

        let weights_res1 = params1.neuron_handler.get_weight_vec(&input_width);

        // Second run with same inputs
        let mut params2: Gng_System = Gng_System::init();
        params2.create_system();

        load_model(&mut params2, filename_input);
        init_dataset(&mut params2, &filename_dataset);
        params2.config_handler.epsilon_w = epsilon_w;
        params2.config_handler.epsilon_n = epsilon_n;
        params2
            .neuron_handler
            .set_neuron_dependencies_debug(neuron_dependencies);
        params2.system_handler.curr_sample_pos = sample_pos;
        update_weights(&mut params2);

        let weights_res2 = params2.neuron_handler.get_weight_vec(&input_width);

        // Check that both runs produce identical results
        assert_eq!(weights_res1.len(), weights_res2.len());
        for (i, (&weight1, &weight2)) in weights_res1.iter().zip(weights_res2.iter()).enumerate() {
            assert!(
                (weight1 - weight2).abs() < 0.001,
                "Weights differ at index {}: {} vs {}",
                i,
                weight1,
                weight2
            );
        }
    }

    //--------------------------------------------------------------------------------------------------

    #[test]
    fn create_edge_t1() {
        let filename_input = "test_data/growing_neural_gas/create_edge_t1/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/create_edge_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();

        //------------------------------------------------------------------------------
        // input

        let neuron_dependencies =
            json_reader::read_array_usize(&reader_input, "state", "neuron_dependencies");
        let winner_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");
        let second_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "second_neuron");

        // input
        //------------------------------------------------------------------------------
        // output
        let edge_start_target =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let edge_end_target = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let edge_age_target = json_reader::read_array_usize(&reader_target, "target", "edge_age");

        // output
        //------------------------------------------------------------------------------
        // preparation
        load_model(&mut params, filename_input);
        //arams.neuron_handler.set_distance_orders_debug(distance_order);
        params
            .neuron_handler
            .set_neuron_dependencies_debug(neuron_dependencies);
        params.system_handler.winner_neuron = winner_neuron_input;
        params.system_handler.second_neuron = second_neuron_input;
        // preparation
        //------------------------------------------------------------------------------
        // function call
        create_edge(&mut params);
        // function call
        //------------------------------------------------------------------------------
        // validation
        let edge_start_res = params.edge_handler.get_edge_start_vec();
        let edge_end_res = params.edge_handler.get_edge_end_vec();
        let edge_age_res = params.edge_handler.get_edge_age_vec();

        for a in 0..edge_start_target.len() {
            assert_eq!(edge_start_res[a], edge_start_target[a]);
            assert_eq!(edge_end_res[a], edge_end_target[a]);
            assert_eq!(edge_age_res[a], edge_age_target[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------

    #[test]
    fn create_edge_t2() {
        let filename_input = "test_data/growing_neural_gas/create_edge_t2/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/create_edge_t2/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();

        //------------------------------------------------------------------------------
        // input

        let neuron_dependencies =
            json_reader::read_array_usize(&reader_input, "state", "neuron_dependencies");
        let winner_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");
        let second_neuron_input =
            json_reader::read_val_usize(&reader_input, "config", "second_neuron");

        // input
        //------------------------------------------------------------------------------
        // output
        let edge_start_target =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let edge_end_target = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let edge_age_target = json_reader::read_array_usize(&reader_target, "target", "edge_age");

        // output
        //------------------------------------------------------------------------------
        // preparation
        load_model(&mut params, filename_input);
        //arams.neuron_handler.set_distance_orders_debug(distance_order);
        params
            .neuron_handler
            .set_neuron_dependencies_debug(neuron_dependencies);
        params.system_handler.winner_neuron = winner_neuron_input;
        params.system_handler.second_neuron = second_neuron_input;
        // preparation
        //------------------------------------------------------------------------------
        // function call
        create_edge(&mut params);
        // function call
        //------------------------------------------------------------------------------
        // validation
        let edge_start_res = params.edge_handler.get_edge_start_vec();
        let edge_end_res = params.edge_handler.get_edge_end_vec();
        let edge_age_res = params.edge_handler.get_edge_age_vec();

        for a in 0..edge_start_target.len() {
            assert_eq!(edge_start_res[a], edge_start_target[a]);
            assert_eq!(edge_end_res[a], edge_end_target[a]);
            assert_eq!(edge_age_res[a], edge_age_target[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------

    #[test]
    fn test_determinism_create_edge() {
        let filename_input = "test_data/growing_neural_gas/create_edge_t2/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/create_edge_t2/target.json".to_string();

        // Load inputs once
        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        // Read expected outputs from target file
        let edge_start_target =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let edge_end_target = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let edge_age_target = json_reader::read_array_usize(&reader_target, "target", "edge_age");

        // First run
        let mut params1: Gng_System = Gng_System::init();
        params1.create_system();
        load_model(&mut params1, filename_input.clone());
        params1
            .neuron_handler
            .set_neuron_dependencies_debug(json_reader::read_array_usize(
                &reader_input,
                "state",
                "neuron_dependencies",
            ));
        params1.system_handler.winner_neuron =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");
        params1.system_handler.second_neuron =
            json_reader::read_val_usize(&reader_input, "config", "second_neuron");

        create_edge(&mut params1);

        let edge_start_res1 = params1.edge_handler.get_edge_start_vec();
        let edge_end_res1 = params1.edge_handler.get_edge_end_vec();
        let edge_age_res1 = params1.edge_handler.get_edge_age_vec();

        // Second run (with same input)
        let mut params2: Gng_System = Gng_System::init();
        params2.create_system();
        load_model(&mut params2, filename_input.clone());
        params2
            .neuron_handler
            .set_neuron_dependencies_debug(json_reader::read_array_usize(
                &reader_input,
                "state",
                "neuron_dependencies",
            ));
        params2.system_handler.winner_neuron =
            json_reader::read_val_usize(&reader_input, "config", "winner_neuron");
        params2.system_handler.second_neuron =
            json_reader::read_val_usize(&reader_input, "config", "second_neuron");

        create_edge(&mut params2);

        let edge_start_res2 = params2.edge_handler.get_edge_start_vec();
        let edge_end_res2 = params2.edge_handler.get_edge_end_vec();
        let edge_age_res2 = params2.edge_handler.get_edge_age_vec();

        // Validate deterministic behavior
        assert_eq!(edge_start_res1, edge_start_res2);
        assert_eq!(edge_end_res1, edge_end_res2);
        assert_eq!(edge_age_res1, edge_age_res2);

        // Optional: also validate against target if needed
        for a in 0..edge_start_target.len() {
            assert_eq!(edge_start_res1[a], edge_start_target[a]);
            assert_eq!(edge_end_res1[a], edge_end_target[a]);
            assert_eq!(edge_age_res1[a], edge_age_target[a]);
        }
    }

    //--------------------------------------------------------------------------------------------------

    #[test]
    pub fn create_neuron_t1() {
        // Setup
        let filename_input = "test_data/growing_neural_gas/create_neuron_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/create_neuron_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/create_neuron_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();
        // Setup
        //------------------------------------------------------------------------------
        // input

        let target_neuron = json_reader::read_val_usize(&reader_input, "state", "curr_sample_pos");

        let neuron_err = json_reader::read_array_f64(&reader_input, "state", "neuron_err");

        let alpha = json_reader::read_val_f64(&reader_input, "config", "alpha");
        let max_neurons = json_reader::read_val_usize(&reader_input, "config", "max_neurons");

        // input
        //------------------------------------------------------------------------------
        // output

        let target_w = json_reader::read_array_f64(&reader_target, "target", "weights");
        let target_edge_start =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let target_edge_age = json_reader::read_array_usize(&reader_target, "target", "edge_age");
        let target_neuron_err = json_reader::read_array_f64(&reader_target, "target", "neuron_err");

        // output
        //------------------------------------------------------------------------------
        // preparation
        load_model(&mut params, filename_input);
        params.config_handler.alpha = alpha;
        params.config_handler.max_neurons = max_neurons;
        params.system_handler.curr_neuron = target_neuron;
        params.neuron_handler.set_errors_debug(neuron_err.clone());

        params
            .sample_handler
            .init_data_set(&filename_dataset, params.config_handler.input_width);
        // preparation
        //------------------------------------------------------------------------------

        // function call
        create_neuron(&mut params);
        // function call
        //------------------------------------------------------------------------------

        let keys_neuron = params.neuron_handler.get_keys();
        let keys_edge = params.edge_handler.get_keys();

        let input_width = params.config_handler.input_width;

        for a in keys_neuron {
            assert!((params.neuron_handler.get_error(*a) - target_neuron_err[*a]).abs() < 0.0001);
            let w_temp = params.neuron_handler.get_weights(*a);
            for w in 0..input_width {
                assert!((target_w[(a * input_width) + w] - w_temp[w]).abs() < 0.0001);
            }
        }
        for a in keys_edge {
            assert_eq!(
                params.edge_handler.get_edge_start(*a),
                &target_edge_start[*a]
            );
            assert_eq!(params.edge_handler.get_edge_end(*a), &target_edge_end[*a]);
            assert_eq!(params.edge_handler.get_edge_age(*a), &target_edge_age[*a]);
        }
    }
    #[test]
    pub fn create_neuron_t2() {
        // Setup
        let filename_input = "test_data/growing_neural_gas/create_neuron_t2/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/create_neuron_t2/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/create_neuron_t2/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();
        // Setup
        //------------------------------------------------------------------------------
        // input

        let target_neuron = json_reader::read_val_usize(&reader_input, "state", "curr_sample_pos");

        let neuron_err = json_reader::read_array_f64(&reader_input, "state", "neuron_err");

        let alpha = json_reader::read_val_f64(&reader_input, "config", "alpha");
        let max_neurons = json_reader::read_val_usize(&reader_input, "config", "max_neurons");

        // input
        //------------------------------------------------------------------------------
        // output

        let target_w = json_reader::read_array_f64(&reader_target, "target", "weights");
        let target_edge_start =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let target_edge_age = json_reader::read_array_usize(&reader_target, "target", "edge_age");
        let target_neuron_err = json_reader::read_array_f64(&reader_target, "target", "neuron_err");

        // output
        //------------------------------------------------------------------------------
        // preparation
        load_model(&mut params, filename_input);
        params.config_handler.alpha = alpha;
        params.config_handler.max_neurons = max_neurons;
        params.system_handler.curr_neuron = target_neuron;
        params.neuron_handler.set_errors_debug(neuron_err.clone());

        params
            .sample_handler
            .init_data_set(&filename_dataset, params.config_handler.input_width);
        // preparation
        //------------------------------------------------------------------------------

        // function call
        create_neuron(&mut params);
        // function call
        //------------------------------------------------------------------------------

        let keys_neuron = params.neuron_handler.get_keys();
        let keys_edge = params.edge_handler.get_keys();

        let input_width = params.config_handler.input_width;
        let keys_neuron_vec: Vec<usize> = keys_neuron.copied().collect();
        for a in &keys_neuron_vec {
            assert!((params.neuron_handler.get_error(*a) - target_neuron_err[*a]).abs() < 0.0001);
            let w_temp = params.neuron_handler.get_weights(*a);
            for w in 0..input_width {
                assert!((target_w[(*a * input_width) + w] - w_temp[w]).abs() < 0.0001);
            }
        }
        for a in keys_edge {
            assert_eq!(
                params.edge_handler.get_edge_start(*a),
                &target_edge_start[*a]
            );
            assert_eq!(params.edge_handler.get_edge_end(*a), &target_edge_end[*a]);
            assert_eq!(params.edge_handler.get_edge_age(*a), &target_edge_age[*a]);
        }
    }

    #[test]
    pub fn create_neuron_t2_determinism() {
        // Setup (same as original test)
        let filename_input = "test_data/growing_neural_gas/create_neuron_t2/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/create_neuron_t2/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/create_neuron_t2/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params1: Gng_System = Gng_System::init();
        let mut params2: Gng_System = Gng_System::init();

        params1.create_system();
        params2.create_system();

        // Input setup
        let target_neuron = json_reader::read_val_usize(&reader_input, "state", "curr_sample_pos");
        let neuron_err = json_reader::read_array_f64(&reader_input, "state", "neuron_err");

        let alpha = json_reader::read_val_f64(&reader_input, "config", "alpha");
        let max_neurons = json_reader::read_val_usize(&reader_input, "config", "max_neurons");

        // Output target values
        let target_w = json_reader::read_array_f64(&reader_target, "target", "weights");
        let target_edge_start =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let target_edge_age = json_reader::read_array_usize(&reader_target, "target", "edge_age");
        let target_neuron_err = json_reader::read_array_f64(&reader_target, "target", "neuron_err");

        // Load model and initialize both systems identically
        load_model(&mut params1, filename_input.clone());
        load_model(&mut params2, filename_input.clone());

        params1.config_handler.alpha = alpha;
        params1.config_handler.max_neurons = max_neurons;
        params1.system_handler.curr_neuron = target_neuron;
        params1.neuron_handler.set_errors_debug(neuron_err.clone());

        params2.config_handler.alpha = alpha;
        params2.config_handler.max_neurons = max_neurons;
        params2.system_handler.curr_neuron = target_neuron;
        params2.neuron_handler.set_errors_debug(neuron_err.clone());

        params1
            .sample_handler
            .init_data_set(&filename_dataset, params1.config_handler.input_width);
        params2
            .sample_handler
            .init_data_set(&filename_dataset, params2.config_handler.input_width);

        // Call function on both systems
        create_neuron(&mut params1);
        create_neuron(&mut params2);

        // Compare results from both runs
        let keys_neuron1 = params1.neuron_handler.get_keys();
        let keys_neuron2 = params2.neuron_handler.get_keys();

        let keys_edge1 = params1.edge_handler.get_keys();
        let keys_edge2 = params2.edge_handler.get_keys();

        let input_width = params1.config_handler.input_width;

        let keys_neuron_vec1: Vec<usize> = keys_neuron1.copied().collect();
        let keys_neuron_vec2: Vec<usize> = keys_neuron2.copied().collect();

        assert_eq!(keys_neuron_vec1.len(), keys_neuron_vec2.len());

        for a in &keys_neuron_vec1 {
            assert!(
                (params1.neuron_handler.get_error(*a) - params2.neuron_handler.get_error(*a)).abs()
                    < 0.0001,
                "Neuron error mismatch at neuron {}",
                a
            );

            let w1 = params1.neuron_handler.get_weights(*a);
            let w2 = params2.neuron_handler.get_weights(*a);
            for w in 0..input_width {
                assert!(
                    (w1[w] - w2[w]).abs() < 0.0001,
                    "Weights mismatch at neuron {}, feature {}",
                    a,
                    w
                );
            }
        }

        for a in keys_edge1 {
            assert_eq!(
                params1.edge_handler.get_edge_start(*a),
                params2.edge_handler.get_edge_start(*a),
                "Edge start mismatch at edge {}",
                a
            );
            assert_eq!(
                params1.edge_handler.get_edge_end(*a),
                params2.edge_handler.get_edge_end(*a),
                "Edge end mismatch at edge {}",
                a
            );
            assert_eq!(
                params1.edge_handler.get_edge_age(*a),
                params2.edge_handler.get_edge_age(*a),
                "Edge age mismatch at edge {}",
                a
            );
        }
    }

    //--------------------------------------------------------------------------------------------------

    #[test]
    pub fn delete_old_edges_t1() {
        // Setup
        let filename_input =
            "test_data/growing_neural_gas/remove_old_edges_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/remove_old_edges_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();
        // Setup
        //------------------------------------------------------------------------------
        // input

        let edge_removal_age =
            json_reader::read_val_usize(&reader_input, "config", "edge_removal_age");

        let edge_start = json_reader::read_array_usize(&reader_input, "gng_model", "edge_start");

        let edge_end = json_reader::read_array_usize(&reader_input, "gng_model", "edge_end");

        let edge_age = json_reader::read_array_usize(&reader_input, "gng_model", "edge_age");

        // input
        //------------------------------------------------------------------------------
        // output

        let target_edge_start =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");

        let target_edge_end = json_reader::read_array_usize(&reader_target, "target", "edge_end");

        let target_edge_age = json_reader::read_array_usize(&reader_target, "target", "edge_age");

        // output
        //------------------------------------------------------------------------------
        // preparation

        for a in 0..edge_start.len() {
            params
                .edge_handler
                .create_edge(edge_start[a], edge_end[a], edge_age[a]);
        }

        params.config_handler.edge_removal_age = edge_removal_age;
        // preparation
        //------------------------------------------------------------------------------

        // function call
        delete_old_edges(&mut params);
        // function call
        //------------------------------------------------------------------------------
        // validation
        let keys = params.edge_handler.get_keys();
        for a in keys {
            let start = params.edge_handler.get_edge_start(*a);
            let end = params.edge_handler.get_edge_end(*a);
            let age = params.edge_handler.get_edge_age(*a);

            assert_eq!(*start, target_edge_start[*a]);
            assert_eq!(*end, target_edge_end[*a]);
            assert_eq!(*age, target_edge_age[*a]);
        }

        // validation
        //------------------------------------------------------------------------------
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn remove_unconnected_neurons_t1() {
        let filename_input =
            "test_data/growing_neural_gas/remove_unconnected_neurons_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/remove_unconnected_neurons_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();

        let mut params: Gng_System = Gng_System::init();
        params.create_system();

        //------------------------------------------------------------------------------
        // input

        // input
        //------------------------------------------------------------------------------
        // output
        let weight_target = json_reader::read_array_f64(&reader_target, "target", "weights");

        // output
        //------------------------------------------------------------------------------
        // function call
        load_model(&mut params, filename_input);
        remove_unconnected_neurons(&mut params);
        //------------------------------------------------------------------------------
        // validation
        let keys_neuron = params.neuron_handler.get_keys();

        let input_width = params.config_handler.input_width;
        for a in keys_neuron {
            let weights = params.neuron_handler.get_weights(*a);
            for w in 0..params.config_handler.input_width {
                assert!((weights[w] - weight_target[(a * input_width) + w]).abs() < 0.0001);
            }
        }

        // validation
        //------------------------------------------------------------------------------
    }

    //--------------------------------------------------------------------------------------------------

    #[test]
    fn test_consistency_T1() {
        use csv_reader::CsvReader;
        //    use neurogas::aux::csv_reader::CsvReader;
        let input_width = 2;
        let weight_rng_min = -1.0;
        let weight_rng_max = 1.0;
        let edge_removal_age = 50;
        let neuron_creation_interval = 200;
        let max_epochs = 3;
        let max_neurons = 50;
        let target_error = 0.096;
        let epsilon_w = 0.1;
        let epsilon_n = 0.006;
        let alpha = 0.5;
        let beta = 0.995;
        let rng_seed_1: u64 = 4506;
        let rng_seed_2: u64 = 4506;

        //##########################################################
        // First run
        let mut gng_system_1 = Gng_System::init();
        gng_system_1.config_handler.set_parameters(
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
            rng_seed_1,
        );
        gng_system_1.rng_manager.seed_rng(rng_seed_1);

        let reader = CsvReader::new(
            "test_data/integration_tests/integration_t0/circles.csv",
            ',',
        );

        let mut in_set: Vec<f64> = Vec::new();
        let res = reader.read_csv_values_f64();
        match res {
            Ok(values) => in_set = values,
            Err(e) => panic!("Failed to read CSV file: {:?}", e),
        }
        gng_system_1
            .sample_handler
            .init_input_vec(&in_set, input_width);

        //##########################################################
        // Second run with identical setup
        let mut gng_system_2 = Gng_System::init();
        gng_system_2.config_handler.set_parameters(
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
            rng_seed_2,
        );
        gng_system_2.rng_manager.seed_rng(rng_seed_2);

        gng_system_2
            .sample_handler
            .init_input_vec(&in_set, input_width);

        //##########################################################
        // First fit run
        init_training(&mut gng_system_1);
        shuffle_dataset(&mut gng_system_1);
        select_sample(&mut gng_system_1);
        calc_neuron_distances(&mut gng_system_1);
        calc_nearest_neurons(&mut gng_system_1);
        calc_neuron_dependencies(&mut gng_system_1);
        increase_edge_age(&mut gng_system_1);
        add_error_to_winner_neuron(&mut gng_system_1);
        update_weights(&mut gng_system_1);
        create_edge(&mut gng_system_1);
        delete_old_edges(&mut gng_system_1);
        remove_unconnected_neurons(&mut gng_system_1);
        create_neuron(&mut gng_system_1);
        decrease_error_global(&mut gng_system_1);
        //##########################################################
        // Second fit run
        init_training(&mut gng_system_2);
        shuffle_dataset(&mut gng_system_2);
        select_sample(&mut gng_system_2);
        calc_neuron_distances(&mut gng_system_2);
        calc_nearest_neurons(&mut gng_system_2);
        calc_neuron_dependencies(&mut gng_system_2);
        increase_edge_age(&mut gng_system_2);
        add_error_to_winner_neuron(&mut gng_system_2);
        update_weights(&mut gng_system_2);
        create_edge(&mut gng_system_2);
        delete_old_edges(&mut gng_system_2);
        remove_unconnected_neurons(&mut gng_system_2);
        create_neuron(&mut gng_system_2);
        decrease_error_global(&mut gng_system_2);

        //##########################################################
        // Result Comparison
        //##########################################################
        // init_training
        let len1 = gng_system_1.neuron_handler.len();
        let len2 = gng_system_2.neuron_handler.len();
        assert_ne!(len1, 0, "Number of neurons should be greater than 0");
        assert_eq!(len1, len2, "Number of neurons should be equal");

        // Compare results for determinism
        // Check neuron weights
        let keys_1 = gng_system_1.neuron_handler.get_keys();
        let keys_2 = gng_system_2.neuron_handler.get_keys();

        let input_width = gng_system_1.config_handler.input_width;
        for &key in keys_1 {
            if let Some(weights_1) = gng_system_1
                .neuron_handler
                .get_weights(key)
                .get(0..input_width)
            {
                if let Some(weights_2) = gng_system_2
                    .neuron_handler
                    .get_weights(key)
                    .get(0..input_width)
                {
                    for i in 0..input_width {
                        assert!(
                            (weights_1[i] - weights_2[i]).abs() < 0.0001,
                            "Weights should be equal for neuron {} at position {}",
                            key,
                            i
                        );
                    }
                }
            }
        }
        // init_training
        //##########################################################
        let sample_order_1 = gng_system_1.system_handler.sample_order;
        let sample_order_2 = gng_system_2.system_handler.sample_order;
        assert_eq!(sample_order_1, sample_order_2, "Sample orders diverged");
        //##########################################################
        // weights
        for a in 0..gng_system_1.neuron_handler.get_num_neurons() {
            for b in 0..gng_system_1.config_handler.input_width {
                let weights_1 = gng_system_1.neuron_handler.get_weights(b);
                let weights_2 = gng_system_2.neuron_handler.get_weights(b);
                assert_eq!(weights_1, weights_2, "Weights diverged");
            }
        }
        // weights
        //##########################################################
        // edges
        for a in 0..gng_system_1.edge_handler.get_num_edges() {
            let edge_start_1 = gng_system_1.edge_handler.get_edge_start(a);
            let edge_end_1 = gng_system_1.edge_handler.get_edge_end(a);
            let edge_start_2 = gng_system_2.edge_handler.get_edge_start(a);
            let edge_end_2 = gng_system_2.edge_handler.get_edge_end(a);
            assert_eq!(edge_start_1, edge_start_2, "edges diverged");
            assert_eq!(edge_end_1, edge_end_2, "edges diverged");
        }
        // edges
        //##########################################################
    }
}
