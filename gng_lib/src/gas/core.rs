use crate::data_structures::{params, train_params};
use std::collections::HashSet;

use crate::handlers::{
    config_handler::ConfigHandler, edge_handler::EdgeHandler, neuron_handler::NeuronHandler,
    sample_handler::SampleHandler, system_handler::State, system_handler::SystemHandler,
};

use crate::gas::{
    json_reader, json_writer::write_json_to_file, json_writer::write_value_to_block,
    rng_manager::RngManager,
};

use rand::seq::SliceRandom;
use serde_json::{json, Value};

pub struct GngParams {
    pub neuron_handler: NeuronHandler,
    pub edge_handler: EdgeHandler,
    pub config_handler: ConfigHandler,
    pub system_handler: SystemHandler,
    pub sample_handler: SampleHandler,
    pub rng_manager: RngManager,
}
impl GngParams {
    pub fn init() -> Self {
        Self {
            neuron_handler: NeuronHandler::init(),
            edge_handler: EdgeHandler::init(),
            config_handler: ConfigHandler::init(),
            system_handler: SystemHandler::init(),
            sample_handler: SampleHandler::init(),
            rng_manager: RngManager::init(123),
        }
    }
    pub fn create_system(&mut self) {
        Self::init();
        self.config_handler.create_config();
        self.system_handler.create_system();
    }
}

//--------------------------------------------------------------------------------------------------
pub fn fit(params: &mut GngParams) {
    let mut current_state = State::Init;

    let mut counter = 0;
    while params.system_handler.get_train_completed() == false {
        counter += 1;
        match current_state {
            State::Init => {
                init_training(params);
                shuffle_dataset(params);
                params.system_handler.set_train_initiated(true);
            }
            State::StartNewIteration => {
                shuffle_dataset(params);
                let curr_epoch = params.system_handler.get_curr_epoch();
                params.system_handler.set_curr_epoch(curr_epoch + 1);
                params.system_handler.set_iteration_completed(true);
                current_state = State::TrainingCompleted;
            }
            State::TrainingCompleted => {
                end_loop(params);
            }
            State::NormalIteration => {
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
                let temp_1 = params.system_handler.get_create_neuron_scheduled();
                let temp_2 = *params.system_handler.get_sample_order_position();
                if params.system_handler.get_create_neuron_scheduled() == true {
                    create_neuron(params);
                    params.system_handler.set_create_neuron_scheduled(false);
                }
                decrease_error_global(params);
            }
            State::EpochCompleted => {
                start_new_epoch(params);
                check_stopping_criterion(params);
            }
            State::IterationCompleted => {}
        }
        params.system_handler.inc_curr_iteration();
        update_state(params, &mut current_state);
    }
}

//--------------------------------------------------------------------------------------------------
pub fn start_new_epoch(params: &mut GngParams) {
    shuffle_dataset(params);
    let curr_epoch = params.system_handler.get_curr_epoch();
    let new_epoch = curr_epoch + 1;
    params.system_handler.set_curr_epoch(new_epoch);
    params.config_handler.get_max_train_iterations();
}

//--------------------------------------------------------------------------------------------------
pub fn check_stopping_criterion(params: &mut GngParams) {}
//--------------------------------------------------------------------------------------------------
pub fn update_state(params: &mut GngParams, state: &mut State) {
    let curr_epoch: usize = params.system_handler.get_curr_epoch();
    let max_train_iterations: usize = *params.config_handler.get_max_train_iterations();

    if curr_epoch >= max_train_iterations {
        params.system_handler.set_train_completed(true);
    }
    if params.system_handler.get_train_initiated() {
        *state = State::NormalIteration;
    };
    if params.system_handler.get_last_sample_reached() {
        params.system_handler.set_last_sample_reached(false);
        *state = State::StartNewIteration;
    }
    if params.system_handler.get_curr_iteration()
        % params.config_handler.get_neuron_creation_interval()
        == 0
    {
        params.system_handler.set_create_neuron_scheduled(true);
    }
}
//--------------------------------------------------------------------------------------------------
pub fn end_loop(params: &mut GngParams) {}
//--------------------------------------------------------------------------------------------------

pub fn load_model(params: &mut GngParams, filename_model: String) {
    let reader_input = json_reader::read_file(&filename_model).unwrap();

    let weights = json_reader::read_array_f64(&reader_input, "gng_model", "weights");
    let edge_start = json_reader::read_array_usize(&reader_input, "gng_model", "edge_start");
    let edge_end = json_reader::read_array_usize(&reader_input, "gng_model", "edge_end");
    let edge_age = json_reader::read_array_usize(&reader_input, "gng_model", "edge_age");
    let input_width = json_reader::read_val_usize(&reader_input, "gng_model", "input_width");

    params.config_handler.set_input_width(input_width);
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
pub fn configure_model(filename_input: String, gng_params: &mut GngParams) {
    let reader_input = json_reader::read_file(&filename_input).unwrap();
    gng_params.config_handler.create_config();
    gng_params
        .config_handler
        .set_input_width(json_reader::read_val_usize(
            &reader_input,
            "config",
            "input_width",
        ));
    gng_params
        .config_handler
        .set_weight_rng_min(json_reader::read_val_f64(
            &reader_input,
            "config",
            "weight_rng_min",
        ));
    gng_params
        .config_handler
        .set_weight_rng_max(json_reader::read_val_f64(
            &reader_input,
            "config",
            "weight_rng_max",
        ));
}
//--------------------------------------------------------------------------------------------------

pub fn init_model(params: &mut GngParams) {
    let width = params.config_handler.get_input_width();
    let rng_min = params.config_handler.get_weight_rng_min();
    let rng_max = params.config_handler.get_weight_rng_max();
    let mut weights_n1: Vec<f64> = Vec::new();
    let mut weights_n2: Vec<f64> = Vec::new();

    for _a in 0..*width {
        weights_n1.push(params.rng_manager.get_f64(*rng_min, *rng_max));
        weights_n2.push(params.rng_manager.get_f64(*rng_min, *rng_max));
    }
    let n1: usize = params.neuron_handler.create_neuron(weights_n1);
    let n2 = params.neuron_handler.create_neuron(weights_n2);
    params.edge_handler.create_edge(n1, n2, 0);
}

//--------------------------------------------------------------------------------------------------
pub fn calc_neuron_distances(params: &mut GngParams) {
    // input: config: input_width
    //                weight
    //        system: curr_sample_pos
    //        sample: sample
    // output: neuron: distance
    //----------------------------------------
    // input
    let input_width = params.config_handler.get_input_width();

    let sample_pos = params.system_handler.get_curr_sample_pos();
    // output
    //----------------------------------------
    let input = params.sample_handler.get_sample(sample_pos);

    let mut distance_temp: Vec<(usize, f64)> = Vec::new();
    let keys = params.neuron_handler.get_keys();
    for n in keys {
        let mut dist = 0.0;
        let w = params.neuron_handler.get_weights(*n); //[n * input_width + i];
        for i in 0..*input_width {
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
pub fn calc_nearest_neurons(params: &mut GngParams) {
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
        params.system_handler.set_winner_neuron(winner);
        if let Some(second) = second_key {
            params.system_handler.set_second_neuron(second);
        } else {
            // Handle case where there's only one neuron
            params.system_handler.set_second_neuron(winner); // Or some default
        }
    } else {
        // Handle case where there are no neurons
    }
}

//--------------------------------------------------------------------------------------------------
/// calculates the neuron dependency array, which contains winner and neighbor neuron flags
/// requires: model_params.winner_edges
/// 1: winner,
/// 2: neighbor of winner
/// 0: irrelevant

pub fn calc_neuron_dependencies(params: &mut GngParams) {
    // params
    let num_neurons = params.neuron_handler.get_num_neurons();

    // input
    let winner_neuron = *params.system_handler.get_winner_neuron();
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
pub fn update_weights(params: &mut GngParams) {
    // params
    let input_width = *params.config_handler.get_input_width();
    let epsilon_w: f64 = *params.config_handler.get_epsilon_w();
    let epsilon_n: f64 = *params.config_handler.get_epsilon_n();

    // input
    let sample_pos = params.system_handler.get_curr_sample_pos();
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
pub fn get_winner_edges(params: &mut GngParams) -> Vec<usize> {
    let winner_neuron = *params.system_handler.get_winner_neuron();

    return params.edge_handler.get_connected_edges(winner_neuron);
}
//--------------------------------------------------------------------------------------------------

pub fn calc_neighbor_neuron_vec_max_err(params: &mut GngParams) {
    // input
    //  edge: edge start
    //  edge: edge end
    //  system: winner_neuron
    // output
    //  system: neighbor_neurons

    let mut res_vec: Vec<usize> = Vec::new();

    let target_neuron = *params.system_handler.get_neuron_max_err();

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
    params
        .system_handler
        .set_neighbor_neuron_vec_max_err(res_vec);
}
//--------------------------------------------------------------------------------------------------

pub fn select_sample(params: &mut GngParams) {
    let sample_order_position: usize = *params.system_handler.get_sample_order_position();
    let sample_order = params.system_handler.get_sample_order();
    let len: usize = sample_order.len();

    params
        .system_handler
        .set_curr_sample_pos(sample_order[sample_order_position]);
    params
        .system_handler
        .set_sample_order_position(sample_order_position + 1);

    if sample_order_position >= len - 1 {
        params.system_handler.set_last_sample_reached(true);
    }
}
//--------------------------------------------------------------------------------------------------
// shuffle dataset
pub fn shuffle_dataset(params: &mut GngParams) {
    let mut res: Vec<usize> = Vec::new();
    let keys = params.sample_handler.get_keys();
    for a in keys {
        res.push(*a);
    }
    let rng_new = params.rng_manager.get_rng();
    res.shuffle(rng_new);

    params.system_handler.set_sample_order(res);

    params.system_handler.set_sample_order_position(0);
}

//--------------------------------------------------------------------------------------------------

pub fn decrease_error_global(params: &mut GngParams) {
    let keys = params.neuron_handler.get_keys();
    let d = params.config_handler.get_beta();

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
pub fn increase_edge_age(params: &mut GngParams) {
    let winner_neuron = *params.system_handler.get_winner_neuron();

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
pub fn add_error_to_winner_neuron(params: &mut GngParams) {
    let sample_pos = params.system_handler.get_curr_sample_pos();
    let input_width = params.config_handler.get_input_width();

    // get winner neuron
    let winner_neuron = params.system_handler.get_winner_neuron();

    let mut sum: f64 = 0.0;

    let val1_vec = params.neuron_handler.get_weights(*winner_neuron);
    let val2_vec = params.sample_handler.get_sample(sample_pos);
    for a in 0..*input_width {
        let val1 = val1_vec[a];

        let val2 = val2_vec[a];
        sum += (val1 - val2) * (val1 - val2);
    }

    let curr_err = params.neuron_handler.get_error(*winner_neuron);
    params
        .neuron_handler
        .set_error(*winner_neuron, curr_err + sum);
}
//--------------------------------------------------------------------------------------------------
// create_neuron and sub gas
pub fn create_neuron(params: &mut GngParams) {
    //------------------------------------------------------------------------------
    // get neuron with biggest error
    calc_max_error_neuron(params);
    // get best neuron amongst neighbors of winner
    calc_neighbor_neuron_vec_max_err(params);

    calc_neighbor_neuron_max_err(params);

    let max_err_neuron: usize = *params.system_handler.get_neuron_max_err();
    let best_neighbor_neuron: usize = *params.system_handler.get_neighbor_neuron_max_err();

    remove_edge(params);

    insert_new_neuron(params);
    let neuron_1 = params.system_handler.get_neuron_max_err();
    let neuron_2 = params.system_handler.get_neighbor_neuron_max_err();
    let err_1 = params.neuron_handler.get_error(*neuron_1) * params.config_handler.get_alpha();
    let err_2 = params.neuron_handler.get_error(*neuron_2) * params.config_handler.get_alpha();

    params.neuron_handler.set_error(*neuron_1, err_1);
    params.neuron_handler.set_error(*neuron_2, err_2);

    let neuron_1 = params.system_handler.get_neuron_max_err();
    let neuron_2 = params.system_handler.get_neighbor_neuron_max_err();
    let neuron_new = params.system_handler.get_newest_neuron_id();
    params.edge_handler.create_edge(*neuron_new, *neuron_1, 0);
    params.edge_handler.create_edge(*neuron_new, *neuron_2, 0);
}
//---------------------------------------

pub fn calc_neighbor_neuron_max_err(params: &mut GngParams) {
    let neighbor_neurons = params.system_handler.get_neighbor_neuron_vec_max_err();
    let mut max_err_neighbor: f64 = *params.neuron_handler.get_error(neighbor_neurons[0]);
    let mut max_error_neighbor_pos = neighbor_neurons[0];
    for a in neighbor_neurons {
        let curr_err = *params.neuron_handler.get_error(*a);
        if max_err_neighbor < curr_err {
            max_err_neighbor = curr_err;
            max_error_neighbor_pos = *a;
        }
    }

    params
        .system_handler
        .set_neighbor_neuron_max_err(max_error_neighbor_pos);
}
pub fn calc_max_error_neuron(params: &mut GngParams) {
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
    params.system_handler.set_neuron_max_err(max_error_pos);
}

//--------------------------------------------------------------------------------------------------
pub fn delete_old_edges(params: &mut GngParams) {
    let edges = delete_old_edges_mark(params);

    for a in edges {
        params.edge_handler.remove_edge(a);
    }
}
fn delete_old_edges_mark(params: &mut GngParams) -> Vec<usize> {
    let keys = params.edge_handler.get_keys();
    let max_age = *params
        .config_handler
        .get_edge_removal_age()
        .expect("removal age not found");
    let mut marked_edges: Vec<usize> = Vec::new();
    for &a in keys {
        if *params.edge_handler.get_edge_age(a) > max_age {
            marked_edges.push(a);
        }
    }
    marked_edges
}
//--------------------------------------------------------------------------------------------------
pub fn remove_unconnected_neurons(params: &mut GngParams) {
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
        println!(
            "neuron removed num_neurons: {} curr_iteration: {}",
            params.neuron_handler.get_num_neurons(),
            params.system_handler.get_curr_iteration()
        );
        params.neuron_handler.remove_neuron(neuron_key);
    }
}

//--------------------------------------------------------------------------------------------------
pub fn create_edge(params: &mut GngParams) {
    // examine, if winner and second neuron are connected
    let winner_neuron: usize = *params.system_handler.get_winner_neuron();
    let second_neuron: usize = *params.system_handler.get_second_neuron();

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
pub fn remove_edge(params: &mut GngParams) {
    let neuron_1 = params.system_handler.get_neuron_max_err();
    let neuron_2 = params.system_handler.get_neighbor_neuron_max_err();
    let keys: Vec<usize> = params.edge_handler.get_keys().copied().collect();

    for a in keys {
        let start = params.edge_handler.get_edge_start(a);
        let end = params.edge_handler.get_edge_end(a);
        if (*start == *neuron_1 && *end == *neuron_2) || (*end == *neuron_1 && *start == *neuron_2)
        {
            params.edge_handler.remove_edge(a);
            break;
        }
    }
}
pub fn insert_new_neuron(params: &mut GngParams) {
    let neuron_1 = params.system_handler.get_neuron_max_err();
    let neuron_2 = params.system_handler.get_neighbor_neuron_max_err();
    let input_width = params.config_handler.get_input_width();

    // process weight between neurons and push them on W Vector
    let w_1 = params.neuron_handler.get_weights(*neuron_1);
    let w_2 = params.neuron_handler.get_weights(*neuron_2);
    let mut w_new: Vec<f64> = Vec::new();
    for a in 0..*input_width {
        w_new.push((w_1[a] + w_2[a]) / 2.0);
    }
    // process errors
    let err_1 = params.neuron_handler.get_error(*neuron_1) * params.config_handler.get_alpha();
    let err_2 = params.neuron_handler.get_error(*neuron_2) * params.config_handler.get_alpha();
    let err_new = (err_1 + err_2) / 2.0;

    let new_id = params.neuron_handler.create_neuron(w_new);
    params.system_handler.set_newest_neuron_id(new_id);

    params.neuron_handler.set_error(new_id, err_new);
}

pub fn init_dataset(params: &mut GngParams, filename_dataset: &String) {
    let &input_width = params.config_handler.get_input_width();
    params
        .sample_handler
        .init_data_set(&filename_dataset, input_width);
}

//--------------------------------------------------------------------------------------------------
pub fn load_config(params: &mut GngParams, filename_config: &String) {
    params.config_handler.load_config(&filename_config);
}

//--------------------------------------------------------------------------------------------------
pub fn init_training(params: &mut GngParams) {
    params.system_handler.set_train_completed(false);

    params.system_handler.set_reshuffle_required(true);

    init_model(params);

    params.system_handler.set_train_initiated(true);
}

//--------------------------------------------------------------------------------------------------
// Export Model
pub fn save_model_json(
    params: &mut GngParams,
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

// Export Model
//--------------------------------------------------------------------------------------------------
