use crate::data_structures::params::GngParams;
use crate::gas::json_reader;

//use crate::gas::csv_reader::CsvReader;

use crate::gas::core::add_error_to_winner_neuron;
use crate::gas::core::calc_distance_order;
//use crate::gas::core::calc_distance_ranking;
use crate::gas::core::calc_neuron_dependencies;
use crate::gas::core::calc_neuron_distances;
use crate::gas::core::calc_winner_edges;
use crate::gas::core::create_edge;
use crate::gas::core::create_neuron;
use crate::gas::core::decrease_error_global;
use crate::gas::core::increase_edge_age;
use crate::gas::core::remove_old_edges;
//use crate::gas::core::remove_unconnected_neurons;
use crate::gas::core::shuffle_dataset;
use crate::gas::core::sub_calc_neighbor_neurons;
use crate::gas::core::calc_neuron_err_ranking;
use crate::gas::core::sub_insert_new_neuron;
use crate::gas::core::sub_remove_edge;
use crate::gas::core::update_weights;

#[cfg(test)]
mod core_tests {
    use super::*;

    #[test]
    pub fn create_neuron_t2() {
        let filename = "test_data/growing_neural_gas/create_neuron_t2.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();

        //------------------------------------------------------------------------------
        // input

        gng_params.model_params.input_width =
            json_reader::read_val_usize(&reader, "input", "input_width");
        gng_params.model_params.num_neurons =
            json_reader::read_val_usize(&reader, "input", "num_neurons");
        gng_params.train_params.alpha = json_reader::read_val_f64(&reader, "input", "alpha");

        gng_params.model_params.w = json_reader::read_array_f64(&reader, "input", "W");

        gng_params.model_params.neuron_err =
            json_reader::read_array_f64(&reader, "input", "neuron_err");
        gng_params.model_params.distance =
            json_reader::read_array_f64(&reader, "input", "distance");

        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");
        gng_params.model_params.edge_age =
            json_reader::read_array_usize(&reader, "input", "edge_age");

        gng_params.model_params.neuron_err_ranking =
            json_reader::read_array_usize(&reader, "input", "neuron_err_ranking");

        // input
        //------------------------------------------------------------------------------
        // output

        let target_w = json_reader::read_array_f64(&reader, "output", "W");
        let target_edge_start = json_reader::read_array_usize(&reader, "output", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader, "output", "edge_end");
        let target_edge_age = json_reader::read_array_usize(&reader, "output", "edge_age");
        let target_neuron_err = json_reader::read_array_f64(&reader, "output", "neuron_err");

        // output
        //------------------------------------------------------------------------------
        // function call

        create_neuron(&mut gng_params);
        // function call
        //------------------------------------------------------------------------------

        for a in 0..target_w.len() {
            assert!((target_w[a] - gng_params.model_params.w[a]).abs() < 0.0001);
        }
        for a in 0..target_edge_age.len() {
            assert_eq!(target_edge_age[a], gng_params.model_params.edge_age[a]);
        }
        for a in 0..target_edge_start.len() {
            assert_eq!(target_edge_start[a], gng_params.model_params.edge_start[a]);
        }
        for a in 0..target_edge_end.len() {
            assert_eq!(target_edge_end[a], gng_params.model_params.edge_end[a]);
        }
        for a in 0..target_neuron_err.len() {
            assert!((target_neuron_err[a] - gng_params.model_params.neuron_err[a]).abs() < 0.0001);
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn calc_neuron_distances_test() {
        let file_name = "test_data/growing_neural_gas/calc_neuron_dist_t1.json";

        let res_read = json_reader::read_file(file_name).unwrap();
        let neuron_res = json_reader::read_array_f64(&res_read, "output", "neuronDist_Target");

        //------------------------------------------------------------------------------
        let mut gng_params: GngParams = GngParams::init();

        gng_params.model_params.w = json_reader::read_array_f64(&res_read, "input", "W_in");
        gng_params.model_params.input_width =
            json_reader::read_val_usize(&res_read, "input", "inputWidth");
        gng_params.model_params.num_neurons =
            json_reader::read_val_usize(&res_read, "input", "numNeurons");
        gng_params.algo_state.sample_id_position =
            json_reader::read_val_usize(&res_read, "input", "smpPos");
        gng_params.input_set_params.sample =
            json_reader::read_array_f64(&res_read, "input", "input_In");
        //------------------------------------------------------------------------------
        calc_neuron_distances(&mut gng_params);

        let vec_size = neuron_res.len();
        assert_eq!(vec_size, gng_params.model_params.num_neurons);
        for a in 0..vec_size {
            assert!((neuron_res[a] - gng_params.model_params.distance[a]).abs() < 0.0001);
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn calc_winner_edges_test() {
        let file_name = "test_data/growing_neural_gas/calc_winner_edges_t1.json";

        let res_read = json_reader::read_file(file_name).unwrap();

        let mut gng_params: GngParams = GngParams::init();

        //------------------------------------------------------------------------------
        // input

        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&res_read, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&res_read, "input", "edge_end");
        gng_params.model_params.distance_ranking =
            json_reader::read_array_usize(&res_read, "output", "distance_ranking");
        gng_params.model_params.input_width =
            json_reader::read_val_usize(&res_read, "variables", "input_width");
        gng_params.model_params.num_neurons =
            json_reader::read_val_usize(&res_read, "variables", "num_neurons");

        // input
        //------------------------------------------------------------------------------
        // output

        let winner_edges_target =
            json_reader::read_array_usize(&res_read, "output", "winner_edges");

        // output
        //------------------------------------------------------------------------------
        // function call

        calc_winner_edges(&mut gng_params);

        // function call
        //------------------------------------------------------------------------------
        // validation
        for a in 0..winner_edges_target.len() {
            assert_eq!(
                gng_params.model_params.winner_edges[a],
                winner_edges_target[a]
            );
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    pub fn calc_neuron_dependencies_t1() {
        let filename = "test_data/growing_neural_gas/calc_neuron_dependencies_t1.json";
        let reader = json_reader::read_file(filename).unwrap();
        let mut gng_params: GngParams = GngParams::init();

        //------------------------------------------------------------------------------
        // input
        gng_params.model_params.input_width =
            json_reader::read_val_usize(&reader, "variables", "input_width");
        gng_params.model_params.num_neurons =
            json_reader::read_val_usize(&reader, "variables", "num_neurons");

        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");
        gng_params.model_params.distance_ranking =
            json_reader::read_array_usize(&reader, "output", "distance_ranking");
        gng_params.model_params.w = json_reader::read_array_f64(&reader, "input", "w");
        gng_params.model_params.winner_edges =
            json_reader::read_array_usize(&reader, "output", "winner_edges");
        // input
        //------------------------------------------------------------------------------
        // output
        let target_neuron_dependencies =
            json_reader::read_array_usize(&reader, "output", "neuron_dependencies");

        // output
        //------------------------------------------------------------------------------
        // function call

        calc_neuron_dependencies(&mut gng_params);
        // function call
        //------------------------------------------------------------------------------
        // validation
        for a in 0..target_neuron_dependencies.len() {
            assert_eq!(
                gng_params.model_params.neuron_dependencies[a],
                target_neuron_dependencies[a] as usize
            );
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn calc_distance_order_test_t1() {
        let file_name = "test_data/growing_neural_gas/calc_distance_order_t1.json";
        let res_read = json_reader::read_file(file_name).unwrap();
        let mut gng_params: GngParams = GngParams::init();

        //------------------------------------------------------------------------------
        // input
        let distance_order = json_reader::read_array_f64(&res_read, "input", "neuronDist_In");
        gng_params.model_params.distance = distance_order.clone();
        let order: Vec<usize> = vec![0; distance_order.clone().len()];
        gng_params.model_params.distance_order = order;
        // input
        //------------------------------------------------------------------------------
        // output
        let target_distance_order =
            json_reader::read_array_usize(&res_read, "output", "neuronOrder_Target");

        // output
        //------------------------------------------------------------------------------
        // function call
        calc_distance_order(&mut gng_params);
        // function call
        //------------------------------------------------------------------------------
        // validation
        for a in 0..target_distance_order.len() {
            assert_eq!(
                gng_params.model_params.distance_order[a],
                target_distance_order[a] as usize
            );
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn test_shuffle_dataset_t1() {
        //let mut input_set_params: input_set_params::params = input_set_params::params
        //       let mut Params = Params
        //       {
        //           sample_ids: vec![],
        //           sample: vec![],
        //           num_samples: 0,
        //       };
        let mut inputs: Vec<f64> = vec![0.0; 36];
        for a in 0..36 {
            inputs[a] = a as f64;
        }

        //       let mut algo_state: AlgorithmState = AlgorithmState::init();
        let mut gng_params: GngParams = GngParams::init();

        shuffle_dataset(&mut gng_params);

        for a in 0..36 {
            assert_eq!(inputs[a], a as f64);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn test_decrease_error_global_t1() {
        let filename = "test_data/growing_neural_gas/decreaseErrorGlobal_T1.json";
        let res_read = json_reader::read_file(filename).unwrap();
        let mut gng_params: GngParams = GngParams::init();

        //------------------------------------------------------------------------------
        // input
        gng_params.model_params.neuron_err =
            json_reader::read_array_f64(&res_read, "input", "neuronErrC_In");
        gng_params.train_params.d = json_reader::read_val_f64(&res_read, "input", "d");

        // input
        //------------------------------------------------------------------------------
        // output
        let target_neuron_err =
            json_reader::read_array_f64(&res_read, "output", "neuronErrC_Target");
        // output
        //------------------------------------------------------------------------------
        // function call
        decrease_error_global(&mut gng_params);
        // function call
        //------------------------------------------------------------------------------
        // validation
        // validation
        for a in 0..target_neuron_err.len() {
            assert!((gng_params.model_params.neuron_err[a] - target_neuron_err[a]).abs() < 0.001);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn increase_edge_age_t1() {
        let filename = "test_data/growing_neural_gas/increaseEdgeAge_T1.json";
        let res_read = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        //----------------------------------------
        // input

        gng_params.model_params.input_width =
            json_reader::read_val_usize(&res_read, "variables", "inputWidth");
        gng_params.model_params.num_neurons =
            json_reader::read_val_usize(&res_read, "variables", "numNeurons");
        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&res_read, "input", "edges_start_rust");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&res_read, "input", "edges_end_rust");
        gng_params.model_params.edge_age =
            json_reader::read_array_usize(&res_read, "input", "edges_age_rust");
        gng_params.model_params.distance_ranking =
            json_reader::read_array_usize(&res_read, "input", "distance_ranking");

        // input
        //----------------------------------------
        // output

        //let target_edge_start = json_reader::read_array_usize(&res_read, "output", "edges_start_target_rust");
        //let target_edge_end   = json_reader::read_array_usize(&res_read, "output", "edges_end_target_rust");
        let target_edge_age =
            json_reader::read_array_usize(&res_read, "output", "edges_age_target_rust");
        // output
        //----------------------------------------

        let mut edges: Vec<usize> = Vec::new();
        for a in 0..gng_params.model_params.edge_age.len() {
            if let Some(elem) = gng_params.model_params.edge_start.get_mut(a) {
                edges.push(*elem);
            }
        }

        //let input_set_params: Params = Params::init();

        //----------------------------------------
        // input

        //----------------------------------------
        // function call
        increase_edge_age(&mut gng_params);
        // function call
        //----------------------------------------
        for a in 0..gng_params.model_params.edge_age.len() {
            assert_eq!(gng_params.model_params.edge_age[a], target_edge_age[a]);
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn add_error_to_winner_neuron_t1() {
        let filename = "test_data/growing_neural_gas/add_error_to_winner_neuron_t1.json";
        let res_read = json_reader::read_file(filename).unwrap();
        let mut gng_params: GngParams = GngParams::init();
        //----------------------------------------
        // input

        gng_params.model_params.input_width =
            json_reader::read_val_usize(&res_read, "variables", "inputWidth");
        gng_params.model_params.w = json_reader::read_array_f64(&res_read, "input", "W_in");
        gng_params.model_params.neuron_err =
            json_reader::read_array_f64(&res_read, "input", "neuronErr_In");
        gng_params.model_params.distance =
            json_reader::read_array_f64(&res_read, "input", "neuronDist");
        gng_params.model_params.distance_order =
            json_reader::read_array_usize(&res_read, "input", "orderPos_In");

        gng_params.algo_state.sample_id_position =
            json_reader::read_val_usize(&res_read, "variables", "smpPos");

        gng_params.input_set_params.sample =
            json_reader::read_array_f64(&res_read, "input", "input_In");

        // input
        //----------------------------------------
        // output

        let neuron_err_target =
            json_reader::read_array_f64(&res_read, "output", "neuronErr_Target");

        // output
        //----------------------------------------
        // function call
        //
        add_error_to_winner_neuron(&mut gng_params);
        // function call
        //----------------------------------------
        // validation
        for a in 0..neuron_err_target.len() {
            assert!((neuron_err_target[a] - gng_params.model_params.neuron_err[a]).abs() < 0.0001);
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    pub fn create_neuron_t1() {
        let filename = "test_data/growing_neural_gas/create_neuron_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        //------------------------------------------------------------------------------
        // input

        gng_params.model_params.input_width =
            json_reader::read_val_usize(&reader, "input", "input_width");
        gng_params.model_params.num_neurons =
            json_reader::read_val_usize(&reader, "input", "num_neurons");
        gng_params.train_params.alpha = json_reader::read_val_f64(&reader, "input", "alpha");

        gng_params.model_params.w = json_reader::read_array_f64(&reader, "input", "W");

        gng_params.model_params.neuron_err =
            json_reader::read_array_f64(&reader, "input", "neuron_err");
        gng_params.model_params.distance =
            json_reader::read_array_f64(&reader, "input", "distance");

        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");
        gng_params.model_params.edge_age =
            json_reader::read_array_usize(&reader, "input", "edge_age");

        gng_params.model_params.neuron_err_ranking =
            json_reader::read_array_usize(&reader, "input", "neuron_err_ranking");

        // input
        //------------------------------------------------------------------------------
        // output

        let target_w = json_reader::read_array_f64(&reader, "output", "W");
        let target_edge_start = json_reader::read_array_usize(&reader, "output", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader, "output", "edge_end");
        let target_edge_age = json_reader::read_array_usize(&reader, "output", "edge_age");
        let target_neuron_err = json_reader::read_array_f64(&reader, "output", "neuron_err");

        // output
        //------------------------------------------------------------------------------
        // function call

        create_neuron(&mut gng_params);
        // function call
        //------------------------------------------------------------------------------

        for a in 0..target_w.len() {
            assert!((target_w[a] - gng_params.model_params.w[a]).abs() < 0.0001);
        }
        for a in 0..target_edge_age.len() {
            assert_eq!(target_edge_age[a], gng_params.model_params.edge_age[a]);
        }
        for a in 0..target_edge_start.len() {
            assert_eq!(target_edge_start[a], gng_params.model_params.edge_start[a]);
        }
        for a in 0..target_edge_end.len() {
            assert_eq!(target_edge_end[a], gng_params.model_params.edge_end[a]);
        }
        for a in 0..target_neuron_err.len() {
            assert!((target_neuron_err[a] - gng_params.model_params.neuron_err[a]).abs() < 0.0001);
        }
    }
    //--------------------------------------------------------------------------------------------------
    //#[test]
    //fn sub_calc_best_neighbor_neuron_t1() {
    //    let filename = "test_data/growing_neural_gas/sub_calc_best_neighbor_neuron_t1.json";
    //    let reader = json_reader::read_file(filename).unwrap();
    //
    //
    //        let mut gng_params: GngParams = GngParams::init();
    //    //------------------------------------------------------------------------------
    //    // input
    //
    //    gng_params.model_params.neuron_err_ranking =
    //        json_reader::read_array_usize(&reader, "input", "neuron_err_ranking");
    //    let neighbor_neur = json_reader::read_array_usize(&reader, "input", "neighbor_neurons");
    //
    //    // input
    //    //------------------------------------------------------------------------------
    //    // output
    //
    //    let target_best_neighbor_neuron = json_reader::read_val_usize(&reader, "output", "best_neighbor_neuron");
    //
    //    let target_neuron_err_ranking = json_reader::read_array_usize(&reader, "output", "neuron_err_ranking");
    //    let target_neighbor_neurons = json_reader::read_array_usize(&reader, "output", "neighbor_neurons");
    //
    //
    //    // output
    //    //------------------------------------------------------------------------------
    //    // function call
    //    let res_best_neighbor_neuron =
    //        sub_calc_best_neighbor_neuron(&mut gng_params, neighbor_neur.clone());
    //    // function call
    //    //------------------------------------------------------------------------------
    //
    //    assert_eq!(target_best_neighbor_neuron, res_best_neighbor_neuron);
    //
    //    let a = target_neighbor_neurons.clone();
    //    let b = neighbor_neur.clone();
    //
    //    let matching = a.iter().zip(&b).filter(|&(a, b)| a == b).count();
    //    assert_eq!(matching, a.len());
    //}

    //--------------------------------------------------------------------------------------------------
    //    #[test]
    //    fn sub_calc_best_neighbor_neuron_T2() {
    //        let filename = "test_data/growing_neural_gas/sub_calc_best_neighbor_neuron_t2.json";
    //        let reader = json_reader::read_file(filename).unwrap();
    //
    //        let mut gng_params: params::gng_params = params::gng_params::init();
    //
    //        gng_params.model_params.neuron_err_ranking =
    //            json_reader::read_array_usize(&reader, "input", "neuron_err_ranking");
    //        let target_best_neighbor_neuron =
    //            json_reader::read_val_usize(&reader, "output", "best_neighbor_neuron");
    //        let neighbor_neur = json_reader::read_array_usize(&reader, "input", "neighbor_neurons");
    //        //------------------------------------------------------------------------------
    //        // function call
    //        let res_best_neighbor_neuron =
    //            sub_calc_best_neighbor_neuron(&mut gng_params, neighbor_neur);
    //        // function call
    //        //------------------------------------------------------------------------------
    //
    //        assert_eq!(target_best_neighbor_neuron, res_best_neighbor_neuron);
    //    }
    //
    //    //--------------------------------------------------------------------------------------------------
    #[test]
    fn sub_calc_neighbor_neurons_t1() {
        let filename = "test_data/growing_neural_gas/sub_calc_neighbor_neurons_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        //----------------------------------------
        // input
        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");

        let target_neuron = json_reader::read_val_usize(&reader, "input", "target_neuron");
        // input
        //----------------------------------------
        // output
        let target_neighbor_neurons =
            json_reader::read_array_usize(&reader, "output", "neighbor_neurons");
        // output
        //----------------------------------------

        let mut res_neighbor_neurons = Vec::new();
        sub_calc_neighbor_neurons(&mut gng_params, &mut res_neighbor_neurons, target_neuron);

        for a in 0..target_neighbor_neurons.len() {
            assert_eq!(target_neighbor_neurons[a], res_neighbor_neurons[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn sub_calc_neighbor_neurons_t2() {
        let filename = "test_data/growing_neural_gas/sub_calc_neighbor_neurons_t2.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");

        let target_neuron = json_reader::read_val_usize(&reader, "input", "target_neuron");

        let target_neighbor_neurons =
            json_reader::read_array_usize(&reader, "output", "neighbor_neurons");

        let mut res_neighbor_neurons = Vec::new();
        sub_calc_neighbor_neurons(&mut gng_params, &mut res_neighbor_neurons, target_neuron);

        for a in 0..target_neighbor_neurons.len() {
            assert_eq!(target_neighbor_neurons[a], res_neighbor_neurons[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn sub_calc_neuron_err_ranking_t1() {
        let filename = "test_data/growing_neural_gas/sub_calc_neuron_err_ranking_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        //----------------------------------------
        // input
        let mut gng_params: GngParams = GngParams::init();
        gng_params.model_params.neuron_err =
            json_reader::read_array_f64(&reader, "input", "neuron_err");

        // input
        //----------------------------------------
        // output

        let target_neuron_err_ranking =
            json_reader::read_array_usize(&reader, "output", "neuron_err_ranking");

        let target_neuron_err = json_reader::read_array_f64(&reader, "output", "neuron_err");

        // output
        //----------------------------------------
        // function call

        calc_neuron_err_ranking(&mut gng_params);

        // function call
        //----------------------------------------

        assert_eq!(
            target_neuron_err_ranking.len(),
            gng_params.model_params.neuron_err_ranking.len()
        );
        for a in 0..target_neuron_err_ranking.len() {
            assert_eq!(
                target_neuron_err_ranking[a],
                gng_params.model_params.neuron_err_ranking[a]
            );
        }
        for a in 0..target_neuron_err.len() {
            assert_eq!(target_neuron_err[a], gng_params.model_params.neuron_err[a]);
        }
    }
    #[test]
    pub fn remove_old_edges_t1() {
        let filename = "test_data/growing_neural_gas/remove_old_edges_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        //----------------------------------------
        // input
        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");
        gng_params.model_params.edge_age =
            json_reader::read_array_usize(&reader, "input", "edge_age");
        gng_params.train_params.edge_removal_age =
            json_reader::read_val_usize(&reader, "input", "edge_removal_age");
        // input
        //----------------------------------------
        // output
        let target_edge_start = json_reader::read_array_usize(&reader, "output", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader, "output", "edge_end");
        let target_edge_age = json_reader::read_array_usize(&reader, "output", "edge_age");

        // output
        //----------------------------------------
        // function call

        remove_old_edges(&mut gng_params);

        // function call
        //----------------------------------------
        // asserts

        for a in 0..target_edge_age.len() {
            assert_eq!(target_edge_age[a], gng_params.model_params.edge_age[a]);
            assert_eq!(target_edge_start[a], gng_params.model_params.edge_start[a]);
            assert_eq!(target_edge_end[a], gng_params.model_params.edge_end[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------
 //   #[test]
 //   pub fn calc_distance_ranking_t1() {
 //       let filename = "test_data/growing_neural_gas/calc_distance_ranking_t1.json";
 //       let reader = json_reader::read_file(filename).unwrap();

 //       let mut gng_params: GngParams = GngParams::init();
 //       //------------------------------------------------------------------------------
 //       // input

 //       gng_params.model_params.distance_ranking =
 //           json_reader::read_array_usize(&reader, "input", "distance_ranking");
 //       gng_params.model_params.distance_order =
 //           json_reader::read_array_usize(&reader, "input", "distance_order");

 //       // input
 //       //------------------------------------------------------------------------------
 //       // output

 //       let target_distance_ranking =
 //           json_reader::read_array_usize(&reader, "output", "distance_ranking");

 //       // output
 //       //------------------------------------------------------------------------------
 //       // function call

 //       calc_distance_ranking(&mut gng_params);

 //       // function call
 //       //------------------------------------------------------------------------------
 //       // validation
 //       for a in 0..target_distance_ranking.len() {
 //           assert_eq!(
 //               target_distance_ranking[a],
 //               gng_params.model_params.distance_ranking[a]
 //           );
 //       }
 //   }
    //--------------------------------------------------------------------------------------------------
    #[test]
    pub fn create_edge_t1() {
        let filename = "test_data/growing_neural_gas/create_edge_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        //------------------------------------------------------------------------------
        // variables
        gng_params.model_params.neuron_dependencies =
            json_reader::read_array_usize(&reader, "input", "neuron_dependencies");
        gng_params.model_params.distance_ranking =
            json_reader::read_array_usize(&reader, "input", "distance_ranking");
        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");
        gng_params.model_params.edge_age =
            json_reader::read_array_usize(&reader, "input", "edge_age");

        // variables
        //------------------------------------------------------------------------------
        // output

        let target_edge_start = json_reader::read_array_usize(&reader, "output", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader, "output", "edge_end");
        let target_edge_age = json_reader::read_array_usize(&reader, "output", "edge_age");
        // output
        //------------------------------------------------------------------------------
        // function call
        create_edge(&mut gng_params);
        // function call
        //------------------------------------------------------------------------------

        for a in 0..target_edge_age.len() {
            assert_eq!(target_edge_age[a], gng_params.model_params.edge_age[a]);
        }
        for a in 0..target_edge_start.len() {
            assert_eq!(target_edge_start[a], gng_params.model_params.edge_start[a]);
        }
        for a in 0..target_edge_end.len() {
            assert_eq!(target_edge_end[a], gng_params.model_params.edge_end[a]);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    pub fn update_weights_t1() {
        let filename = "test_data/growing_neural_gas/update_weights_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        //------------------------------------------------------------------------------
        // input
        gng_params.model_params.input_width =
            json_reader::read_val_usize(&reader, "input", "input_width");
        gng_params.train_params.epsilon_w =
            json_reader::read_val_f64(&reader, "input", "epsilon_w");
        gng_params.train_params.epsilon_n =
            json_reader::read_val_f64(&reader, "input", "epsilon_n");
        gng_params.model_params.w = json_reader::read_array_f64(&reader, "input", "w");
        gng_params.model_params.neuron_dependencies =
            json_reader::read_array_usize(&reader, "input", "neuron_dependencies");
        gng_params.model_params.distance_ranking =
            json_reader::read_array_usize(&reader, "input", "distance_ranking");
        gng_params.algo_state.curr_sample_pos =
            json_reader::read_val_usize(&reader, "input", "smp_pos");
        gng_params.input_set_params.sample =
            json_reader::read_array_f64(&reader, "input", "input_set");
        // input
        //------------------------------------------------------------------------------
        // output

        let w_target = json_reader::read_array_f64(&reader, "output", "w_target");

        // output
        //------------------------------------------------------------------------------
        // function call

        update_weights(&mut gng_params);

        // function call
        //------------------------------------------------------------------------------
        // validation
        for a in 0..w_target.len() {
            assert!((w_target[a] - gng_params.model_params.w[a]).abs() < 0.0001);
        }
    }
    //--------------------------------------------------------------------------------------------------
    #[test]
    fn sub_remove_edge_t1() {
        let filename = "test_data/growing_neural_gas/sub_remove_edge_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();

        gng_params.model_params.edge_start =
            json_reader::read_array_usize(&reader, "input", "edge_start");
        gng_params.model_params.edge_end =
            json_reader::read_array_usize(&reader, "input", "edge_end");
        gng_params.model_params.edge_age =
            json_reader::read_array_usize(&reader, "input", "edge_age");

        let neuron_1 = json_reader::read_val_usize(&reader, "input", "neuron_1");
        let neuron_2 = json_reader::read_val_usize(&reader, "input", "neuron_2");

        let target_edge_start = json_reader::read_array_usize(&reader, "output", "edge_start");
        let target_edge_end = json_reader::read_array_usize(&reader, "output", "edge_end");
        //let target_edge_age = json_reader::read_array_usize(&reader, "output", "edge_age");
        sub_remove_edge(&mut gng_params, &neuron_1, &neuron_2);

        for a in 0..target_edge_start.len() {
            assert_eq!(target_edge_start[a], gng_params.model_params.edge_start[a]);
        }
        for a in 0..target_edge_end.len() {
            assert_eq!(target_edge_end[a], gng_params.model_params.edge_end[a]);
        }
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    fn sub_insert_new_neuron_t1() {
        let filename = "test_data/growing_neural_gas/sub_insert_new_neuron_t1.json";
        let reader = json_reader::read_file(filename).unwrap();

        let mut gng_params: GngParams = GngParams::init();
        //------------------------------------------------------------------------------
        // input
        gng_params.model_params.input_width =
            json_reader::read_val_usize(&reader, "input", "input_width");
        gng_params.model_params.w = json_reader::read_array_f64(&reader, "input", "W");
        gng_params.model_params.distance =
            json_reader::read_array_f64(&reader, "input", "distance");
        gng_params.model_params.neuron_err =
            json_reader::read_array_f64(&reader, "input", "neuron_err");
        gng_params.train_params.alpha = json_reader::read_val_f64(&reader, "input", "alpha");
        gng_params.model_params.num_neurons =
            json_reader::read_val_usize(&reader, "input", "num_neurons");

        let neuron_1 = json_reader::read_val_usize(&reader, "input", "neuron_1");
        let neuron_2 = json_reader::read_val_usize(&reader, "input", "neuron_2");
        // input
        //------------------------------------------------------------------------------
        // output

        let target_w = json_reader::read_array_f64(&reader, "output", "W");
        let target_distance = json_reader::read_array_f64(&reader, "output", "distance");
        let target_neuron_err = json_reader::read_array_f64(&reader, "output", "neuron_err");

        // output
        //------------------------------------------------------------------------------

        sub_insert_new_neuron(&mut gng_params, neuron_1, neuron_2);

        //------------------------------------------------------------------------------

        for a in 0..target_w.len() {
            assert!((target_w[a] - gng_params.model_params.w[a]).abs() < 0.0001);
        }

        for a in 0..target_distance.len() {
            assert!((target_distance[a] - gng_params.model_params.distance[a]).abs() < 0.0001);
        }

        for a in 0..target_neuron_err.len() {
            assert!((target_neuron_err[a] - gng_params.model_params.neuron_err[a]).abs() < 0.0001);
        }
    }

    //--------------------------------------------------------------------------------------------------
    //}
    //
    //
    //
    //
    //
    //

    //#[test]
    //fn test_init_dataset_t2() {
    //        //------------------------------------------------------------------------------
    //        // input
    //    const csv_filename: &str = "input_sets/inputSet.csv";
    //
    //
    //        let mut gng_params: GngParams = GngParams::init();
    //    let csv_reader = gas::csv_reader::CsvReader::new(csv_filename, ',');
    //    let res: i32 = csv_reader.count_lines().expect("Error reading file");
    //    //println!("Number of lines: {:?}", res);
    //
    //    let mut samples: Vec<f64> = Vec::new();
    //
    //    match csv_reader.read_csv_values_f64() {
    //        Ok(values) => csv_reader.set_array_f64(&mut samples, values),
    //        Err(err) => panic!("Error reading CSV values: {}", err),
    //    }
    //    let mut sample_man : Manager::EntityManager<Samples> = Manager::EntityManager::new();
    //
    //    let temp : Vec<f64> = vec![0.1,0.2];
    //    sample_man.create(Samples{position:temp});
    //
    //    //println!("input values: {:?}", samples);
    //    //let mut input_set_params: data_structures::input_set_params::params =
    //    //    data_structures::input_set_params::params::init();
    //
    //    //let mut algo_state: train_params::algorithm_state = train_params::algorithm_state::init();
    //
    //
    //   // input
    //    //------------------------------------------------------------------------------
    //    // Function call
    //    //init_dataset2(&mut gng_params);
    //    //------------------------------------------------------------------------------
    //    // validation
    //    //println!("input values final: {:?}", input_set_params.sample);
    //    assert_eq!(1, 1);
    //}
    ////--------------------------------------------------------------------------------------------------
    //
}
