#[cfg(test)]
mod ecs_tests {

    use crate::gas::core::add_error_to_winner_neuron;
    use crate::gas::core::calc_neuron_dependencies;
    use crate::gas::core::calc_neuron_distances;
    use crate::gas::core::configure_model;
    use crate::gas::core::create_edge;
    use crate::gas::core::create_neuron;
    use crate::gas::core::delete_old_edges;
    use crate::gas::core::init_dataset;
    use crate::gas::core::init_model;
    use crate::gas::core::load_model;
    use crate::gas::core::remove_unconnected_neurons;
    use crate::gas::core::update_weights;
    use crate::gas::core::GngParams;
    use crate::gas::json_reader;

    use crate::gas::core::calc_nearest_neurons;
    #[test]
    fn test_init_dataset_t1() {
        let filename_input = "test_data/growing_neural_gas/init_dataset/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/init_dataset/target.json".to_string();
        let filename_dataset = "test_data/growing_neural_gas/init_dataset/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        //------------------------------------------------------------------------------

        let mut gng_params: GngParams = GngParams::init();

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
        let &input_width = gng_params.config_handler.get_input_width();
        init_dataset(&mut gng_params, &filename_dataset, input_width);
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

        let mut params: GngParams = GngParams::init();
        params.create_system();

        // init
        //----------------------------------------------------------
        // target
        let weights_target = json_reader::read_array_f64(&reader_target, "target", "weights");
        let edge_start_target =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let edge_end_target = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        let edge_age_target = json_reader::read_array_usize(&reader_target, "target", "edge_age");
        //    let input_width_target = json_reader::read_val_usize(&reader_target,    "target", "input_width");
        // target
        //----------------------------------------------------------
        // function call

        load_model(&mut params, filename_input);

        // function call
        //----------------------------------------------------------
        // validation
        let width = params.config_handler.get_input_width();
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
    fn test_init_model_t1() {
        let filename_input = "test_data/growing_neural_gas/init_model/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/init_model/target.json".to_string();

        let reader_input = json_reader::read_file(&filename_input).unwrap();
        let reader_target = json_reader::read_file(&filename_target).unwrap();

        //------------------------------------------------------------------------------

        let mut comps: GngParams = GngParams::init();
        comps.create_system();

        // input
        //----------------------------------------------------------
        // input

        comps
            .config_handler
            .set_input_width(json_reader::read_val_usize(
                &reader_input,
                "config",
                "input_width",
            ));
        comps
            .config_handler
            .set_weight_rng_min(json_reader::read_val_f64(
                &reader_input,
                "config",
                "weight_rng_min",
            ));
        comps
            .config_handler
            .set_weight_rng_max(json_reader::read_val_f64(
                &reader_input,
                "config",
                "weight_rng_max",
            ));

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

        let input_width = comps.config_handler.get_input_width();
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
    pub fn calc_neuron_distances_t1() {
        let filename_input =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/target.json".to_string();
        let filename_dataset =
            "test_data/growing_neural_gas/calc_neuron_distances_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();

        let mut params: GngParams = GngParams::init();
        params.create_system();

        params.system_handler.set_curr_sample_pos(1);
        // init
        //----------------------------------------------------------
        // target
        let distance_target = json_reader::read_array_f64(&reader_target, "target", "distance");
        // target
        //----------------------------------------------------------
        // function call

        load_model(&mut params, filename_input);
        let &input_width = params.config_handler.get_input_width();
        init_dataset(&mut params, &filename_dataset, input_width);
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
    fn calc_nearest_neurons_t1() {
        let filename_input =
            "test_data/growing_neural_gas/calc_nearest_neurons_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_nearest_neurons_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: GngParams = GngParams::init();
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
        assert_eq!(
            *params.system_handler.get_winner_neuron(),
            winner_neuron_target
        );

        assert_eq!(
            *params.system_handler.get_second_neuron(),
            second_neuron_target
        );
    }
    #[test]
    fn calc_distance_order_t2() {
        let filename_input =
            "test_data/growing_neural_gas/calc_nearest_neurons_t2/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_nearest_neurons_t2/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: GngParams = GngParams::init();
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
        assert_eq!(
            *params.system_handler.get_winner_neuron(),
            winner_neuron_target
        );

        assert_eq!(
            *params.system_handler.get_second_neuron(),
            second_neuron_target
        );
    }
    #[test]
    fn calc_neuron_dependencies_t1() {
        let filename_input =
            "test_data/growing_neural_gas/calc_neuron_dependencies_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/calc_neuron_dependencies_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: GngParams = GngParams::init();
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
        params.system_handler.set_winner_neuron(winner_neuron_input);
        params.system_handler.set_second_neuron(second_neuron_input);
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

        let mut params: GngParams = GngParams::init();
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
        let input_width = params.config_handler.get_input_width();
        params
            .sample_handler
            .init_data_set(&filename_dataset, *input_width);
        params.system_handler.set_curr_sample_pos(sample_pos);
        params
            .neuron_handler
            .set_errors_debug(neuron_err_input.clone());
        params.system_handler.set_winner_neuron(winner_neuron_input);
        add_error_to_winner_neuron(&mut params);

        // function call
        //------------------------------------------------------------------------------
        // validation
        for a in 0..neuron_err_target.len() {
            assert!((neuron_err_target[a] - params.neuron_handler.get_error(a)).abs() < 0.001);
        }
    }
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

        let mut params: GngParams = GngParams::init();
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
        let &input_width = params.config_handler.get_input_width();
        init_dataset(&mut params, &filename_dataset, input_width);
        params.config_handler.set_epsilon_w(epsilon_w);
        params.config_handler.set_epsilon_n(epsilon_n);
        params
            .neuron_handler
            .set_neuron_dependencies_debug(neuron_dependencies);
        params.system_handler.set_curr_sample_pos(sample_pos);
        update_weights(&mut params);

        let weights_res = params.neuron_handler.get_weight_vec(&input_width);
        for a in 0..weights_target.len() {
            assert!((weights_res[a] - weights_target[a]).abs() < 0.001);
        }
    }
    #[test]
    fn create_edge_t1() {
        let filename_input = "test_data/growing_neural_gas/create_edge_t1/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/create_edge_t1/target.json".to_string();
        //let filename_dataset = "test_data/growing_neural_gas/create_edge_t1/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: GngParams = GngParams::init();
        params.create_system();

        //------------------------------------------------------------------------------
        // input

        let neuron_dependencies =
            json_reader::read_array_usize(&reader_input, "state", "neuron_dependencies");
        //let distance_order = json_reader::read_array_usize(&reader_input, "state", "distance_order");
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
        params.system_handler.set_winner_neuron(winner_neuron_input);
        params.system_handler.set_second_neuron(second_neuron_input);
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

    #[test]
    fn create_edge_t2() {
        let filename_input = "test_data/growing_neural_gas/create_edge_t2/input.json".to_string();
        let filename_target = "test_data/growing_neural_gas/create_edge_t2/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: GngParams = GngParams::init();
        params.create_system();

        //------------------------------------------------------------------------------
        // input

        let neuron_dependencies =
            json_reader::read_array_usize(&reader_input, "state", "neuron_dependencies");
        //let distance_order = json_reader::read_array_usize(&reader_input, "state", "distance_order");
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
        params.system_handler.set_winner_neuron(winner_neuron_input);
        params.system_handler.set_second_neuron(second_neuron_input);
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

        let mut params: GngParams = GngParams::init();
        params.create_system();
        // Setup
        //------------------------------------------------------------------------------
        // input

        let target_neuron = json_reader::read_val_usize(&reader_input, "state", "curr_sample_pos");

        let neuron_err = json_reader::read_array_f64(&reader_input, "state", "neuron_err");

        let alpha = json_reader::read_val_f64(&reader_input, "config", "alpha");

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
        params.config_handler.set_alpha(alpha);
        params.system_handler.set_curr_neuron(target_neuron);
        params.neuron_handler.set_errors_debug(neuron_err);

        params
            .sample_handler
            .init_data_set(&filename_dataset, *params.config_handler.get_input_width());
        // preparation
        //------------------------------------------------------------------------------

        // function call
        create_neuron(&mut params);
        // function call
        //------------------------------------------------------------------------------

        let keys_neuron = params.neuron_handler.get_keys();
        let keys_edge = params.edge_handler.get_keys();

        let input_width = params.config_handler.get_input_width();

        for a in keys_neuron {
            assert!((params.neuron_handler.get_error(*a) - &target_neuron_err[*a]).abs() < 0.0001);
            let w_temp = params.neuron_handler.get_weights(*a);
            for w in 0..*input_width {
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

        let mut params: GngParams = GngParams::init();
        params.create_system();
        // Setup
        //------------------------------------------------------------------------------
        // input

        let target_neuron = json_reader::read_val_usize(&reader_input, "state", "curr_sample_pos");

        let neuron_err = json_reader::read_array_f64(&reader_input, "state", "neuron_err");

        let alpha = json_reader::read_val_f64(&reader_input, "config", "alpha");

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
        params.config_handler.set_alpha(alpha);
        params.system_handler.set_curr_neuron(target_neuron);
        params.neuron_handler.set_errors_debug(neuron_err);

        params
            .sample_handler
            .init_data_set(&filename_dataset, *params.config_handler.get_input_width());
        // preparation
        //------------------------------------------------------------------------------

        // function call
        create_neuron(&mut params);
        // function call
        //------------------------------------------------------------------------------

        let keys_neuron = params.neuron_handler.get_keys();
        let keys_edge = params.edge_handler.get_keys();

        let input_width = params.config_handler.get_input_width();
        //   let res_w = params.neuron_handler.get_weight_vec(input_width);

        for a in keys_neuron {
            assert!((params.neuron_handler.get_error(*a) - &target_neuron_err[*a]).abs() < 0.0001);
            let w_temp = params.neuron_handler.get_weights(*a);
            for w in 0..*input_width {
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
    pub fn delete_old_edges_t1() {
        // Setup
        let filename_input =
            "test_data/growing_neural_gas/remove_old_edges_t1/input.json".to_string();
        let filename_target =
            "test_data/growing_neural_gas/remove_old_edges_t1/target.json".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();
        let reader_input = json_reader::read_file(&filename_input).unwrap();

        let mut params: GngParams = GngParams::init();
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

        params.config_handler.set_edge_removal_age(edge_removal_age);
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
    #[test]
    fn remove_unconnected_neurons_t1() {
        let filename_input =
            "test_data/growing_neural_gas/2_CURRENT/remove_unconnected_neurons_t1/input.json"
                .to_string();
        let filename_target =
            "test_data/growing_neural_gas/2_CURRENT/remove_unconnected_neurons_t1/target.json"
                .to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();

        let mut params: GngParams = GngParams::init();
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

        let input_width = *params.config_handler.get_input_width();
        for a in keys_neuron {
            let weights = params.neuron_handler.get_weights(*a);
            for w in 0..*params.config_handler.get_input_width() {
                assert!((weights[w] - weight_target[(a * input_width) + w]).abs() < 0.0001);
            }
        }

        // validation
        //------------------------------------------------------------------------------
    }
}
