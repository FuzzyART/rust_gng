//use crate::gas::core_ecs::configure_model_ecs;
//use crate::gas::core_ecs::fit_ecs;
//use crate::gas::core_ecs::init_dataset_ecs;
//use crate::gas::core_ecs::GngParamsECS;
//use crate::gas::json_ecs_reader;
//use crate::gas::json_reader;


//use crate::gas::core_ecs::load_config_ecs;
//use crate::gas::core_ecs::init_training_ecs;



#[cfg(test)]
mod integration_tests {

    #[test]
    pub fn algorithm_t0() {
    }
    //--------------------------------------------------------------------------------------------------
    #[cfg(feature = "stage_2")]
    #[test]
pub fn algorithm_t2() {
 
        /// load dataset, init model with 2 neurons
        let filename_input = "test_data/integration_tests/integration_t2/input.json".to_string();
        let filename_target = "test_data/integration_tests/integration_t2/target.json".to_string();
        let filename_dataset = "test_data/integration_tests/integration_t2/dataset.csv".to_string();

        let reader_target = json_reader::read_file(&filename_target).unwrap();

        let mut components: GngParamsECS = GngParamsECS::init();
        // init
        //----------------------------------------------------------
        // target
        let samples_target = json_reader::read_array_f64(&reader_target, "target", "samples");
        let num_samples_target =
            json_reader::read_val_usize(&reader_target, "target", "num_samples");


        let w_target = json_reader::read_array_f64(&reader_target, "target", "w");
        let edge_start_target =
            json_reader::read_array_usize(&reader_target, "target", "edge_start");
        let edge_end_target = json_reader::read_array_usize(&reader_target, "target", "edge_end");
        // target
        //----------------------------------------------------------
        // function call

        configure_model_ecs(filename_input, &mut components);
        init_dataset_ecs(&filename_dataset, &mut components);
        fit_ecs(&mut components);

        // function call
        //----------------------------------------------------------
        // validation
        let width = components.config_handler.get_input_width();

        let samples_res = components.sample_handler.get_samples_vec();
        let num_samples_res = components.sample_handler.get_num_samples();
        let samples_res = components.sample_handler.get_samples_vec();
        let num_samples_res = components.sample_handler.get_num_samples();

        let weights = components.neuron_handler.get_weight_vec(&width);
        let edges_start = components.edge_handler.get_edge_start_vec();
        let edges_end = components.edge_handler.get_edge_end_vec();
        println!("{:?}",components.neuron_handler.get_weight_vec(width));
        println!("{:?}",components.sample_handler.get_samples_vec());
        assert_eq!(num_samples_res, num_samples_target);

        for a in 0..samples_res.len() {
            assert!((samples_res[a] - samples_target[a]).abs() < 0.0001);
        }
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
}
