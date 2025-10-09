use crate::gas::json_reader;
use crate::handlers::config_handler::ConfigHandler;
use crate::handlers::sample_handler::SampleHandler;

#[cfg(test)]
mod sample_handle_tests {
    use super::*;
    #[test]
    fn test_init_dataset_t1() {
        let filename_input = "test_data/growing_neural_gas/init_dataset/input.json";
        let filename_target = "test_data/growing_neural_gas/init_dataset/target.json";
        let filename_dataset = "test_data/growing_neural_gas/init_dataset/dataset.csv".to_string();

        let target_reader = json_reader::read_file(filename_target).unwrap();

        let mut config_handler: ConfigHandler = ConfigHandler::init();
        let mut sample_handler: SampleHandler = SampleHandler::init();

        config_handler.create_config();
        config_handler.set_config_filename(filename_dataset);
        config_handler.read_input_width(filename_input);

        let input_set_filename = config_handler
            .get_config_filename()
            .expect("variable not found");
        let input_width = config_handler.get_input_width();

        // input
        //------------------------------------------------------------------------------
        // output
        let target_vec = json_reader::read_array_f64(&target_reader, "target", "samples");
        // output
        //------------------------------------------------------------------------------
        // Function call
        sample_handler.init_data_set(input_set_filename, *input_width);
        sample_handler.print_samples();
        let res_vec = sample_handler.get_samples_vec();
        // Function call
        //------------------------------------------------------------------------------
        // validation
        let len = res_vec.len();
        if len == 0 {
            assert_eq!(0, 1);
        }
        for a in 0..len {
            assert!((target_vec[a] - res_vec[a]).abs() < 0.0001);
        }

        assert_eq!(1, 1);
    }
}
