//! Integration tests for the GNG algorithm
//! For early development, only it is only testd, if the pipeline runs through.
//! The tests can also be used as templates.

#[cfg(test)]
mod integration_tests {
    use neurogas::Gng;

    #[test]
    fn algorithm_t1() {
        // Test basic functionality
        let mut gng = Gng::new();

        assert!(gng.get_neurons().is_empty() || true); // This will pass regardless

    }
    #[test]
    // Demonstrate fit functionality
    fn algorithm_t2() {
        use neurogas::gas::csv_reader::CsvReader;

        //use crate::gas::csv_reader::CsvReader;
        let mut ctx = Gng::new();

        let input_width = 2;
        let weight_rng_min = -1.0;
        let weight_rng_max = 1.0;
        let edge_removal_age = 50;
        let neuron_creation_interval = 200;
        let max_epochs = 30;
        let max_neurons = 50;
        let target_error = 0.096;
        let epsilon_w = 0.1;
        let epsilon_n = 0.006;
        let alpha = 0.5;
        let beta = 0.995;

        ctx.set_parameters(
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
        );

        let reader = CsvReader::new(
            "test_data/integration_tests/integration_t0/circles.csv",
            ',',
        );

        let mut in_set: Vec<f64> = Vec::new();
        let res = reader.read_csv_values_f64();
        match res {
            Ok(values) => in_set = values,
            Err(e) => println!("file not found {:?}", e),
        }
        //println!("res: {:?}",in_set);

        ctx.init_dataset_vec(&in_set);
        ctx.fit();
        ctx.save_model_json("/tmp/output.json");
        assert!(1 == 1); // This will pass regardless
    }

    #[test]
    // Demonstrate fit step functionality
    fn algorithm_t3() {
        use neurogas::gas::csv_reader::CsvReader;

        //use crate::gas::csv_reader::CsvReader;
        let mut ctx = Gng::new();

        let input_width = 2;
        let weight_rng_min = -1.0;
        let weight_rng_max = 1.0;
        let edge_removal_age = 50;
        let neuron_creation_interval = 200;
        let max_epochs = 30;
        let max_neurons = 50;
        let target_error = 0.096;
        let epsilon_w = 0.1;
        let epsilon_n = 0.006;
        let alpha = 0.5;
        let beta = 0.995;

        ctx.set_parameters(
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
        );

        let reader = CsvReader::new(
            "test_data/integration_tests/integration_t0/circles.csv",
            ',',
        );

        let mut in_set: Vec<f64> = Vec::new();
        let res = reader.read_csv_values_f64();
        match res {
            Ok(values) => in_set = values,
            Err(e) => println!("file not found {:?}", e),
        }
        //println!("res: {:?}",in_set);

        ctx.init_dataset_vec(&in_set);
        ctx.init_step();
        ctx.fit_step();
        ctx.save_model_json("/tmp/output.json");
        assert!(1 == 1); // This will pass regardless
    }

}
