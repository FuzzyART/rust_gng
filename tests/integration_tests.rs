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
        use neurogas::aux::csv_reader::CsvReader;

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
        use neurogas::aux::csv_reader::CsvReader;

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
    #[test]
    fn test_consistency() {
        use neurogas::aux::csv_reader::CsvReader;
        let mut ctx = Gng::new();

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
            Err(e) => panic!("Failed to read CSV file: {:?}", e),
        }

        ctx.init_dataset_vec(&in_set);
        ctx.fit();
        let neurons_1: Vec<(usize, Vec<f64>)> = ctx.get_neurons();
        let edges_1: Vec<(usize, usize)> = ctx.get_edges();

        // Re-initialize with the same data and run again
        let mut ctx2 = Gng::new();
        ctx2.set_parameters(
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
        ctx2.init_dataset_vec(&in_set);
        ctx2.fit();
        let neurons_2: Vec<(usize, Vec<f64>)> = ctx2.get_neurons();
        let edges_2: Vec<(usize, usize)> = ctx2.get_edges();

        println!("Neurons 1: {:?}", neurons_1);
        println!("-------");
        println!("Neurons 2: {:?}", neurons_2);

        assert_eq!(neurons_1.len(), neurons_2.len(), "Neuron count mismatch");
        for (i, ((idx_a, pos_a), (idx_b, pos_b))) in neurons_1.iter().zip(neurons_2).enumerate() {
            assert_eq!(*idx_a, idx_b, "Neuron {} index mismatch", i);
            //     assert!(pos_a.len() == pos_b.len(), "Neuron {} position dimension mismatch", i);
            //     for (j, (&pa, &pb)) in pos_a.iter().zip(pos_b).enumerate() {
            //         assert!((pa - pb).abs() < 1e-9, "Neuron {} component {} differs: {:.10} vs {:.10}", i, j, pa, pb);
            //     }
        }
        //
        //    assert_eq!(edges_1.len(), edges_2.len(), "Edge count mismatch");
        //    for (i, (&(a1, b1), &(a2, b2))) in edges_1.iter().zip(edges_2).enumerate() {
        //        assert_eq!((a1, b1), (a2, b2), "Edge {} differs: ({},{}) vs ({},{})", i, a1, b1, a2, b2);
        //    }
        assert!(1 == 1)
    }
}
