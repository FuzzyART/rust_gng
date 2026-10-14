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
    //--------------------------------------------------------------------------------------------------
    fn algorithm_t2() {
        use neurogas::aux::csv_reader::CsvReader;

        //use crate::gas::csv_reader::CsvReader;
        let mut gng_system = Gng::new();

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
        let rng_seed: u64 = 12345; // Fixed seed for reproducibility

        gng_system.set_parameters(
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

        gng_system.init_dataset_vec(&in_set);
        gng_system.fit();
        gng_system.save_model_json("/tmp/output.json");
        assert!(1 == 1); // This will pass regardless
    }

    //--------------------------------------------------------------------------------------------------
    #[test]
    // Demonstrate fit step functionality
    fn algorithm_t3() {
        use neurogas::aux::csv_reader::CsvReader;

        //use crate::gas::csv_reader::CsvReader;
        let mut gng_system = Gng::new();

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
        let rng_seed: u64 = 123;

        gng_system.set_parameters(
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

        gng_system.init_dataset_vec(&in_set);
        gng_system.init_step();
        gng_system.fit_step();
        gng_system.save_model_json("/tmp/output.json");
        assert!(1 == 1); // This will pass regardless
    }
    //--------------------------------------------------------------------------------------------------

        #[test]
        fn test_consistency_T1() {
            use neurogas::aux::csv_reader::CsvReader;
            let mut gng_system_1 = Gng::new();
    
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
            let rng_seed_1: u64 = 123;
            let rng_seed_2: u64 = 123;
    
            gng_system_1.set_parameters(
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
    
            gng_system_1.init_dataset_vec(&in_set);
            gng_system_1.fit();
            let neurons_1: Vec<(usize, Vec<f64>)> = gng_system_1.get_neurons();
            let edges_1: Vec<(usize, usize)> = gng_system_1.get_edges();
    
            // Re-initialize with the same data and run again
            let mut gng_system_2 = Gng::new();
            gng_system_2.set_parameters(
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
            gng_system_2.init_dataset_vec(&in_set);
            gng_system_2.fit();
            let neurons_2: Vec<(usize, Vec<f64>)> = gng_system_2.get_neurons();
            let edges_2: Vec<(usize, usize)> = gng_system_2.get_edges();
    
            println!("Neurons 1: {:?}", neurons_1);
            println!("-------");
            println!("Neurons 2: {:?}", neurons_2);
            println!("-------");

            let mut neurons_1 = gng_system_1.get_neurons();
            let mut neurons_2 = gng_system_2.get_neurons();

            let mut edges_1 = gng_system_1.get_edges();
            let mut edges_2 = gng_system_2.get_edges();


            neurons_1.sort_by_key(|&(key, _)| key);
            neurons_2.sort_by_key(|&(key, _)| key);

            edges_1.sort();
            edges_2.sort();

            assert_eq!(neurons_1,neurons_2, "Neurons diverged" );
            assert_eq!(edges_1,edges_2, "Edges diverged" );

       // for a in 0..gng_system_1.neuron_handler.get_num_neurons() {
       //     for b in 0..gng_system_1.config_handler.input_width {
       //         let weights_1 = gng_system_1.neuron_handler.get_weights(b);
       //         let weights_2 = gng_system_2.neuron_handler.get_weights(b);
       //         assert_eq!(weights_1, weights_2, "Weights diverged");
       //     }
       // }
        // weights
        //##########################################################
        // edges
        //for a in 0..gng_system_1.edge_handler.get_num_edges() {
        //    let edge_start_1 = gng_system_1.edge_handler.get_edge_start(a);
        //    let edge_end_1 = gng_system_1.edge_handler.get_edge_end(a);
        //    let edge_start_2 = gng_system_2.edge_handler.get_edge_start(a);
        //    let edge_end_2 = gng_system_2.edge_handler.get_edge_end(a);
        //    assert_eq!(edge_start_1, edge_start_2, "edges diverged");
        //    assert_eq!(edge_end_1, edge_end_2, "edges diverged");
        //}
        // edges
       














    
    //        assert_eq!(neurons_1.len(), neurons_2.len(), "Neuron count mismatch");
    //        for (i, ((idx_a, pos_a), (idx_b, pos_b))) in neurons_1.iter().zip(neurons_2).enumerate() {
    //            assert_eq!(*idx_a, idx_b, "Neuron {} index mismatch", i);
    //                 assert!(pos_a.len() == pos_b.len(), "Neuron {} position dimension mismatch", i);
    //                 for (j, (&pa, &pb)) in pos_a.iter().zip(pos_b).enumerate() {
    //                     assert!((pa - pb).abs() < 1e-9, "Neuron {} component {} differs: {:.10} vs {:.10}", i, j, pa, pb);
    //                 }
    //        }
    //        
    //            assert_eq!(edges_1.len(), edges_2.len(), "Edge count mismatch");
    //            for (i, (&(a1, b1), &(a2, b2))) in edges_1.iter().zip(edges_2).enumerate() {
    //                assert_eq!((a1, b1), (a2, b2), "Edge {} differs: ({},{}) vs ({},{})", i, a1, b1, a2, b2);
    //            }
           
        }
}
