use::gng::Gng;

fn main() {
    const CONFIG_FILE: &str = "../config.json";
    const DATA_FILE: &str = "../blobs.csv";
    const OUTPUT_FILE: &str = "/tmp/output.json";
    let mut gng = Gng::new();

    gng.set_parameters(
        2,      // input_width
        -1.0,   // weight_rng_min
        1.0,    // weight_rng_max
        50,     // edge_removal_age
        200,    // neuron_creation_interval
        10000,  // max_train_iterations
        0.096,  // target_error
        0.1,    // epsilon_w
        0.006,  // epsilon_n
        0.5,    // alpha
        0.995,  // beta
    );

    gng.init_dataset(DATA_FILE);
    gng.fit();
    gng.save_model_json(OUTPUT_FILE);
}

