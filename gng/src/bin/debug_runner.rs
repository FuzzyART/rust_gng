use neurogas::gas::csv_reader::CsvReader;
use neurogas::Gng;

fn main() {
    let mut ctx = Gng::new();

    let input_width = 2;
    let weight_rng_min = -1.0;
    let weight_rng_max = 1.0;
    let edge_removal_age = 50;
    let neuron_creation_interval = 100;
    let max_epochs = 50;
    let max_neurons = 80;
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

    let reader = CsvReader::new("test_data/debug_dataset.csv", ',');

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
}
