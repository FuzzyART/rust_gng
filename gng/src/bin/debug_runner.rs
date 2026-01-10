//use gng_lib::Context;
//
//fn main() {
//
//    let config_file = "input.json".to_string();
//    let data_file = "/tmp/circles.csv".to_string();
//    let output_file = "/tmp/output.json".to_string();
//
//    let mut ctx = Context::new();
//    ctx.load_config(&config_file);
//    ctx.init_dataset(&data_file);
//    ctx.fit();
//    ctx.save_model_json(&output_file);
//}


//use clap::Parser;
use gng::gas::csv_reader::CsvReader;
use gng::Gng;
//
//#[derive(Parser, Debug)]
//#[command(author, version, about, long_about = None)]
//struct Args {
//    /// Configuration file path
//    #[arg(long = "config", short = 'c', default_value = "../config.json")]
//    config_file: String,
//
//    /// Data file path
//    #[arg(long = "data", short = 'd', default_value = "/tmp/circles.csv")]
//    data_file: String,
//
//    /// Output file path
//    #[arg(long = "output", short = 'o', default_value = "/tmp/output.json")]
//    output_file: String,
//}

fn main() {
        let mut ctx = Gng::new();
//        ctx.create_system();

        let input_width = 2;
        let weight_rng_min = -1.0;
        let weight_rng_max = 1.0;
        let edge_removal_age = 50;
        let neuron_creation_interval = 200;
        let max_train_iterations =  10000;
        let target_error = 0.096;
        let epsilon_w = 0.1;
        let epsilon_n = 0.006;
        let alpha =  0.5;
        let beta = 0.995;

        

        ctx.set_parameters(
            input_width,
            weight_rng_min,
            weight_rng_max,
            edge_removal_age,
            neuron_creation_interval,
            max_train_iterations,
            target_error,
            epsilon_w,
            epsilon_n,
            alpha,
            beta,
        );
 //      ctx.set_input_width(input_width);
        
        let reader = CsvReader::new("test_data/debug_dataset.csv",',');

        //let in_set:Vec<f64> = reader.read_csv_values_f64().expect("file not found");
        let mut in_set:Vec<f64> = Vec::new();
        let res = reader.read_csv_values_f64();
        match res{
            Ok(values) => in_set = values,
            Err(e) => println!("file not found {:?}",e),

        }
        //println!("res: {:?}",in_set);


    //    ctx.load_config(&args.config_file);
        ctx.init_dataset_vec(&in_set);
        ctx.fit();
    //    ctx.save_model_json(&args.output_file);
}

