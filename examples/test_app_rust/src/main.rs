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


use clap::Parser;
use gng_py::Context;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(long = "config", short = 'c', default_value = "../config.json")]
    config_file: String,

    /// Data file path
    #[arg(long = "data", short = 'd', default_value = "/tmp/circles.csv")]
    data_file: String,

    /// Output file path
    #[arg(long = "output", short = 'o', default_value = "/tmp/output.json")]
    output_file: String,
}

fn main() {
    let args = Args::parse();

    let mut ctx = Context::new();
    ctx.load_config(&args.config_file);
    ctx.init_dataset(&args.data_file);
    ctx.fit();
    ctx.save_model_json(&args.output_file);
}

