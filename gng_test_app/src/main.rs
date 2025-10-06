use gng_lib::Context;

fn main() {

    let config_file = "input.json".to_string();
    let data_file = "/tmp/circles.csv".to_string();
    let output_file = "/tmp/output.json".to_string();

    let mut ctx = Context::new();
    ctx.load_config(&config_file);
    ctx.init_dataset(&data_file);
    ctx.fit();
    ctx.save_model_json(&output_file);
}
