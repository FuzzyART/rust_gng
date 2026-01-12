use::neurogas::Gng;

fn main() {
    const CONFIG_FILE: &str = "../config.json";
    const DATA_FILE: &str = "../blobs.csv";
    const OUTPUT_FILE: &str = "/tmp/output.json";
    let mut gng = Gng::new();
    gng.load_config(CONFIG_FILE);
    gng.init_dataset(DATA_FILE);
    gng.fit();
    gng.save_model_json(OUTPUT_FILE);
}