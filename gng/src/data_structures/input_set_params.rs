pub struct Params {
    pub sample_ids: Vec<usize>,
    pub sample: Vec<f64>,
    pub num_samples: usize,
}
impl Params {
    pub fn init() -> Self {
        Self {
            sample_ids: Vec::new(),
            sample: Vec::new(),
            num_samples: 0,
        }
    }
}
