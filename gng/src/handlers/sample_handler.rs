#[allow(unused_imports)]
use crate::handlers::config_handler::ConfigHandler;

use crate::gas::csv_reader::CsvReader;
#[allow(unused_imports)]
use crate::gas::json_reader;

use crate::ecs::manager;

//==================================================================================================

#[derive(Debug)]
pub struct Samples {
    pub position: Vec<f64>,
}
impl Samples {
    pub fn init(pos: Vec<f64>) -> Self {
        Self { position: pos }
    }
}
//==================================================================================================

pub struct SampleHandler {
    pub sample_man: manager::EntityManager<Samples>,
    pub num_samples: usize,
    pub sample_width: usize,
}

//--------------------------------------------------
// Implementations
//--------------------------------------------------
impl SampleHandler {
    pub fn init() -> Self {
        Self {
            sample_man: manager::EntityManager::new(),
            num_samples: 0,
            sample_width: 0,
        }
    }
    //--------------------------------------------------

    pub fn init_data_set(&mut self, input_set_filename: &String, sample_width: usize) {
        self.sample_width = sample_width;
        let csv_reader = CsvReader::new(&input_set_filename, ',');

        let mut samples: Vec<f64> = Vec::new();

        match csv_reader.read_csv_values_f64() {
            Ok(values) => csv_reader.set_array_f64(&mut samples, values),
            Err(err) => panic!("Error reading CSV values: {}", err),
        }

        self.num_samples = samples.len() / sample_width;

        for a in 0..self.num_samples {
            let mut line: Vec<f64> = Vec::new();
            for i in 0..sample_width {
                line.push(samples[(sample_width * a) + i]);
            }
            self.sample_man.create(Samples { position: line });
        }
    }
    //--------------------------------------------------
    pub fn init_input_vec(&mut self, samples: &Vec<f64>, sample_width: usize) {
        self.sample_width = sample_width;

        self.num_samples = samples.len() / sample_width;

        for a in 0..self.num_samples {
            let mut line: Vec<f64> = Vec::new();
            for i in 0..sample_width {
                line.push(samples[(sample_width * a) + i]);
            }
            self.sample_man.create(Samples { position: line });
        }
    }
    //--------------------------------------------------
    pub fn get_sample(&self, num_sample: usize) -> &Vec<f64> {
        return self
            .sample_man
            .get(num_sample)
            .map(|val| &val.position)
            .expect("sample not found");
    }
    //==================================================================================================
    // Debug functions
    pub fn print_samples(&self) {
        println!("num samples: {:?}", self.num_samples);
        for (id, b) in self.sample_man.iter() {
            println!("id: {:?} pos: {:?}", id, b.position);
        }
    }
    pub fn get_samples_vec(&self) -> Vec<f64> {
        let mut res_vec: Vec<f64> = vec![0.0; self.num_samples * self.sample_width];

        println!("num samples: {:?}", self.num_samples);
        for (id, line) in self.sample_man.iter() {
            for a in 0..self.sample_width {
                res_vec[(*id * self.sample_width) + a] = line.position[a];
            }
        }
        return res_vec;
    }
    pub fn get_num_samples(&self) -> usize {
        self.sample_man.len()
    }
    pub fn get_keys(&self) -> impl Iterator<Item = &usize> {
        self.sample_man.keys()
    }
}
