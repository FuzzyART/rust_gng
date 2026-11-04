//! A module for handling sample data with entity management.
//!
//! The [`SampleHandler`] struct provides functionality to initialize and manage
//! sample data using an [`EntityManager`]. It supports loading samples from CSV files
//! or input vectors, retrieving specific samples, and debugging output.

use crate::aux::csv_reader::CsvReader;
use crate::ecs::manager;

//============================================================
/// Represents a sample with position data.
///
/// The `Samples` struct holds a vector of f64 values representing the position
/// of a sample. This is used by the [`SampleHandler`] to store and manage samples.
#[derive(Debug)]
pub struct Samples {
    /// Position data for the sample.
    pub position: Vec<f64>,
}

impl Samples {
    /// Creates a new `Samples` instance with the given position data.
    ///
    /// # Arguments
    ///
    /// * `pos` - A vector of f64 values representing the position of the sample.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::Samples;
    ///
    /// let samples = Samples::init(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(samples.position, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn init(pos: Vec<f64>) -> Self {
        Self { position: pos }
    }
}

//============================================================
/// Handles sample data using an entity manager.
///
/// The `SampleHandler` struct manages a collection of samples through an
/// [`EntityManager`] and provides methods to initialize data from CSV files or
/// input vectors, retrieve specific samples, and debug output.
///
/// # Examples
///
/// ```ignore
/// use gng::components::sample_component::SampleHandler;
///
/// let mut handler = SampleHandler::init();
/// // Initialize with sample data...
/// ```
#[derive(Debug)]
pub struct SampleHandler {
    /// Entity manager for storing samples.
    pub sample_man: manager::EntityManager<Samples>,
    /// Number of samples in the handler.
    pub num_samples: usize,
    /// Width of each sample (number of elements per sample).
    pub sample_width: usize,
}

//--------------------------------------------------
// Implementations
//--------------------------------------------------
impl SampleHandler {
    /// Creates a new, empty `SampleHandler`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let handler = SampleHandler::init();
    /// assert_eq!(handler.num_samples, 0);
    /// ```
    pub fn init() -> Self {
        Self {
            sample_man: manager::EntityManager::new(),
            num_samples: 0,
            sample_width: 0,
        }
    }

    /// Initializes the sample handler with data from a CSV file.
    ///
    /// This method reads a CSV file and creates samples based on the data.
    /// The number of samples is calculated as `data_length / sample_width`.
    ///
    /// # Arguments
    ///
    /// * `input_set_filename` - Path to the CSV file containing the sample data.
    /// * `sample_width` - Number of elements in each sample.
    ///
    /// # Panics
    ///
    /// This method will panic if there is an error reading the CSV file.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let mut handler = SampleHandler::init();
    /// // handler.init_data_set("data.csv", 3); // Will panic on file error
    /// ```
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

    /// Initializes the sample handler with data from an input vector.
    ///
    /// This method creates samples from an existing vector of f64 values.
    /// The number of samples is calculated as `data_length / sample_width`.
    ///
    /// # Arguments
    ///
    /// * `samples` - A vector of f64 values representing all sample data.
    /// * `sample_width` - Number of elements in each sample.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let mut handler = SampleHandler::init();
    /// let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// handler.init_input_vec(&sample_data, 3);
    /// assert_eq!(handler.num_samples, 2);
    /// ```
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

    /// Retrieves the position data of a specific sample.
    ///
    /// # Arguments
    ///
    /// * `num_sample` - The index of the sample to retrieve.
    ///
    /// # Returns
    ///
    /// Returns a reference to the position vector of the specified sample.
    ///
    /// # Panics
    ///
    /// This method will panic if the requested sample does not exist.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let mut handler = SampleHandler::init();
    /// // Initialize with data...
    /// // let position = handler.get_sample(0); // Returns reference to sample 0's position
    /// ```
    pub fn get_sample(&self, num_sample: usize) -> &Vec<f64> {
        return self
            .sample_man
            .get(num_sample)
            .map(|val| &val.position)
            .expect("sample not found");
    }

    /// Prints all samples to the console for debugging.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let mut handler = SampleHandler::init();
    /// // Initialize with data...
    /// // handler.print_samples(); // Prints all samples to stdout
    /// ```
    pub fn print_samples(&self) {
        println!("num samples: {:?}", self.num_samples);
        for (id, b) in self.sample_man.iter() {
            println!("id: {:?} pos: {:?}", id, b.position);
        }
    }

    /// Converts all samples into a flat vector of f64 values.
    ///
    /// # Returns
    ///
    /// Returns a vector containing all sample data in a flattened format.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let mut handler = SampleHandler::init();
    /// // Initialize with data...
    /// // let flat_data = handler.get_samples_vec(); // Returns flattened sample data
    /// ```
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

    /// Returns the number of samples currently stored.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let mut handler = SampleHandler::init();
    /// // Initialize with data...
    /// // let count = handler.get_num_samples(); // Returns number of samples
    /// ```
    pub fn get_num_samples(&self) -> usize {
        self.sample_man.len()
    }

    /// Returns an iterator over the keys (IDs) of all samples.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::components::sample_component::SampleHandler;
    ///
    /// let mut handler = SampleHandler::init();
    /// // Initialize with data...
    /// // for key in handler.get_keys() {
    /// //     println!("Sample ID: {}", key);
    /// // }
    /// ```
    pub fn get_keys(&self) -> impl Iterator<Item = &usize> {
        self.sample_man.keys()
    }
}
