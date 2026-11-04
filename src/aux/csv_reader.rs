use csv::ReaderBuilder;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{self, BufReader};

use std::io::{BufWriter, Write};

/// A simple CSV reader.
///
/// This struct provides functionality to read and process CSV files,
/// including counting lines, reading values as strings or f64, and
/// saving vectors to a new CSV file.
pub struct CsvReader {
    file_path: String,
    delimiter: char,
}

impl CsvReader {
    /// Create a new `CsvReader` instance.
    ///
    /// # Arguments
    ///
    /// * `file_path` - A string slice that holds the path to the CSV file.
    /// * `delimiter` - The character used to separate fields in the CSV file.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::csv_reader::CsvReader;
    /// let csv_reader = CsvReader::new("data.csv", ',');
    /// ```
    pub fn new(file_path: &str, delimiter: char) -> Self {
        CsvReader {
            file_path: file_path.to_string(),
            delimiter,
        }
    }

    /// Read the CSV file and return the number of lines.
    ///
    /// # Returns
    ///
    /// * `Result<i32, io::Error>` - The number of lines in the CSV file on success,
    ///   or an I/O error if the file cannot be read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::csv_reader::CsvReader;
    /// let csv_reader = CsvReader::new("data.csv", ',');
    /// let line_count = csv_reader.count_lines().unwrap();
    /// ```
    pub fn count_lines(&self) -> Result<i32, io::Error> {
        let contents = fs::read_to_string(&self.file_path)?;
        Ok(contents.lines().count() as i32)
    }

    /// Read the CSV file and return all values as strings.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<String>, Box<dyn Error>>` - A vector of all string values from the CSV file on success,
    ///   or an error if the file cannot be read or parsed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::csv_reader::CsvReader;
    /// let csv_reader = CsvReader::new("data.csv", ',');
    /// let values = csv_reader.read_csv_values().unwrap();
    /// ```
    pub fn read_csv_values(&self) -> Result<Vec<String>, Box<dyn Error>> {
        // Open the CSV file.
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);

        // Create a CSV reader.
        let mut rdr = ReaderBuilder::new()
            .delimiter(self.delimiter as u8)
            .from_reader(reader);

        // Vector to store all values from the CSV file.
        let mut all_values = Vec::new();

        for result in rdr.records() {
            // The iterator yields Result<StringRecord, Error>, so we check for errors
            // and unwrap the StringRecord.
            let record = result?;

            // Iterate over each field in the record and push it to the vector.
            for value in &record {
                all_values.push(value.to_string());
            }
        }

        Ok(all_values)
    }

    /// Set a mutable reference to an f64 vector with new values.
    ///
    /// This function replaces the contents of the target vector with the provided values.
    /// It's primarily used internally by other methods for managing data.
    ///
    /// # Arguments
    ///
    /// * `arr` - A mutable reference to the target vector that will be updated.
    /// * `values` - The new values to set in the vector.
    pub fn set_array_f64(&self, arr: &mut Vec<f64>, values: Vec<f64>) {
        *arr = values;
    }

    /// Read the CSV file and return all values as f64.
    ///
    /// This method attempts to parse all fields in the CSV file as floating-point numbers.
    /// If any field cannot be parsed, an error is returned.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<f64>, Box<dyn Error>>` - A vector of all f64 values from the CSV file on success,
    ///   or an error if the file cannot be read or a value cannot be parsed as f64.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The file cannot be opened or read.
    /// * Any field in the CSV file cannot be parsed as a floating-point number.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::csv_reader::CsvReader;
    /// let csv_reader = CsvReader::new("data.csv", ',');
    /// let values = csv_reader.read_csv_values_f64().unwrap();
    /// ```
    pub fn read_csv_values_f64(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        // Open the CSV file.
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);

        // Create a CSV reader with the specified delimiter.
        let mut rdr = ReaderBuilder::new()
            .delimiter(self.delimiter as u8)
            .from_reader(reader);

        // Vector to store all values from the CSV file.
        let mut all_values = Vec::new();
        for result in rdr.records() {
            // The iterator yields Result<StringRecord, Error>, so we check for errors
            // and unwrap the StringRecord.
            let record = result?;

            // Iterate over each field in the record. We use a loop to convert the strings to f64.
            for value in &record {
                match value.parse::<f64>() {
                    Ok(float_value) => all_values.push(float_value),
                    Err(e) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Could not parse value '{}' as a float: {}", value, e),
                        )
                        .into())
                    } // Convert io::Error to Box<dyn Error>
                }
            }
        }

        Ok(all_values)
    }

    /// Save two vectors of f64 values to a new CSV file.
    ///
    /// This function writes two vectors as columns in a new CSV file with headers "x" and "y".
    ///
    /// # Arguments
    ///
    /// * `vec1` - The first vector of f64 values (will be written as the "x" column).
    /// * `vec2` - The second vector of f64 values (will be written as the "y" column).
    /// * `file_path` - A string slice that holds the path to the output CSV file.
    ///
    /// # Returns
    ///
    /// * `Result<(), std::io::Error>` - An empty result on success, or an I/O error if the file cannot be created or written.
    ///
    /// # Panics
    ///
    /// This function will panic if the vectors have different lengths, as it uses `zip`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_gng::aux::csv_reader::CsvReader;
    /// let csv_reader = CsvReader::new("data.csv", ',');
    /// let vec1 = vec![1.0, 2.0, 3.0];
    /// let vec2 = vec![4.0, 5.0, 6.0];
    /// csv_reader.save_vecs_to_csv(vec1, vec2, "output.csv").unwrap();
    /// ```
    pub fn save_vecs_to_csv(
        &self,
        vec1: Vec<f64>,
        vec2: Vec<f64>,
        file_path: &str,
    ) -> Result<(), std::io::Error> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "x,y")?;
        for (a, b) in vec1.iter().zip(vec2.iter()) {
            writeln!(writer, "{},{}", a, b)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CSV_FILENAME: &str = "test_data/test_csv_values.csv";

    #[test]
    fn test_count_lines() {
        let csv_reader = CsvReader::new(CSV_FILENAME, ',');
        match csv_reader.count_lines() {
            Ok(num_lines) => assert!(num_lines > 0), // Assuming the file is not empty
            Err(err) => panic!("Error reading file: {}", err),
        }
    }

    #[test]
    fn test_read_csv_values() {
        let csv_reader = CsvReader::new(CSV_FILENAME, ',');
        match csv_reader.read_csv_values() {
            Ok(values) => assert!(!values.is_empty()), // Assuming the file is not empty
            Err(err) => panic!("Error reading CSV values: {}", err),
        }
    }
}
