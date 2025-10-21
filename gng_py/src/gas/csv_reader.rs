use csv::ReaderBuilder;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{self, BufReader};

use std::io::{BufWriter, Write};

/// A simple CSV reader.
pub struct CsvReader {
    file_path: String,
    delimiter: char,
}

impl CsvReader {
    /// Create a new `CsvReader` instance.
    pub fn new(file_path: &str, delimiter: char) -> Self {
        CsvReader {
            file_path: file_path.to_string(),
            delimiter,
        }
    }

    /// Read the CSV file and return the number of lines.
    pub fn count_lines(&self) -> Result<i32, io::Error> {
        let contents = fs::read_to_string(&self.file_path)?;
        Ok(contents.lines().count() as i32)
    }

    /// Read the CSV file and return all values.
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
    pub fn set_array_f64(&self, arr: &mut Vec<f64>, values: Vec<f64>) {
        *arr = values;
    }

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


 // let csv_reader = CsvReader::new(csv_filename, ',');
 // match csv_reader.read_values() {
//   Ok(num_lines) => println!("Number of lines: {}", num_lines),
 //     Err(err) => println!("Error reading file: {}", err),
 // }

}
