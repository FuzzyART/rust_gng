use serde_json::{self, Value};
use std::error::Error;
use std::fs;

use serde::{Deserialize, Serialize};

/// A cell containing a floating-point value.
#[derive(Serialize, Deserialize, Debug)]
pub struct Cell {
    /// The floating-point value stored in the cell.
    pub value: f64,
}

/// A table composed of a vector of cells.
#[derive(Serialize, Deserialize, Debug)]
pub struct Table {
    /// Vector of cells that make up the table.
    pub cells: Vec<Cell>,
}
//--------------------------------------------------------------------------------------------------
// public

/// Reads a two-dimensional array of `f64` values from a JSON structure.
///
/// # Arguments
///
/// * `content` - A reference to the root JSON value containing the data.
/// * `x` - The first key used to access the nested structure.
/// * `y` - The second key used to access the nested structure.
///
/// # Returns
///
/// A vector of `f64` values extracted from the specified location in the JSON.
pub fn read_array_f64(content: &Value, x: &str, y: &str) -> Vec<f64> {
    let input = &content[x][y];
    return _read_array_f64(&input);
}
/// Reads a JSON file and parses it into a `serde_json::Value`.
///
/// # Arguments
///
/// * `file_name` - The path to the JSON file to read.
///
/// # Returns
///
/// A `Result<Value, String>` representing either the parsed JSON or an error message.

pub fn read_file(file_name: &str) -> Result<Value, String> {
    let file_content = fs::read_to_string(file_name).expect("file not found");
    //let json_content = serde_json::from_str(&file_content).expect("json file corrupted");
    let json_content: Value = serde_json::from_str(&file_content).expect("json read fault 1");
    return Ok(json_content);
}
/// Reads a two-dimensional array of `usize` values from a JSON structure.
///
/// # Arguments
///
/// * `content` - A reference to the root JSON value containing the data.
/// * `x` - The first key used to access the nested structure.
/// * `y` - The second key used to access the nested structure.
///
/// # Returns
///
/// A vector of `usize` values extracted from the specified location in the JSON.
pub fn read_array_usize(content: &Value, x: &str, y: &str) -> Vec<usize> {
    let input = &content[x][y];
    return _read_array_usize(&input);
}
/// Reads a single `usize` value from a JSON structure.
///
/// # Arguments
///
/// * `content` - A reference to the root JSON value containing the data.
/// * `x` - The first key used to access the nested structure.
/// * `y` - The second key used to access the nested structure.
///
/// # Returns
///
/// A single `usize` value extracted from the specified location in the JSON.
pub fn read_val_usize(content: &Value, x: &str, y: &str) -> usize {
    let input = &content[x][y];
    return _read_val_usize(&input);
}
/// Reads a single `f64` value from a JSON structure.
///
/// # Arguments
///
/// * `content` - A reference to the root JSON value containing the data.
/// * `x` - The first key used to access the nested structure.
/// * `y` - The second key used to access the nested structure.
///
/// # Returns
///
/// A single `f64` value extracted from the specified location in the JSON.
pub fn read_val_f64(content: &Value, x: &str, y: &str) -> f64 {
    let input = &content[x][y];
    return _read_val_f64(&input);
}
/// Reads a single `u64` value from a JSON structure.
///
/// # Arguments
///
/// * `content` - A reference to the root JSON value containing the data.
/// * `x` - The first key used to access the nested structure.
/// * `y` - The second key used to access the nested structure.
///
/// # Returns
///
/// A single `u64` value extracted from the specified location in the JSON.
pub fn read_val_u64(content: &Value, x: &str, y: &str) -> u64 {
    let input = &content[x][y];
    return _read_val_u64(&input);
}
/// Reads a JSON file and parses it into a `serde_json::Value`, with panic handling.
///
/// # Arguments
///
/// * `file_name` - The path to the JSON file to read.
///
/// # Returns
///
/// A `Result<Value, String>` representing either the parsed JSON or an error message.

pub fn read_file_temp(file_name: &str) -> Result<Value, String> {
    let file_content = fs::read_to_string(file_name).expect("file not found");
    let json_content: Value = serde_json::from_str(&file_content).expect("json read fault 1");
    return Ok(json_content);
}
/// Reads a string value from a nested JSON structure using a sequence of keys.
///
/// # Arguments
///
/// * `res` - A reference to the root JSON value containing the data.
/// * `keys` - A slice of string keys used to navigate through the nested structure.
///
/// # Returns
///
/// A `Result<String, Box<dyn Error>>` representing either the string value or an error.

pub fn read_val_str(res: &Value, keys: &[&str]) -> Result<String, Box<dyn Error>> {
    let mut current = res;

    for key in keys {
        current = match current.get(key) {
            Some(v) => v,
            None => return Err(format!("Key not found: {}", key).into()),
        };
    }

    // Convert the final value to a String
    Ok(serde_json::to_string(current)?)
}

// public
//--------------------------------------------------------------------------------------------------
/// Internal helper function to read a two-dimensional array of `f64` values.
///
/// # Arguments
///
/// * `value` - A reference to a JSON value representing an array.
///
/// # Returns
///
/// A vector of `f64` values extracted from the nested arrays.
fn _read_array_f64(value: &Value) -> Vec<f64> {
    let mut res_arr: Vec<f64> = vec![];
    match value {
        Value::Array(array) => {
            let si_y = &array.len();
            for y in 0..*si_y {
                let si_x = &array[y].as_array().expect("Error Array X read").len();
                for x in 0..*si_x {
                    let val = &array[y][x].as_f64();
                    match val {
                        Some(val) => {
                            res_arr.push(*val);
                        }
                        None => println!("Array read fault 1 {:?}", array[y][x]),
                    }
                }
            }
        }
        _ => println!("array read fault 2 {:?}", value),
    }
    return res_arr;
}
/// Internal helper function to read a two-dimensional array of `i64` values.
///
/// # Arguments
///
/// * `value` - A reference to a JSON value representing an array.
///
/// # Returns
///
/// A vector of `i64` values extracted from the nested arrays.
fn read_array_i64(value: &Value) -> Vec<i64> {
    let mut res_arr: Vec<i64> = vec![];
    match value {
        Value::Array(array) => {
            let si_y = &array.len();
            for y in 0..*si_y {
                let si_x = &array[y].as_array().expect("Error Array X read").len();
                for x in 0..*si_x {
                    let val = &array[y][x].as_i64();
                    match val {
                        Some(val) => {
                            res_arr.push(*val);
                        }
                        None => println!("Array read fault 1 {:?}", array[y][x]),
                    }
                }
            }
        }
        _ => println!("array read fault 2 {:?}", value),
    }
    return res_arr;
}

/// Internal helper function to read a two-dimensional array of `usize` values.
///
/// # Arguments
///
/// * `value` - A reference to a JSON value representing an array.
///
/// # Returns
///
/// A vector of `usize` values extracted from the nested arrays.
fn _read_array_usize(value: &Value) -> Vec<usize> {
    let mut res_arr: Vec<usize> = vec![];

    match value {
        Value::Array(array) => {
            for item in array {
                if let Some(arr) = item.as_array() {
                    for element in arr {
                        if let Some(num) = element.as_u64() {
                            res_arr.push(num as usize);
                        } else {
                            println!("Array read fault: Invalid number format");
                        }
                    }
                } else {
                    println!("Array read fault: Expected nested array");
                }
            }
        }
        _ => println!("Array read fault: Expected array value"),
    }

    res_arr
}

/// Internal helper function to read a single `i64` value.
///
/// # Arguments
///
/// * `value` - A reference to a JSON value representing a number.
///
/// # Returns
///
/// An `i64` value extracted from the JSON number.
fn _read_val_i64(value: &Value) -> i64 {
    let mut res = 0;
    match value {
        Value::Number(val) => {
            let temp = val.as_i64();
            match temp {
                Some(temp) => res = temp,
                None => println!("read value is not an int"),
            }
        }

        _ => println!("read_val_int: {:?} is not an int", value),
    }
    return res;
}

/// Internal helper function to read a single `f64` value.
///
/// # Arguments
///
/// * `value` - A reference to a JSON value representing a number.
///
/// # Returns
///
/// An `f64` value extracted from the JSON number.
fn _read_val_f64(value: &Value) -> f64 {
    let mut res: f64 = 0.0;
    match value {
        Value::Number(val) => {
            let temp = val.as_f64();
            match temp {
                Some(temp) => res = temp,
                None => println!("read value is not an int"),
            }
        }

        _ => println!("read_val_int: {:?} is not an int", value),
    }
    return res;
}

/// Internal helper function to read a single `u64` value.
///
/// # Arguments
///
/// * `value` - A reference to a JSON value representing a number.
///
/// # Returns
///
/// An `u64` value extracted from the JSON number.
fn _read_val_u64(value: &Value) -> u64 {
    let mut res: u64 = 0;
    match value {
        Value::Number(val) => {
            let temp = val.as_u64();
            match temp {
                Some(temp) => res = temp,
                None => println!("read value is not an int"),
            }
        }

        _ => println!("read_val_int: {:?} is not an int", value),
    }
    return res;
}

/// Internal helper function to read a single `usize` value.
///
/// # Arguments
///
/// * `value` - A reference to a JSON value representing a number.
///
/// # Returns
///
/// An `usize` value extracted from the JSON number.
fn _read_val_usize(value: &Value) -> usize {
    let mut res = 0;
    match value {
        Value::Number(val) => {
            let temp = val.as_u64();
            match temp {
                Some(temp) => res = temp,
                None => println!("read value is not an usize"),
            }
        }

        _ => println!("read_val_int: {:?} is not an int", value),
    }
    let res = res as usize;
    return res;
}