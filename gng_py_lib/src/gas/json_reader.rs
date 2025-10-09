//use json;
use serde_json::{self, Value};
//use std::env;
//use std::{array, fs};
use std::error::Error;
use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Cell {
    pub value: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Table {
    pub cells: Vec<Cell>,
}

//--------------------------------------------------------------------------------------------------
// public
pub fn read_array_f64(content: &Value, x: &str, y: &str) -> Vec<f64> {
    let input = &content[x][y];
    return _read_array_f64(&input);
}

pub fn read_file(file_name: &str) -> Result<Value, String> {
    let file_content = fs::read_to_string(file_name).expect("file not found");
    //let json_content = serde_json::from_str(&file_content).expect("json file corrupted");
    let json_content: Value = serde_json::from_str(&file_content).expect("json read fault 1");
    return Ok(json_content);
}
pub fn read_array_usize(content: &Value, x: &str, y: &str) -> Vec<usize> {
    let input = &content[x][y];
    return _read_array_usize(&input);
}
pub fn read_val_usize(content: &Value, x: &str, y: &str) -> usize {
    let input = &content[x][y];
    return _read_val_usize(&input);
}
pub fn read_val_f64(content: &Value, x: &str, y: &str) -> f64 {
    let input = &content[x][y];
    return _read_val_f64(&input);
}

pub fn read_file_temp(file_name: &str) -> Result<Value, String> {
    let file_content = fs::read_to_string(file_name).expect("file not found");
    let json_content: Value = serde_json::from_str(&file_content).expect("json read fault 1");
    return Ok(json_content);
}

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
pub fn read_array_i64(value: &Value) -> Vec<i64> {
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

pub fn _read_array_usize(value: &Value) -> Vec<usize> {
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

pub fn _read_val_i64(value: &Value) -> i64 {
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
