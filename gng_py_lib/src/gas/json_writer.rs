use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

//--------------------------------------------------------------------------------------------------
// Write functions

pub fn write_value_to_block<T: Serialize>(data: &mut Value, block: &str, key: &str, value: T) {
    // Ensure block exists
    let block_obj = data
        .as_object_mut()
        .expect("Root should be a JSON object")
        .entry(block.to_string())
        .or_insert_with(|| json!({}));

    // Insert key-value into block
    block_obj
        .as_object_mut()
        .expect("Block should be a JSON object")
        .insert(key.to_string(), json!(value));
}

pub fn write_json_to_file(
    file_name: &str,
    content: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = serde_json::to_string_pretty(content)?;
    let mut file = File::create(file_name)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}
