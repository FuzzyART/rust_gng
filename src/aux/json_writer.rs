//! A module for writing JSON data to blocks and files.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

//--------------------------------------------------------------------------------------------------
// Write functions

/// Writes a serializable value into a specified block and key within a JSON [`Value`].
///
/// This function ensures that the given `block` exists in the root JSON object. If it does not,
/// a new empty object is created for that block. The provided `key` and `value` are then inserted
/// into this block.
///
/// # Arguments
///
/// * `data` - A mutable reference to the root JSON [`Value`] where the block will be inserted.
/// * `block` - The name of the block (as a string slice) within which the key-value pair is stored.
/// * `key` - The key (as a string slice) under which the value is stored in the block.
/// * `value` - The value to be serialized and inserted into the block. Must implement [`Serialize`].
///
/// # Examples
///
/// ```ignore
/// use serde_json::Value;
/// use gng::aux::json_writer::write_value_to_block;
///
/// let mut data = serde_json::json!({});
/// write_value_to_block(&mut data, "settings", "debug", true);
/// assert_eq!(data["settings"]["debug"], true);
/// ```
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

/// Serializes a [`Value`] to pretty-printed JSON and writes it to a file.
///
/// This function takes a filename and a [`Value`], serializes the value into a formatted JSON string,
/// and writes that string to the specified file. If the file does not exist, it will be created.
/// If it exists, it will be overwritten.
///
/// # Arguments
///
/// * `file_name` - A string slice that holds the name of the file to write to.
/// * `content` - A reference to the [`Value`] to serialize and write.
///
/// # Returns
///
/// Returns a [`Result`] that is:
///
/// * [`Ok(())`] if the operation was successful.
/// * [`Err`] containing an error from file I/O or serialization if the operation failed.
///
/// # Examples
///
/// ```ignore
/// use serde_json::Value;
/// use gng::aux::json_writer::write_json_to_file;
///
/// let content = serde_json::json!({"name": "test", "value": 42});
/// let result = write_json_to_file("output.json", &content);
/// assert!(result.is_ok());
/// ```
pub fn write_json_to_file(
    file_name: &str,
    content: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = serde_json::to_string_pretty(content)?;
    let mut file = File::create(file_name)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}