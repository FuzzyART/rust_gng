#![allow(warnings)]
pub mod data_structures;
pub mod ecs;
pub mod gas;
pub mod handlers;

pub mod internal {
    // Re-export the original types and functions from core
    pub use crate::gas::core::{
        fit as core_fit, init_dataset as core_init_dataset, load_config as core_load_config,
        save_model_json as core_save_model_json, GngParams,
    };
}

pub struct Context {
    cont_params: internal::GngParams,
}

impl Context {
    pub fn new() -> Self {
        Context {
            cont_params: internal::GngParams::init(),
        }
    }

    pub fn load_config(&mut self, filename_config: &str) {
        self.cont_params.create_system();
        // If core expects &String, convert from &str
        internal::core_load_config(&mut self.cont_params, &filename_config.to_string());
    }

    pub fn init_dataset(&mut self, filename_dataset: &str) {
        internal::core_init_dataset(&mut self.cont_params, &filename_dataset.to_string());
    }

    pub fn fit(&mut self) {
        internal::core_fit(&mut self.cont_params);
    }
    pub fn save_model_json(&mut self, filename_output: &str) {
        internal::core_save_model_json(&mut self.cont_params, filename_output.to_string());
    }
}
#[pymethods]
impl PyContext {
    #[new]
    fn new() -> Self {
        PyContext {
            context: Context::new(),
        }
    }

    fn load_config(&mut self, filename_config: &str) -> PyResult<()> {
        self.context.load_config(filename_config);
        Ok(())
    }

    fn init_dataset(&mut self, filename_dataset: &str) -> PyResult<()> {
        self.context.init_dataset(filename_dataset);
        Ok(())
    }

    fn fit(&mut self) -> PyResult<()> {
        self.context.fit();
        Ok(())
    }

    fn save_model_json(&mut self, filename_output: &str) -> PyResult<()> {
        self.context.save_model_json(filename_output);
        Ok(())
    }
}
/// Define the Python module - renamed to match your package name
#[pymodule]
fn gng_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyContext>()?;
    Ok(())
}