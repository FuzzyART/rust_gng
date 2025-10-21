#![allow(warnings)]
pub mod data_structures;
pub mod ecs;
pub mod gas;
pub mod handlers;

use pyo3::prelude::*;
use numpy::{PyArray1};

pub mod internal {
    // Re-export the original types and functions from core
    pub use crate::gas::core::{Handler,
        fit as core_fit, init_dataset as core_init_dataset, 
init_dataset_vec as core_init_dataset_vec,
         load_config as core_load_config,
        save_model_json as core_save_model_json, 
        get_model_string as core_get_model_string,
    };
}

pub struct Context {
    cont_params: internal::Handler,
}

impl Context {
    pub fn new() -> Self {
        Context {
            cont_params: internal::Handler::init(),
        }
    }

    pub fn load_config(&mut self, filename_config: &str) {
        self.cont_params.create_system();
        // If core expects &String, convert from &str
        internal::core_load_config(&mut self.cont_params, &filename_config.to_string());
    }

    pub fn init_dataset_vec(&mut self, data: &Vec<f64>) {
        internal::core_init_dataset_vec(&mut self.cont_params, data);
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
    pub fn get_model_string(&mut self) ->String {
        internal::core_get_model_string(&mut self.cont_params)
    }

    pub fn foo(&mut self)->String{
        "hello int".to_string()
    }
    pub fn foo_vec(&mut self)->(Vec<usize>,Vec<f64>){
        let res1:Vec<usize> = vec!(1,3);
        let res2:Vec<f64> = vec!(1.0,2.2);
        (res1,res2)
    }


    
}
/// Python wrapper for Context struct
#[pyclass]
//#[derive(Debug, Clone)]
pub struct PyContext {
    context: Context,
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

    fn init_dataset_vec(&mut self, dataset: &PyArray1<f64>  ) -> PyResult<()> {
        let slice = unsafe { dataset.as_slice()? };
        let vec_data: Vec<f64> = slice.to_vec();
        self.context.init_dataset_vec(&vec_data);
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

    fn foo(&mut self) -> PyResult<String>{
        Ok(self.context.foo())
    }
    fn foo_val(&mut self) -> PyResult<f64>{
        Ok(6.4)
    }
    fn foo_vec(&mut self) -> PyResult<(Vec<usize>,Vec<f64>)>{
        Ok(self.context.foo_vec())
    }

    fn save_model_json(&mut self, filename_output: &str) -> PyResult<()> {
        self.context.save_model_json(filename_output);
        Ok(())
    }
    fn get_model_string(&mut self) -> PyResult<String>{
        Ok(self.context.get_model_string())
    }
}
/// Define the Python module - renamed to match your package name
#[pymodule]
fn gng_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyContext>()?;
    Ok(())
}



