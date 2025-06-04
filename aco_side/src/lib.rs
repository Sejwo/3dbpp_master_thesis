pub mod structs;
pub mod utils;
use crate::structs::AcoStruct;
use pyo3::prelude::*;
use serde_json;
use std::fs;

/// Run ACO from a JSON file, return full metrics as a JSON string.
#[pyfunction]
fn solve_from_json(input_path: &str) -> PyResult<String> {
    let mut model = AcoStruct::from_json(input_path);
    let result = model.run_model();
    let json = serde_json::to_string_pretty(&result).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Serialization error: {}", e))
    })?;
    Ok(json)
}

/// Run ACO from a JSON file, write full metrics + placements to an output JSON file for visualisation
#[pyfunction]
fn solve_and_write(input_path: &str, output_path: &str) -> PyResult<()> {
    // build model
    let mut model = AcoStruct::from_json(input_path);
    model.run_model();
    // now call the built-in to_json, which writes both placements and score:
    model.to_json(output_path);
    Ok(())
}

/// Python module definition
#[pymodule]
fn aco3dbpp(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(solve_and_write, m)?)?;
    Ok(())
}
