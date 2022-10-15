use std::{collections::HashMap, fs::read_to_string};

use lazy_static::lazy_static;
use pyo3::{types::PyModule, PyResult, Python};
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
struct Windows(pub Vec<[String; 2]>);

lazy_static! {
    pub static ref WINDOWS: HashMap<(String, String), Vec<[String; 2]>> = {
        let mut hm = HashMap::new();
        let path = "in/nab/labels/combined_windows.json";
        let data = read_to_string(path).expect("windows file not found");
        let json: Value = serde_json::from_str(&data).expect("invalid json");
        let entries: Vec<_> = json.as_object().unwrap().clone().into_iter().collect();
        for (id, wins) in entries {
            let wins: Vec<[String; 2]> = serde_json::from_value(wins).unwrap();
            let (class, rest) = id.split_once('/').unwrap();
            let (name, _) = rest.split_once('.').unwrap();
            hm.insert((class.to_string(), name.to_string()), wins);
        }
        hm
    };
}

pub fn autoperiod(values: Vec<f32>) -> PyResult<usize> {
    let py_main = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/main.py"));
    Python::with_gil(|py| {
        let main = PyModule::from_code(py, py_main, "main.py", "main")?;
        let p = main.getattr("p")?;
        let res = p.call1((values,))?;
        let period = res.extract::<usize>()?;
        Ok(period)
    })
}
