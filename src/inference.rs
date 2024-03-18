use std::sync::LazyLock;
use std::{fs, slice};
use tract_onnx::prelude::tract_ndarray::{Array1, s};
use tract_onnx::prelude::{Datum, Framework, Graph, InferenceFact, InferenceModelExt, RunnableModel, Tensor, TractResult, tvec, TypedFact, TypedOp};
use toml::{Table, Value};

extern crate libc;

use tract_onnx::prelude::*;

fn toml_value_to_vec_f32(value: &Value) -> Vec<f32> {
    value.as_array().unwrap().into_iter().map(|item| item.as_float().unwrap() as f32).collect::<Vec<f32>>()
}


static CONFIGURATION_TABLE: LazyLock<Table> =
    LazyLock::new(|| {
        let filename = "haplo_configuration.toml";
        let configuration_file_contents = fs::read_to_string(filename).unwrap();
        let table = toml::from_str(&configuration_file_contents).unwrap();
        return table;
    });

pub static MODEL: LazyLock<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> = LazyLock::new(|| {
    let onnx_path_string = CONFIGURATION_TABLE["onnx_model_path"].as_str().unwrap();
    let model = onnx()
        .model_for_path(onnx_path_string).unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, PARAMETER_MEANS.len()))).unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();
    model
});

pub static PARAMETER_STANDARD_DEVIATIONS: LazyLock<Array1<f32>> = LazyLock::new(|| {
    let parameter_standard_deviations: Vec<f32>;
    if let Some(value) = CONFIGURATION_TABLE.get("parameter_standard_deviations") {
        parameter_standard_deviations = toml_value_to_vec_f32(value);
    } else {
        parameter_standard_deviations = vec![
            0.28133126679885656, 0.28100480365686287, 0.28140136435474244, 0.907001394792043, 1.811683338833852, 0.2815981892528909, 0.281641754864262, 0.28109705707606697, 0.9062620846468298, 1.8139690831565327, 2.886950440590801
        ];
    }
    tract_ndarray::arr1(&parameter_standard_deviations)
});

pub static PARAMETER_MEANS: LazyLock<Array1<f32>> = LazyLock::new(|| {
    let parameter_means: Vec<f32>;
    if let Some(value) = CONFIGURATION_TABLE.get("parameter_means") {
        parameter_means = toml_value_to_vec_f32(value);
    } else {
        parameter_means = vec![
            -0.0008009571736463096, -0.0008946310379428422, -2.274708783534052e-05, 1.5716876559520705, 3.1388159291733086, -0.001410436081400537, -0.0001470613574040905, -3.793528434430451e-05, 1.5723036365564083, 3.1463088925150258, 5.509554132916939
        ];
    }
    tract_ndarray::arr1(&parameter_means)
});

pub static PHASE_AMPLITUDE_MEAN: LazyLock<f32> = LazyLock::new(|| {
    let phase_amplitude_mean: f32;
    if let Some(value) = CONFIGURATION_TABLE.get("phase_amplitude_mean") {
        phase_amplitude_mean = value.as_float().unwrap() as f32;
    } else {
        phase_amplitude_mean = 34025.080543335825
    }
    phase_amplitude_mean
});

pub static PHASE_AMPLITUDE_STANDARD_DEVIATION: LazyLock<f32> = LazyLock::new(|| {
    let phase_amplitude_standard_deviation: f32;
    if let Some(value) = CONFIGURATION_TABLE.get("phase_amplitude_standard_deviation") {
        phase_amplitude_standard_deviation = value.as_float().unwrap() as f32;
    } else {
        phase_amplitude_standard_deviation = 47698.66676993027
    }
    phase_amplitude_standard_deviation
});

pub fn infer_from_parameters_to_phase_amplitudes_array(parameters: Array1<f32>) -> TractResult<Array1<f32>> {
    let normalized_parameters = (parameters - &*PARAMETER_MEANS) / &*PARAMETER_STANDARD_DEVIATIONS;
    let input_tensor: Tensor = normalized_parameters.insert_axis(tract_ndarray::Axis(0)).into();
    let output_tensor = MODEL.run(tvec!(input_tensor.into()))?;
    let output_array = output_tensor[0].to_array_view::<f32>()?;
    let normalized_phase_amplitudes = output_array.slice(s![0, ..]);
    let phase_amplitudes = (&normalized_phase_amplitudes * *PHASE_AMPLITUDE_STANDARD_DEVIATION) + *PHASE_AMPLITUDE_MEAN;
    Ok(phase_amplitudes)
}

#[no_mangle]
pub extern "C" fn infer_from_parameters_to_phase_amplitudes(parameters_array_pointer: *const f32, phase_amplitudes_array_pointer: *mut f32) {
    let parameters_slice = unsafe { slice::from_raw_parts(parameters_array_pointer, PARAMETER_MEANS.len()) };
    let parameters_array: Array1<f32> = tract_ndarray::arr1(parameters_slice);
    let phase_amplitudes_array = infer_from_parameters_to_phase_amplitudes_array(parameters_array).unwrap();
    let mut phase_amplitudes_array_iter = phase_amplitudes_array.iter();
    for index in 0..64 {
        unsafe {
            *phase_amplitudes_array_pointer.offset(index as isize) = phase_amplitudes_array_iter.next().unwrap().clone();
        }
    }
}
