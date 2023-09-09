use std::sync::LazyLock;
use std::{fs, slice};
use tract_onnx::prelude::tract_ndarray::{Array1, s};
use tract_onnx::prelude::{Datum, Framework, Graph, InferenceFact, InferenceModelExt, RunnableModel, Tensor, TractResult, tvec, TypedFact, TypedOp};
use toml::Table;
extern crate libc;
use tract_onnx::prelude::*;

pub static MODEL: LazyLock<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> = LazyLock::new(|| {
    let filename = "haplo_configuration.toml";
    let configuration_file_contents = fs::read_to_string(filename).unwrap();
    let value = configuration_file_contents.parse::<Table>().unwrap();
    let onnx_path_string = value["onnx_model_path"].as_str().unwrap();
    let model = tract_onnx::onnx()
        .model_for_path(onnx_path_string).unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 11))).unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();
    model
});

pub static PARAMETER_STANDARD_DEVIATIONS: LazyLock<tract_ndarray::Array1<f32>> = LazyLock::new(|| {
    tract_ndarray::arr1(&[
        0.28133126679885656, 0.28100480365686287, 0.28140136435474244, 0.907001394792043, 1.811683338833852, 0.2815981892528909, 0.281641754864262, 0.28109705707606697, 0.9062620846468298, 1.8139690831565327, 2.886950440590801
    ])
});

pub static PARAMETER_MEANS: LazyLock<tract_ndarray::Array1<f32>> = LazyLock::new(|| {
    tract_ndarray::arr1(&[
        -0.0008009571736463096, -0.0008946310379428422, -2.274708783534052e-05, 1.5716876559520705, 3.1388159291733086, -0.001410436081400537, -0.0001470613574040905, -3.793528434430451e-05, 1.5723036365564083, 3.1463088925150258, 5.509554132916939
    ])
});

pub static PHASE_AMPLITUDE_MEAN: f32 = 34025.080543335825;
pub static PHASE_AMPLITUDE_STANDARD_DEVIATION: f32 = 47698.66676993027;

pub fn infer_from_parameters_to_phase_amplitudes_array(parameters: Array1<f32>) -> TractResult<Array1<f32>> {
    let normalized_parameters = (parameters - &*PARAMETER_MEANS) / &*PARAMETER_STANDARD_DEVIATIONS;
    let input_tensor: Tensor = normalized_parameters.insert_axis(tract_ndarray::Axis(0)).into();
    let output_tensor = MODEL.run(tvec!(input_tensor.into()))?;
    let output_array = output_tensor[0].to_array_view::<f32>()?;
    let normalized_phase_amplitudes = output_array.slice(s![0, ..]);
    let phase_amplitudes = (&normalized_phase_amplitudes * PHASE_AMPLITUDE_STANDARD_DEVIATION) + PHASE_AMPLITUDE_MEAN;
    Ok(phase_amplitudes)
}

#[no_mangle]
pub extern "C" fn infer_from_parameters_to_phase_amplitudes(parameters_array_pointer: *const f32, phase_amplitudes_array_pointer: *mut f32) {
    let parameters_slice = unsafe { slice::from_raw_parts(parameters_array_pointer, 11) };
    let parameters_array: Array1<f32> = tract_ndarray::arr1(parameters_slice);
    let phase_amplitudes_array = infer_from_parameters_to_phase_amplitudes_array(parameters_array).unwrap();
    let mut phase_amplitudes_array_iter = phase_amplitudes_array.iter();
    for index in 0..64 {
        unsafe {
            *phase_amplitudes_array_pointer.offset(index as isize) = phase_amplitudes_array_iter.next().unwrap().clone();
        }
    }
}
