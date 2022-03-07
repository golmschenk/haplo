
use tract_onnx::prelude::*;
use ml4a::{MODEL, PARAMETER_MEANS, PARAMETER_STANDARD_DEVIATIONS, PHASE_AMPLITUDE_MEAN, PHASE_AMPLITUDE_STANDARD_DEVIATION};
use crate::tract_ndarray::{Array1, s};

fn main() -> TractResult<()> {
    let parameters: tract_ndarray::Array1<f32> = tract_ndarray::arr1(&[
        -0.137349282472716, 4.651922986569446E-002, -0.126309026142708, 2.57614122691645, 3.94358482944553, 0.303202923979724, 0.132341360556433, 0.304479697430865, 0.758863131388038, 3.84855473811096, 2.77055893884855
    ]);
    let phase_amplitudes = infer_from_parameters_to_phase_amplitudes(parameters)?;

    println!("Phase amplitudes: {:?}", phase_amplitudes);

    Ok(())
}

fn infer_from_parameters_to_phase_amplitudes(parameters: Array1<f32>) -> TractResult<Array1<f32>> {
    let normalized_parameters = (parameters - &*PARAMETER_MEANS) / &*PARAMETER_STANDARD_DEVIATIONS;
    let input_tensor: Tensor = normalized_parameters.insert_axis(tract_ndarray::Axis(0)).insert_axis(tract_ndarray::Axis(2)).into();
    let output_tensor = MODEL.run(tvec!(input_tensor))?;
    let output_array = output_tensor[0].to_array_view::<f32>()?;
    let normalized_phase_amplitudes = output_array.slice(s![0, ..]);
    let phase_amplitudes = (&normalized_phase_amplitudes * PHASE_AMPLITUDE_STANDARD_DEVIATION) + PHASE_AMPLITUDE_MEAN;
    Ok(phase_amplitudes)
}
