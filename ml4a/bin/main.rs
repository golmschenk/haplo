
use tract_onnx::prelude::*;
use ml4a::{infer_from_parameters_to_phase_amplitudes, MODEL, PARAMETER_MEANS, PARAMETER_STANDARD_DEVIATIONS, PHASE_AMPLITUDE_MEAN, PHASE_AMPLITUDE_STANDARD_DEVIATION};
use crate::tract_ndarray::{Array1};

fn main() -> TractResult<()> {
    let parameters: tract_ndarray::Array1<f32> = tract_ndarray::arr1(&[
        -0.137349282472716, 4.651922986569446E-002, -0.126309026142708, 2.57614122691645, 3.94358482944553, 0.303202923979724, 0.132341360556433, 0.304479697430865, 0.758863131388038, 3.84855473811096, 2.77055893884855
    ]);
    let phase_amplitudes = infer_from_parameters_to_phase_amplitudes(parameters)?;

    println!("Phase amplitudes: {:?}", phase_amplitudes);

    Ok(())
}
