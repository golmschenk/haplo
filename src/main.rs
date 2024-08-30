mod inference;
use crate::inference::{infer_from_parameters_to_phase_amplitudes_array};

use tract_onnx::prelude::*;


fn main() -> TractResult<()> {
    let parameters: tract_ndarray::Array1<f32> = tract_ndarray::arr1(&[
        -0.137349282472716, 4.651922986569446E-002, -0.126309026142708, 2.57614122691645, 3.94358482944553, 0.303202923979724, 0.132341360556433, 0.304479697430865, 0.758863131388038, 3.84855473811096, 2.77055893884855
    ]);
    let phase_amplitudes = infer_from_parameters_to_phase_amplitudes_array(parameters)?;

    println!("Phase amplitudes: {:?}", phase_amplitudes);

    Ok(())
}
