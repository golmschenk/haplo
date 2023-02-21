import subprocess
from pathlib import Path

import numpy as np

from ml4a.infer import load_trained_parameters_to_phase_amplitudes_model
from ml4a.nicer_model import Nyx9Wider
from ml4a.residual_model import LiraTraditionalShape, LiraTraditionalShape8xWidthWith0d5DoNoBn

# model = load_trained_parameters_to_phase_amplitudes_model()
model = LiraTraditionalShape8xWidthWith0d5DoNoBn()
model_path = Path('logs/LiraTraditionalShape8xWidthWith0d5DoNoBn_chi_squared_loss_larger_dataset_10x_batch_cont/best_validation_model.ckpt')
model.load_weights(model_path).expect_partial()
random_input = np.random.random(size=[1, 11, 1])
_ = model.predict(random_input)
model.save('check')
subprocess.run(['python', '-m', 'tf2onnx.convert', '--saved-model', './check', '--opset=10', '--output', 'ml4a/lira.onnx'])
