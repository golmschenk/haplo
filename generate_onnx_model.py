import subprocess
from pathlib import Path

import numpy as np

from ml4a.infer import load_trained_parameters_to_phase_amplitudes_model
from ml4a.nicer_model import Nyx9Wider
from ml4a.residual_model import LiraTraditionalShape

# model = load_trained_parameters_to_phase_amplitudes_model()
model = LiraTraditionalShape()
model_path = Path('logs/LiraTraditionalShape_normalized_loss_lr_1e-4_exported_2022_10_31/best_validation_model.ckpt')
model.load_weights(model_path).expect_partial()
random_input = np.random.random(size=[1, 11, 1])
_ = model.predict(random_input)
model.save('check')
subprocess.run(['python', '-m', 'tf2onnx.convert', '--saved-model', './check', '--opset=10', '--output', 'ml4a/lira.onnx'])
