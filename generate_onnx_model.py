import subprocess

import numpy as np

from ml4a.infer import load_trained_parameters_to_phase_amplitudes_model

model = load_trained_parameters_to_phase_amplitudes_model()
random_input = np.random.random(size=[1, 11, 1])
_ = model.predict(random_input)
model.save('check')
subprocess.run(['python', '-m', 'tf2onnx.convert', '--saved-model', './check', '--opset=10', '--output', 'lira.onnx'])
