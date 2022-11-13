# Setup
This setup expects `conda` is installed in some form already. The minimal `conda` installer is available [here](https://docs.conda.io/en/latest/miniconda.html).

1. Create a new `conda` environment for the project:

        conda create -n ml4a python=3.8 -y

2. Enter the `conda` environment:

        conda activate ml4a

3. Install the `ml4a` Python package:

        pip install https://github.com/golmschenk/ml4a/tarball/master

4. Download the trained model states:

        python -m ml4a.download_model_states

   We note that the model states downloaded via this command are for minimal model architectures designed for ease of use. The model state for our current best "parameters to phase amplitudes" model architecture can be downloaded [here](https://olmschenk-my.sharepoint.com/:f:/p/greg/EgNXsdvmyMlJg58-bEFS3_YBKcf5SDQpkKM1nEqVGWBt-A?e=L318gC).


5. Exit the `conda` environment:

        conda deactivate


# Usage
1. Enter the `conda` environment created during setup:

        conda activate ml4a

2. Add your input data to a CSV file. Each row should be a single example and the values should be delimited commas.There is no requirement for the number of digits for a value. Scientific notation using `e` or
   `E` is allowed. See [parameters_template.csv](parameters_template.csv) and
   [phase_amplitudes_template.csv](phase_amplitudes_template.csv) for examples.

3. Run the inference for either a parameters or phase amplitudes CSV input, specifying your desired input and output
   paths:

        python -m ml4a.infer_from_phase_amplitudes_to_parameters input_phase_amplitudes.csv output_parameters.csv

   or

        python -m ml4a.infer_from_parameters_to_phase_amplitudes input_parameters.csv output_phase_amplitudes.csv

4. Exit `conda` environment:

        conda deactivate
