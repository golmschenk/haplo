# Getting started

## Set up
First, download the example project directory from [here](https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2Fgolmschenk%2Fhaplo_example_application). Be sure to rename the project directory to a name that describes your project before getting started.

Inside this project directory you will find several directories:
1. A `data` directory that contains an example data file in a text format output by Constantinos Kalapotharakos' modeling code.
2. A `scripts` directory that contains the example scripts we'll run in this tutorial.
3. A `sessions` directory that will contain the logs and output models from the training sessions. It currently contains one pretrained model.
4. A `wandb` directory that is empty now, but is required by a dependant library we use to log training.

All scripts are designed to be run from the project root directory. That is, you will run:
```shell
python scripts/<script_name>.py
```
rather than going to the `scripts` directory directly.

## Preparing the data
First, we need to convert the data from Constantinos Kalapotharakos' format to a format the haplo package expects. This is expected to change, but for now, the expected format is an SQLite database. To convert this data, we will use the `scripts/example_data_preparation.py`. If you open this file, you will see a very simple script. To run this on another Constantinos Kalapotharakos' format file, just change the paths specified in this script. However, for this example, the script points to the example data in your data directory, so no change is needed. Just run
```shell
python scripts/example_data_preparation.py
```
This will produce the specificed `.db` file in the data directory.

## Visualizing the data
Next, we can see how we could inspect an example from this dataset. Open the `scripts/example_data_visualization.py` file. Here, you will see a `NicerDataset` object created based on the example SQLite database file. Then
```python
example0 = dataset[0]
parameters0, phase_amplitudes0 = example0
```
is used to get the first parameters and phase amplitudes of the first example. This index could be changed to get any example from the dataset. The remainder of the code is simply creating figures to display the parameters and phase amplitudes. Run
```shell
python scripts/example_data_visualization.py
```
to see the data from this first parameters and phase folded light curve pair. You should see something like the following.
```{image} data_visualization.png
:width: 800px
```

## Training the network
Next, we will train the network. Open the `scripts/example_train_session.py` file. Inside the main `example_train_session` function, there are several parts. There are several components that can be adjusted here, but for now, we'll just go over the high-level details. First, the dataset is prepared. Then, the neural network training setup is specified. This includes the model choice, metrics choice, optimizer choice, and logging settings. Finally, we pass all this to the `train_session` function, which runs the training loop. Note that on the line that starts with `hyperparameter_configuration = ...` we set `cycles=10`. This is the number of train and validate cycles that will be run. 

Before running this script, we need to set up WandB, a logging library we are using. The best option for this is to create a WandB account on the [WandB website](https://wandb.ai/site). Once complete, from the terminal within the project root directory, run `wandb login` and follow the instructions. [WandB can also be run entirely locally](https://github.com/wandb/server), but we recommend using the WandB provided service if possible.

For our example training session, the only thing that might need to be changed in the script is the `wandb_entity` being passed to the logging configuration. If you are a member of the team that developed haplo, you can ask Greg to be added to the `ramjet` WandB team. In this case, you do not need to alter the script. If you are not added to the `ramjet` WandB team, you should change `wandb_entity=ramjet` to `wandb_entity=<your_wandb_username>`. The free tier from WandB is plenty sufficient for this example and probably sufficient for many full projects. If you need more and qualify for their free academic premium plan, you can apply for that.

With all this finished, we are ready to run the example script using
```shell
python scripts/example_train_session.py
```

The WandB link where you can watch the training progress should be displayed in the terminal. After every train and validate cycle, the trained model will be saved to the `sessions` directory.

## Inferring with the trained network
Now, we will run the trained network on a set of parameters it hasn't seen. Open the `scripts/example_infer_session.py` file. Again, here, we just give a brief explanation of the file. First, we create the dataset, quite similar to the train script, but we will now use the test split of the dataset (data the network has not seen) instead of the train split. Next, we load the trained network model, and set it to a evaluation (infer) mode rather than the default training mode. From there, we load a single parameters and light curve pair. The parameters are then passed through the network for inference. Note, the network is built to take in a batch of parameters to produce multiple light curves at the same time. So to process a single example, we need to add an extra dimension and remove it afterward. This is why there is a `expand` and `squeeze` involved. Lastly, after this, we produce a figure that compares what the network produced with the true value from the dataset. 

To run this script, we first need to change the path of the train network model weights file. On the line that starts with `saved_model_path = ...`, change `'sessions/your/path/to/model.pt'` to point to your trained model `lowest_validation_model.pt` file in your `sessions` directory (which will be named based on the datetime it was run). The resulting line will look something like `saved_model_path = Path('sessions/2025_08_06_16_06_06/lowest_validation_model.pt')`. Then, run
```shell
python scripts/example_infer_session.py
```

You should see something like the following.
```{image} example_infer_output_with_bad_model.png
:width: 400px
```

The red line is the network prediction where the blue is the ground truth. Of course, in this case, the network does very poorly, since we trained it with a tiny example dataset and only for 10 train cycles. To see what a good model looks like, change the path of the trained model to the included `sessions/example_pretrain_model/lowest_500m_validation_model.pt` file. Then re-run the infer script. And now the result should look much better this time around.

```{image} example_infer_output_with_good_model.png
:width: 400px
```

## Exporting the trained model
The trained model is currently stored as a PyTorch `.pt` file. To use the model in outside of PyTorch, including in our Rust package, we need to export the model to the Open Neural Network eXchange (ONNX) format. To do that, we use the `scripts/example_export_to_onnx.py` file. This is a very simple file, and the only thing that needs to be changed is to set the `.pt` path to point to your `.pt` file. Then run
```shell
python scripts/example_export_to_onnx.py
```