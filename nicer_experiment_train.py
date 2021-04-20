import datetime
import random
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.python.keras import callbacks
from pathlib import Path

from nicer_example import NicerExample
from nicer_model import SimpleModel, WiderWithDropoutModel, Nyx4, Nyx5, Nyx2, Nyx4Narrow, Nyx6, Nyx7, Nyx8, Nyx9, Nyx10, \
    Nyx11, Nyx12, Nyx13, Nyx14, Nyx15, Nyx16, Nyx17, Nyx18, Nyx19, Nyx20, Nyx21, Nyx22, Nyx23, Eos0, Eos1, Eos2, \
    Nyx9Narrow, Nyx9Wide, Nyx9Wider


def main():
    print("Imports complete.", flush=True)
    wandb.init(project='ml4a', entity='ramjet', settings=wandb.Settings(start_method='fork'))
    dataset_path = Path("data/mcmc_vac_all_f90.dat")
    examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
    random.Random(0).shuffle(examples)
    tenth_dataset_count = int(len(examples) * 0.1)
    train_examples = examples[:-2*tenth_dataset_count]
    validation_examples = examples[-2*tenth_dataset_count:-tenth_dataset_count]
    test_examples = examples[-tenth_dataset_count:]
    print(f'Dataset sizes: train={len(train_examples)}, validation={len(validation_examples)}, '
          f'test={len(test_examples)}')
    train_dataset = NicerExample.to_prepared_tensorflow_dataset(train_examples, shuffle=True)
    validation_dataset = NicerExample.to_prepared_tensorflow_dataset(validation_examples)

    model = Nyx9Wider()
    wandb.run.notes = f"{type(model).__name__}er_no_do"
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    loss_metric = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = Path("logs").joinpath(f'{datetime_string}')
    best_validation_model_save_path = trial_directory.joinpath('best_validation_model.ckpt')
    best_validation_checkpoint_callback = callbacks.ModelCheckpoint(
        best_validation_model_save_path, monitor='val_loss', mode='min', save_best_only=True,
        save_weights_only=True)
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.fit(train_dataset, epochs=5000, validation_data=validation_dataset,
              callbacks=[WandbCallback(), best_validation_checkpoint_callback])


if __name__ == "__main__":
    main()
