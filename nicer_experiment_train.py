import datetime
import random
import tensorflow as tf
import wandb
from tensorflow.python.keras import callbacks
from pathlib import Path

from nicer_example import NicerExample
from nicer_model import SimpleModel, WiderWithDropoutModel, Nyx4, Nyx5, Nyx2, Nyx4Narrow, Nyx6, Nyx7, Nyx8, Nyx9, Nyx10, \
    Nyx11, Nyx12


def main():
    wandb.init(project='ml4a', entity='ramjet', sync_tensorboard=True)
    dataset_path = Path("data/mcmc_vac_all_f90.dat")
    examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
    random.Random(0).shuffle(examples)
    train_examples = examples[:8000]
    validation_examples = examples[8000:9000]
    test_examples = examples[9000:]
    train_dataset = NicerExample.to_prepared_tensorflow_dataset(train_examples, shuffle=True)
    validation_dataset = NicerExample.to_prepared_tensorflow_dataset(validation_examples)

    model = Nyx12()
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    loss_metric = tf.keras.losses.MeanSquaredError()
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = Path("logs").joinpath(f'{datetime_string}')
    tensorboard_callback = callbacks.TensorBoard(log_dir=trial_directory)
    best_validation_model_save_path = trial_directory.joinpath('best_validation_model.ckpt')
    best_validation_checkpoint_callback = callbacks.ModelCheckpoint(
        best_validation_model_save_path, monitor='val_loss', mode='min', save_best_only=True,
        save_weights_only=True)
    model.compile(optimizer=optimizer, loss=loss_metric)
    model.fit(train_dataset, epochs=100000, validation_data=validation_dataset,
              callbacks=[tensorboard_callback, best_validation_checkpoint_callback])


if __name__ == "__main__":
    main()
