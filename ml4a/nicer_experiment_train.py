import datetime
import random
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.python.keras import callbacks
from pathlib import Path

from ml4a.nicer_example import NicerExample
from ml4a.residual_model import ResModel1NoDoAvgPoolEnd8Wider, \
    ResModel1InitialDenseNoDoConvEndDoublingWider, ResModel1InitialDenseNoDoConvEndDoublingWiderer, \
    ResModel1InitialDenseNoDoConvEndDoublingWidererL2


def main():
    print("Imports complete.", flush=True)
    wandb.init(project='ml4a', entity='ramjet', settings=wandb.Settings(start_method='fork'))
    dataset_path = Path("../data/mcmc_vac_all_f90.dat")
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

    model = ResModel1InitialDenseNoDoConvEndDoublingWidererL2()
    wandb.run.notes = f"{type(model).__name__}"
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    loss_metric = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = Path("logs").joinpath(f'{wandb.run.notes}')
    best_validation_model_save_path = trial_directory.joinpath('best_validation_model.ckpt')
    best_validation_checkpoint_callback = callbacks.ModelCheckpoint(
        best_validation_model_save_path, monitor='val_loss', mode='min', save_best_only=True,
        save_weights_only=True)
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.fit(train_dataset, epochs=5000, validation_data=validation_dataset,
              callbacks=[WandbCallback(), best_validation_checkpoint_callback])


if __name__ == "__main__":
    main()
