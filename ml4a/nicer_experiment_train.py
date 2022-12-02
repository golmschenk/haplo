import datetime
import random
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.python.keras import callbacks
from pathlib import Path

from ml4a.losses import RelativeMeanSquaredErrorLoss
from ml4a.nicer_example import NicerExample
from ml4a.nicer_model import Nyx9Wider, SimpleModel, Nyx9Re, Nyx9ReNarrowStartWideEnd, Nyx9ReTraditionalShape, \
    Nyx9ReTraditionalShape4xWide
from ml4a.residual_model import ResModel1NoDoAvgPoolEnd8Wider, \
    ResModel1InitialDenseNoDoConvEndDoublingWider, ResModel1InitialDenseNoDoConvEndDoublingWiderer, \
    ResModel1InitialDenseNoDoConvEndDoublingWidererL2, LiraWithDoNoLrExtraEndLayer, LiraNoL2, LiraExtraEndLayer, \
    LiraWithDoExtraEndLayer, NormalizingModelWrapper, Lira, Lira4xWide, LiraNoBn, LiraNoBnWithDo, LiraTraditionalShape, \
    LiraTraditionalShapeDoubleWidth, LiraTraditionalShapeDoubleWidthEndBranchDropout, LiraTraditionalShapeExtraEndLayer, \
    LiraTraditionalShapeDoubleWidthWithExtraEndLayer, LiraTraditionalShape4xWidth, \
    LiraTraditionalShape4xWidthWithExtraEndLayer, LiraTraditionalShapeDoubleWidthWithExtraEndLayerEndActivations, \
    LiraTraditionalShape2xWidth2xDepth, LiraTraditionalShapeEndSum, LiraTraditionalShapeWithoutDimensionDecrease, \
    LiraTraditionalShape2LayerSkipsWithoutDimensionDecrease, ResnetLike


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
    train_dataset = NicerExample.to_prepared_tensorflow_dataset(train_examples, shuffle=True,
                                                                normalize_parameters_and_phase_amplitudes=True)
    validation_dataset = NicerExample.to_prepared_tensorflow_dataset(validation_examples,
                                                                     normalize_parameters_and_phase_amplitudes=True)

    model = ResnetLike()
    wandb.run.notes = f"{type(model).__name__}_normalized_loss_cont"
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    loss_metric = RelativeMeanSquaredErrorLoss()
    metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanSquaredLogarithmicError()]
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = Path("logs").joinpath(f'{wandb.run.notes}')
    best_validation_model_save_path = trial_directory.joinpath('best_validation_model.ckpt')
    best_validation_checkpoint_callback = callbacks.ModelCheckpoint(
        best_validation_model_save_path, monitor='val_loss', mode='min', save_best_only=True,
        save_weights_only=True)
    # model.load_weights('logs/LiraTraditionalShape2LayerSkipsWithoutDimensionDecrease_normalized_loss/best_validation_model.ckpt')
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.run_eagerly = True
    model.fit(train_dataset, epochs=5000, validation_data=validation_dataset,
              callbacks=[WandbCallback(save_model=False), best_validation_checkpoint_callback])


if __name__ == "__main__":
    main()
