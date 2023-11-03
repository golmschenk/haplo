import tensorflow as tf
import datetime
import gc
import random
import wandb
from wandb.keras import WandbCallback
from tensorflow.python.keras import callbacks
from pathlib import Path

from haplo.nicer_dataset import NicerDataset
from ml4a.losses import RelativeMeanSquaredErrorLoss, PlusOneBeforeUnnormalizationChiSquaredStatisticLoss, \
    PlusOneChiSquaredMeanDenominatorStatisticLoss, \
    PlusOneChiSquaredStatisticLoss
from ml4a.nicer_example import NicerExample
from ml4a.nicer_model import Nyx9Wider, SimpleModel, Nyx9Re, Nyx9ReNarrowStartWideEnd, Nyx9ReTraditionalShape, \
    Nyx9ReTraditionalShape4xWide
from ml4a.pregenerated_output_model import PregeneratedOutputModel, PregeneratedOutputModelNoDo
from ml4a.residual_model import ResModel1NoDoAvgPoolEnd8Wider, LiraTraditionalShape8xWidthWith0d5DoNoBn, \
    LiraTraditionalShape8xWidthWith0d5Do, LiraTraditionalShape8xWidthWithNoDoNoBnStrongLeakyReluStartingActivations, \
    LiraNewShapeWithNoDoNoBnStartingActivations, LiraTraditionalShape8xWidthWithNoDoNoBnNoDiDeStrongLeakyRelu, KairaBn, \
    LiraTraditionalShape8xWidthWithBnRe, Kaira20Bn, Kaira50Bn, LiraTraditionalShape8xWidthWithBn, \
    LiraTraditionalShape8xWidthWithNoDoNoBn, LiraTraditionalShape8xWidthWithNoDoNoBnNoL2


def main():
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        print("Imports complete.", flush=True)
        wandb.init(project='haplo', entity='ramjet', settings=wandb.Settings(start_method='fork'))
        model = LiraTraditionalShape8xWidthWithNoDoNoBn()
        wandb.run.notes = f"tf_{type(model).__name__}_more_corrected_loss_names"
        optimizer = tf.optimizers.Adam(learning_rate=1e-4, clipnorm=1)
        loss_metric = PlusOneBeforeUnnormalizationChiSquaredStatisticLoss()
        metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanSquaredLogarithmicError(), PlusOneBeforeUnnormalizationChiSquaredStatisticLoss().plus_one_before_unnormalization_chi_squared_statistic, RelativeMeanSquaredErrorLoss.relative_mean_squared_error_loss, PlusOneChiSquaredMeanDenominatorStatisticLoss().loss, PlusOneChiSquaredStatisticLoss().plus_one_chi_squared_statistic]
        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        trial_directory = Path("logs").joinpath(f'{wandb.run.notes}')
        best_validation_model_save_path = trial_directory.joinpath('best_validation_model.ckpt')
        best_validation_checkpoint_callback = callbacks.ModelCheckpoint(
            best_validation_model_save_path, monitor='val_loss', mode='min', save_best_only=True,
            save_weights_only=True)
        # model.load_weights('logs/LiraTraditionalShape8xWidthWithNoDoNoBn_chi_squared_loss_50m_dataset_small_batch_clip_norm_1_cont/best_validation_model.ckpt')
        model.run_eagerly = True
        model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
        # dataset_path = Path('data/mcmc_vac_all_800k.dat')
        train_dataset_path = Path('data/50m_rotated_parameters_and_phase_amplitudes.arrow')
        full_train_dataset = NicerDataset.new(dataset_path=train_dataset_path)
        examples = []
        for parameters, phase_amplitudes in full_train_dataset:
            examples.append(NicerExample.new(parameters=parameters, phase_amplitudes=phase_amplitudes, likelihood=0))
        # examples = NicerExample.list_from_constantinos_kalapotharakos_file(dataset_path)
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
    model.fit(train_dataset, epochs=5000, validation_data=validation_dataset,
              callbacks=[WandbCallback(save_model=False), best_validation_checkpoint_callback])


if __name__ == "__main__":
    main()
