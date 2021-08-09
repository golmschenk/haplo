from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D, MaxPooling1D, AveragePooling1D

from nicer_phase_amplitudes_to_parameters_model_components import BottleNeckResidualLightCurveNetworkBlock


class Mira(Model):
    def __init__(self, number_of_label_types=11, number_of_input_channels: int = 1):
        super().__init__()
        self.reshape0 = Reshape([64, 1])
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [16, 32, 64, 128, 256, 512]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(3):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels,
                                                                            dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=1)
        self.prediction_layer = Convolution1D(number_of_label_types, kernel_size=1, activation=sigmoid)
        self.scaling_layer = Convolution1D(number_of_label_types, kernel_size=1, activation=None)
        self.reshape = Reshape([number_of_label_types])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.reshape0(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.final_pooling(x, training=training)
        x = self.prediction_layer(x, training=training)
        x = self.scaling_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class MiraNoBnNoDo(Model):
    def __init__(self, number_of_label_types=11, number_of_input_channels: int = 1):
        super().__init__()
        self.reshape0 = Reshape([64, 1])
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0))
        input_channels = output_channels
        for output_channels in [16, 32, 64, 128, 256, 512]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2,
                batch_normalization=False, dropout_rate=0))
            for _ in range(3):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                    output_channels=output_channels, batch_normalization=False, dropout_rate=0))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=1)
        self.prediction_layer = Convolution1D(number_of_label_types, kernel_size=1, activation=sigmoid)
        self.scaling_layer = Convolution1D(number_of_label_types, kernel_size=1, activation=None)
        self.reshape = Reshape([number_of_label_types])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.reshape0(x, training=training)
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.final_pooling(x, training=training)
        x = self.prediction_layer(x, training=training)
        x = self.scaling_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs
