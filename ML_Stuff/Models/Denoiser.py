import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Embedding, Input, Conv2D, Reshape, Conv2DTranspose,MultiHeadAttention, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

class denoiserModel:
    def __init__(self,
                 input_shape=(9, 16),
                 depth=4,
                 show_summary=True,
                 loss='mse',
                 activation='swish',
                 learning_rate=1e-3,
                 epsilon=1e-5,
                 optimizer='adam',
                 fft_size=1024,
                 shift_size=120,
                 win_length=600,
                 window=tf.signal.hann_window
                 ):
        self.in_shape = input_shape
        self.out_shape = input_shape
        self.depth = depth
        self.activation = activation
        self.decay = epsilon
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = window

        self.predictor_model = self.build_model()
    
        if show_summary:
            self.predictor_model.summary()

        self.lr = learning_rate
        self.optimizer = optimizer

        if self.optimizer == 'adam':
            optimizer_instance = Adam(
                learning_rate=self.lr,
                epsilon=self.decay
                )
        elif self.optimizer == 'sgd':
            optimizer_instance = SGD(
                learning_rate=self.lr,
                momentum=self.decay,
                nesterov=False
                )
        elif self.optimizer == 'rms':
            optimizer_instance = RMSprop(
                learning_rate=self.lr,
                centered=False,
                epsilon=1e-7,
                rho=.9
                )
        elif self.optimizer == 'adamw':
            optimizer_instance = tfa.optimizers.AdamW(
                learning_rate=self.lr,
                weight_decay=self.decay
            )

        self.predictor_model.compile(
            optimizer=optimizer_instance,
            loss=loss,
            metrics=['mae'],
            )

    def build_model(self):
        # Initial input layer to the network
        input_layer = Input(shape=self.in_shape)
        x = input_layer
        x = Reshape((self.in_shape[0], self.in_shape[1], 1))(x)
    
        for i in range(self.depth):
            x = Conv2D(filters=(16 * (i + 1)), kernel_size=(1, 4), strides=(1, 2), padding='same', activation='LeakyReLU')(x)

        position_embedding = Embedding(input_dim=4, output_dim=self.depth * 16)
        positions = tf.range(start=0, limit=4, delta=1)
        embedded_pos = position_embedding(positions)
        embedded_pos = tf.expand_dims(embedded_pos, axis=0)
        repeat_pos_emb = RepeatLayer()(embedded_pos, self.in_shape[0])
        x = x + repeat_pos_emb

        x = MultiHeadAttention(num_heads=20, key_dim=64, attention_axes=(2, 3))(x, x)
        x = LeakyReLU()(x)

        for i in range(self.depth):
            x = Conv2DTranspose(filters=(16 * (self.depth - i)), kernel_size=(1, 5), strides=(1, 2), padding='same', activation='LeakyReLU')(x)

        x = Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=(1, 1))(x)
        output_layer = x

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name='denoiser'
            )

        return model


class RepeatLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(RepeatLayer, self).__init__()

    def call(self, repeatee, num_repeats):
        # shapes = tf.shape(patches)

        return tf.repeat(repeatee, num_repeats, axis=0)


if __name__ == '__main__':
    # Load the model.
    MODEL = denoiserModel(
        input_shape=(469, 64),
        depth=3,
        learning_rate=1e-3
    ).predictor_model
