# Imports.
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

class MultiLabelClassifier:
    def __init__(self,
                 input_shape=(512, 512, 3),
                 output_shape=9,
                 show_summary=True,
                 activation='relu',
                 learning_rate=1e-4,
                 epsilon=1e-5,
                 optimizer='adamw',
                 dropout=0.1,
                 batch_size=32
                 ):
        self.in_shape = input_shape
        self.out_shape = output_shape
        self.dropout = dropout
        self.activation = activation
        self.decay = epsilon
        self.batch_size = batch_size

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
            loss='binary_crossentropy',
            metrics=['mae'],
            )

    def build_model(self):
        # Initial input layer to the network
        input_layer = Input(shape=self.in_shape)

        x = input_layer
        x = Conv2D(filters=16, kernel_size=(5, 5), activation=self.activation)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(self.dropout)(x)
        x = Conv2D(filters=32, kernel_size=(5, 5), activation=self.activation)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(self.dropout)(x)
        x = Conv2D(filters=64, kernel_size=(5, 5), activation=self.activation)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(self.dropout)(x)
        x = Conv2D(filters=64, kernel_size=(5, 5), activation=self.activation)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(self.dropout)(x)

        x = tf.keras.layers.Flatten()(x)
        x = Dense(units=128, activation=self.activation)(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = Dense(units=64, activation=self.activation)(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = Dense(units=self.out_shape, activation='sigmoid')(x)

        output_layer = x

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name='ImageClassifier'
            )

        return model


if __name__ == '__main__':

    model = ().build_model()