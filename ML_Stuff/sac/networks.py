import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, concatenate, Dense, Flatten, Input, MaxPool2D, subtract, LayerNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from  tensorflow.keras.initializers import RandomUniform

class ActorNetwork(keras.Model):
    def __init__(
        self,
        n_actions,
        input_shape,
        input_type='image_diff',
        lr=1e-4,
        noise=1e-6,
        max_step=0.1,
        name='actor',
        chkpt_dir='',
        activation='LeakyReLU',
        hidden_kernel=False,
        hidden_bias=False,
        output_kernel=False,
        output_bias=False,
        skip=False,
        show_summary=True):

        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.in_shape = input_shape
        self.input_type = input_type
        self.lr = lr
        self.noise = noise
        self.max_step = max_step
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.activation = activation
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.model_name}.h5')
        self.skip = skip
        self.show_summary = show_summary

        # Kernel/bias initializers.
        if hidden_kernel:
            self.hidden_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_kernel = 'glorot_uniform'
        
        if hidden_bias:
            self.hidden_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_bias = 'zeros'
        
        if output_kernel:
            self.output_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_kernel = 'glorot_uniform'
        
        if output_bias:
            self.output_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_bias = 'zeros'

        self.optimizer = Adam(learning_rate=self.lr)

        self.model = self.build_model()

    def build_model(self):
        # Input layer.
        img_input = Input(shape=self.in_shape)
        target_input = Input(shape=self.in_shape)
        
        # Position embedding.
        length = self.in_shape[0] * self.in_shape[1]

        positions = tf.range(start=0, limit=length, delta=1)

        position_embedding = tf.keras.layers.Embedding(
            input_dim=length, output_dim=1
        )

        pos_embed = position_embedding(positions)
        pos_embed = tf.reshape(pos_embed, (self.in_shape[0], self.in_shape[1], 1))

        skip_val = pos_embed

        curr_x = tf.math.add(img_input, pos_embed)
        target_x = tf.math.add(target_input, pos_embed)
        

        # Current state CNN.
        curr_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='curr_conv1')(curr_x)
        # curr_x = MaxPool2D(padding='same')(curr_x)
        curr_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='curr_conv2')(curr_x)
        # curr_x = MaxPool2D(padding='same')(curr_x)
        curr_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='curr_conv3')(curr_x)
        # curr_x = MaxPool2D(padding='same')(curr_x)
        curr_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='curr_conv4')(curr_x)
        
        # Target state CNN.
        target_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='target_conv1')(target_x)
        # target_x = MaxPool2D(padding='same')(target_x)
        target_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='target_conv2')(target_x)
        # target_x = MaxPool2D(padding='same')(target_x)
        target_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='target_conv3')(target_x)
        # target_x = MaxPool2D(padding='same')(target_x)
        target_x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                    padding='same', activation=self.activation,
                    kernel_initializer=self.hidden_kernel,
                    name='target_conv4')(target_x)

        if self.skip:
            curr_x = tf.math.add(curr_x, skip_val)
            curr_x = LayerNormalization()(curr_x)
            target_x = tf.math.add(target_x, skip_val)
            target_x = LayerNormalization()(target_x)

        curr_x = Flatten()(curr_x)
        target_x = Flatten()(target_x)
        
        # Concatenate CNN outputs.
        x = concatenate([curr_x, target_x], axis=-1)

        # Fully-connected layers.
        x = Dense(units=512, activation=self.activation)(x)
        x = Dense(units=512, activation=self.activation)(x)
        x = Dense(units=512, activation=self.activation)(x)        

        mu = Dense(units=self.n_actions, activation=None, kernel_initializer=self.output_kernel)(x)
        var = Dense(units=self.n_actions, activation=None, kernel_initializer=self.output_kernel)(x)
        
        # var = tf.clip_by_value(var, self.noise, 1.0)

        model = tf.keras.models.Model(
            inputs=[img_input, target_input],
            outputs=[mu, var],
            name=self.model_name
            )

        if self.show_summary:
            model.summary()

        return model

    def sample_normal(self, state):
        mu, var = self.model(state)

        var = tf.exp(var)

        probabilities = tfp.distributions.Normal(mu, var)
        sample = probabilities.sample()

        action = tf.math.tanh(sample)
        log_probs = probabilities.log_prob(sample) - tf.math.log(1 - tf.math.pow(action,2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=-1, keepdims=True)

        return action, log_probs, mu, var

    def view_filters(self, observation, output_dir=''):

        # Get layer names.
        layer_names = [layer.name for layer in self.model.layers]

        # Get layer outputs.
        layer_outputs = [layer.output for layer in self.model.layers]

        # Create model to return feature maps.
        feature_map_model = tf.keras.models.Model(
            inputs=self.model.input, outputs=layer_outputs)
        
        feature_maps = feature_map_model.predict(observation)
        print(len(feature_maps))
    
        # Loop through layers.
        print('[INFO] Visualizing feature maps...')
        for layer_name, feature_map in zip(layer_names, feature_maps):
            if len(feature_map.shape) == 4:
                k = feature_map.shape[-1]
                
                for i in range(k):
                    feature_image = feature_map[0, :, :, i]
                    feature_image -= feature_image.mean()
                    feature_image /= feature_image.std()
                    feature_image *= 64
                    feature_image += 128
                    feature_image = np.clip(feature_image, 0, 255).astype('uint8')
                    image = Image.fromarray(feature_image)
                    image.save(f'{output_dir}/{layer_name}_f{str(i).zfill(2)}.png')

class CriticNetwork(keras.Model):
    def __init__(
        self,
        n_actions,
        input_shape,
        input_type='image_diff',
        lr=1e-4,
        noise=1e-6,
        max_step=0.1,
        name='critic',
        chkpt_dir='',
        hidden_kernel=False,
        hidden_bias=False,
        output_kernel=False,
        output_bias=False):

        super(CriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.in_shape = input_shape
        self.input_type = input_type
        self.lr = lr
        self.noise = noise
        self.max_step = max_step
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.model_name}.h5')

        # Kernel/bias initializers.
        if hidden_kernel:
            self.hidden_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_kernel = 'glorot_uniform'
        
        if hidden_bias:
            self.hidden_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_bias = 'zeros'
        
        if output_kernel:
            self.output_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_kernel = 'glorot_uniform'
        
        if output_bias:
            self.output_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_bias = 'zeros'

        self.optimizer = Adam(learning_rate=self.lr)

        self.model = self.build_model()

    def build_model(self):
        # Input layer.
        img_input = Input(shape=self.in_shape)
        target_input = Input(shape=self.in_shape)
        action_input = Input(shape=self.n_actions)
        
        if self.input_type == 'image_diff':
            input_layer = subtract([img_input, target_input])
        elif self.input_type == 'two_images':
            # Concatenate images along channel dimension.
            input_layer = concatenate([img_input, target_input], axis=-1)
        else:
            raise ValueError('input_type invalid! Must be "image_diff" or "two_images"')

        # Convolution layers.
        x = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
            padding='same', activation='LeakyReLU', kernel_initializer=self.hidden_kernel)(input_layer)
        x = MaxPool2D(padding='same')(x)
        x = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
            padding='same', activation='LeakyReLU', kernel_initializer=self.hidden_kernel)(x)

        x = Flatten()(x)
        x = Dense(units=512, activation='LeakyReLU', kernel_initializer=self.hidden_kernel)(x)
        x = Dense(units=128, activation='LeakyReLU', kernel_initializer=self.hidden_kernel)(x)
        x = Dense(units=8, activation='LeakyReLU', kernel_initializer=self.hidden_kernel)(x)
        x = concatenate([x, action_input], axis=-1)
        x = Dense(units=256, activation='LeakyReLU', kernel_initializer=self.hidden_kernel)(x)
        x = Dense(units=256, activation='LeakyReLU', kernel_initializer=self.hidden_kernel)(x)

        q = Dense(units=1, activation=None, kernel_initializer=self.output_kernel)(x)

        model = tf.keras.models.Model(
            inputs=[img_input, target_input, action_input],
            outputs=q,
            name=self.model_name
            )

        model.summary()

        return model

class AsymmetricCriticNetwork(keras.Model):
    def __init__(
        self,
        n_actions,
        input_shape,
        input_type='image_diff',
        lr=1e-4,
        noise=1e-6,
        max_step=0.1,
        name='critic',
        chkpt_dir='',
        hidden_kernel=False,
        hidden_bias=False,
        output_kernel=False,
        output_bias=False,
        show_summary=True):

        super(AsymmetricCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.input_type = input_type
        self.lr = lr
        self.noise = noise
        self.max_step = max_step
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.model_name}.h5')
        self.show_summary = show_summary

        # Kernel/bias initializers.
        if hidden_kernel:
            self.hidden_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_kernel = 'glorot_uniform'
        
        if hidden_bias:
            self.hidden_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_bias = 'zeros'
        
        if output_kernel:
            self.output_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_kernel = 'glorot_uniform'
        
        if output_bias:
            self.output_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_bias = 'zeros'

        self.optimizer = Adam(learning_rate=self.lr)

        self.model = self.build_model()

    def build_model(self):
        # Input layer.
        state_input = Input(shape=2 * self.n_actions)
        action_input = Input(shape=self.n_actions)

        x = concatenate([state_input, action_input], axis=-1)
        x = Dense(units=512, activation='relu',
            kernel_initializer=self.hidden_kernel,
            bias_initializer=self.hidden_bias,
        )(x)
        x = Dense(units=512, activation='relu',
            kernel_initializer=self.hidden_kernel,
            bias_initializer=self.hidden_bias,
        )(x)
        x = Dense(units=512, activation='relu',
            kernel_initializer=self.hidden_kernel,
            bias_initializer=self.hidden_bias,
        )(x)
        q = Dense(units=1, activation=None,
            kernel_initializer=self.output_kernel,
            bias_initializer=self.output_bias,
        )(x)

        model = tf.keras.models.Model(
            inputs=[state_input, action_input],
            outputs=q,
            name=self.model_name
            )

        if self.show_summary:
            model.summary()

        return model

#########################################################################

class CarActorNetwork(keras.Model):
    def __init__(
        self,
        n_actions,
        input_shape,
        input_type='image_diff',
        lr=1e-4,
        noise=1e-6,
        max_step=0.1,
        name='actor',
        chkpt_dir='',
        hidden_kernel=False,
        hidden_bias=False,
        output_kernel=False,
        output_bias=False):

        super(CarActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.in_shape = input_shape
        self.input_type = input_type
        self.lr = lr
        self.noise = noise
        self.max_step = max_step
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.model_name}.h5')

        # Kernel/bias initializers.
        if hidden_kernel:
            self.hidden_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_kernel = 'glorot_uniform'
        
        if hidden_bias:
            self.hidden_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_bias = 'zeros'
        
        if output_kernel:
            self.output_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_kernel = 'glorot_uniform'
        
        if output_bias:
            self.output_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_bias = 'zeros'

        self.optimizer = Adam(learning_rate=self.lr)

        self.model = self.build_model()

    def build_model(self):
        # Input layer.
        img_input = Input(shape=self.in_shape)

        x = Flatten()(img_input)
        x = Dense(units=256, activation='relu',
            kernel_initializer=self.hidden_kernel,
            bias_initializer=self.hidden_bias
        )(x)
        x = Dense(units=256, activation='relu',
            kernel_initializer=self.hidden_kernel,
            bias_initializer=self.hidden_bias
        )(x)

        mu = Dense(units=self.n_actions, activation=None,
            kernel_initializer=self.output_kernel,
            bias_initializer=self.output_bias
        )(x)
        var = Dense(units=self.n_actions, activation=None,
            kernel_initializer=self.output_kernel,
            bias_initializer=self.output_bias
        )(x)
        
        # var = tf.clip_by_value(var, self.noise, 1.0)

        model = tf.keras.models.Model(
            inputs=img_input,
            outputs=[mu, var],
            name=self.model_name
            )

        model.summary()

        return model

    def sample_normal(self, state):
        mu, var = self.model(state)

        var = tf.exp(var)

        probabilities = tfp.distributions.Normal(mu, var)
        sample = probabilities.sample()

        action = tf.math.tanh(sample)
        log_probs = probabilities.log_prob(sample) - tf.math.log(1 - tf.math.pow(action,2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=-1, keepdims=True)

        return action, log_probs, mu, var

class CarCriticNetwork(keras.Model):
    def __init__(
        self,
        n_actions,
        input_shape,
        input_type='image_diff',
        lr=1e-4,
        noise=1e-6,
        max_step=0.1,
        name='critic',
        chkpt_dir='',
        hidden_kernel=False,
        hidden_bias=False,
        output_kernel=False,
        output_bias=False):

        super(CarCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.in_shape = input_shape
        self.input_type = input_type
        self.lr = lr
        self.noise = noise
        self.max_step = max_step
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.model_name}.h5')

        # Kernel/bias initializers.
        if hidden_kernel:
            self.hidden_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_kernel = 'glorot_uniform'
        
        if hidden_bias:
            self.hidden_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.hidden_bias = 'zeros'
        
        if output_kernel:
            self.output_kernel = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_kernel = 'glorot_uniform'
        
        if output_bias:
            self.output_bias = RandomUniform(minval=-3e-3, maxval=3e-3)
        else:
            self.output_bias = 'zeros'

        self.optimizer = Adam(learning_rate=self.lr)

        self.model = self.build_model()

    def build_model(self):
        # Input layer.
        state_input = Input(shape=self.in_shape)
        action_input = Input(shape=self.n_actions)

        x = concatenate([state_input, action_input], axis=-1)
        x = Dense(units=256, activation='relu',
            kernel_initializer=self.hidden_kernel,
            bias_initializer=self.hidden_bias,
        )(x)
        x = Dense(units=256, activation='relu',
            kernel_initializer=self.hidden_kernel,
            bias_initializer=self.hidden_bias,
        )(x)

        q = Dense(units=1, activation=None,
            kernel_initializer=self.output_kernel,
            bias_initializer=self.output_bias,
        )(x)

        model = tf.keras.models.Model(
            inputs=[state_input, action_input],
            outputs=q,
            name=self.model_name
            )

        model.summary()

        return model