import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2DTranspose, Reshape, Dense, Input, MultiHeadAttention, LayerNormalization


class ViTAE:
    def __init__(self,
                 model_params={
                    'INPUT_SHAPE': (512, 512, 3),
                    'ACTIVATION': 'gelu',
                    'LEARNING_RATE': 1e-4,
                    'EPSILON': 1e-5,
                    'OPTIMIZER': 'adamw',
                    'ATTENTION_LAYERS': 6,
                    'PATCH_SIZE': 64,
                    'NUM_HEADS': 16,
                    'PROJ_DIMS': 512,
                    'KEY_DIMS': 64,
                    'DROPOUT': 0.05,
                    'BATCH_SIZE': 32
                 },
                 output_shape=9,
                 show_summary=True,
                 ):
        self.in_shape = model_params['INPUT_SHAPE']
        self.out_shape = output_shape
        self.attention_layers = model_params['ATTENTION_LAYERS']
        self.patch_size = model_params['PATCH_SIZE']
        self.num_heads = model_params['NUM_HEADS']
        self.proj_dims = model_params['PROJ_DIMS']
        self.key_dims = model_params['KEY_DIMS']
        self.dropout = model_params['DROPOUT']
        self.activation = model_params['ACTIVATION']
        self.decay = model_params['EPSILON']
        self.batch_size = model_params['BATCH_SIZE']

        self.cls_token = tf.Variable(
            initial_value=tf.zeros(
                shape=(1, 1, self.proj_dims),
                dtype=tf.float32
            ),
            trainable=True
        )

        self.predictor_model = self.build_model()

        if show_summary:
            self.predictor_model.summary()


    def build_model(self):
        input_layer = Input(shape=self.in_shape)
        x = input_layer

        if self.attention_layers > 0:
            patch_size = self.patch_size
            num_patches = (self.in_shape[0] // patch_size) * (self.in_shape[1] // patch_size)
            patches = Patches(patch_size)(x)
            x = PatchEncoder(
                num_patches=num_patches,
                projection_dim=self.proj_dims,
                activation=None)(patches, self.cls_token)

            for _ in range(self.attention_layers):
                x = self.buildAttentionBlock(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = Dense(units=self.out_shape)(x)
        sv = x

        x = Dense(units=256 * 64)(x)
        x = Reshape(target_shape=(16, 16, 64))(x)

        x = Conv2DTranspose(filters=64, kernel_size=3, padding='same', strides=(2,2), activation=self.activation)(x)
        x = Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=(2,2), activation=self.activation)(x)
        x = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2,2), activation=self.activation)(x)
        x = Conv2DTranspose(filters=8, kernel_size=3, padding='same', strides=(2,2), activation=self.activation)(x)
        x = Conv2DTranspose(filters=4, kernel_size=3, padding='same', strides=(2,2), activation=self.activation)(x)

        x = Conv2DTranspose(filters=3, kernel_size=3, padding='same')(x)
        recon_img = x

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[sv, recon_img],
            name='AttentionModel'
            )

        return model


    def buildAttentionBlock(self, x):
        shortcut = x

        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dims, dropout=0)(x, x)

        x = Add()([x, shortcut])
        x = LayerNormalization(epsilon=self.decay)(x)
        x = Dense(self.proj_dims, activation=self.activation)(x)

        return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        num_patches = patches.shape[-2] * patches.shape[-3]
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])

        return patches

    def get_config(self):
        cfg = super().get_config()
        return cfg    


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, activation=None, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=self.projection_dim, activation=activation)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches+1, output_dim=projection_dim
        )


    def call(self, patches, cls_token):
        patches = self.projection(patches)
        cls_token = RepeatLayer()(cls_token, patches)
        patches = tf.keras.layers.Concatenate(axis=1)([cls_token, patches])
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        encoded = patches + self.position_embedding(positions)
        return encoded


    def get_config(self):
        cfg = super().get_config()
        return cfg


class RepeatLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(RepeatLayer, self).__init__()


    def call(self, class_token, patches):
        shapes = tf.shape(patches)

        return tf.repeat(class_token, shapes[0], axis=0)


if __name__ == '__main__':
    model = ViTAE().build_model()
