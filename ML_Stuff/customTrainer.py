# 3D Pose Estimation
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm
from tqdm._utils import _term_move_up
import yaml

from Models.vit_ae import ViTAE
# Private function that generates TF 2 Dataset
from Utils.generate_tf_dataset import generateTFDataset

# importing config and assigning key-value pairs as local variables
cwd = os.getcwd()

with open(f'{cwd}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

locals().update(config)


# Latent space MSE + reconstruction loss
def mseReconLoss(sv_pred, sv, decode_im, gt_im):
    mae = MeanSquaredError()(sv_pred, sv)
    recon_loss = MeanSquaredError()(decode_im, gt_im)

    return mae + recon_loss


if __name__ == '__main__':

    OPTIMIZER = tfa.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=EPSILON
    )

    NUM_STEPS = np.around(TRAIN_SIZE / BATCH_SIZE).astype('float32')
    NUM_STEPS_VAL = np.around(TEST_SIZE / BATCH_SIZE).astype('float32')

    home_dir = os.path.expanduser('~')
    DATASET_DIR = f'{home_dir}/Datasets/MHA60k'

    ds_train, ds_test, ds_seq, sv_elements = generateTFDataset(
        dataset_dir = DATASET_DIR,
        batch_size=MODEL_PARAMS['BATCH_SIZE'],
        dataset_params=DATASET_PARAMS
    )

    model = ViTAE(
        model_params=MODEL_PARAMS,
        output_shape=sv_elements,
    ).predictor_model

    train_loss_history = []
    val_loss_history = []
    best_loss = 9999
    prefix = _term_move_up() + '\r'

    for epoch in range(EPOCHS):
        print(f'Begin training epoch {epoch + 1}/{EPOCHS}...')
        training_epoch_loss = 0
        custom_training_epoch_loss = 0

        for step, (x_train_batch, y_train_batch) in enumerate(tqdm(ds_train)):
            with tf.GradientTape() as tape:
                logits, recon_im = model(x_train_batch, training=True)

                loss_value = maeReconLoss(logits, y_train_batch, recon_im, x_train_batch)
                mae_loss = MeanAbsoluteError()(logits, y_train_batch) # extra info

                training_epoch_loss += mae_loss.numpy().astype('float32')
                custom_training_epoch_loss += loss_value.numpy().astype('float32')
                
            grads = tape.gradient(loss_value, model.trainable_weights)
            OPTIMIZER.apply_gradients(zip(grads, model.trainable_weights))

            tqdm.write(prefix + f'Step {step+1}/{NUM_STEPS}: mae: {(training_epoch_loss / (step+1)):.5f}, Custom Loss: {(custom_training_epoch_loss / (step+1)):.5f}')
        
        val_epoch_loss = 0
        for step, (x_test_batch, y_test_batch) in enumerate(ds_test):
            val_logits, val_recon_im = model(x_test_batch, training=False)
            loss_value = MeanAbsoluteError()(y_test_batch, val_logits)
            val_epoch_loss += loss_value.numpy().astype('float32')
        
        train_loss_history.append(training_epoch_loss / NUM_STEPS)
        val_loss_history.append(val_epoch_loss / NUM_STEPS)
        
        print(f'Epoch {epoch+1} Training Loss: {training_epoch_loss / NUM_STEPS}')
        print(f'Epoch {epoch+1} Training Custom Loss: {custom_training_epoch_loss / NUM_STEPS}')
        print(f'Epoch {epoch+1} Validation Loss: {val_epoch_loss / NUM_STEPS_VAL}')
