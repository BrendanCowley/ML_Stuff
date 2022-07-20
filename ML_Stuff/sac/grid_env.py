from gym import Env
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import yaml

# Custom imports.
from RL.HeadGame_TwoImage_Continuous.scene1 import renderSceneData
from Utils.vector_conversions import learnToRender

# Dot Game Environment.
class GridImageEnv(Env):
    def __init__(
        self,
        sv_elems=9,
        render_elems=7,
        img_shape=(512, 512, 3),
        num_moves=1000,
        max_step=0.001,
        step_scale=0.90,
        start_pos='fixed',
        mae_threshold=0,
        blend_sources=[], 
        dataset_dir='',
        step_size_schedule=False, 
        mae_step_threshold=None,
        render_bool=False,
        print_info=False,
        reward_scale=False,
        def_instance=None,
        learn_list=None):
        
        self.sv_elems = sv_elems
        self.render_elems = render_elems
        self.img_shape = img_shape
        self.max_step = max_step
        self.step_scale = step_scale
        self.start_pos = start_pos
        self.blend_sources = blend_sources
        self.dataset_dir = dataset_dir
        self.step_size_schedule = step_size_schedule
        self.mae_step_threshold = mae_step_threshold
        self.render_bool = render_bool
        self.print_info = print_info
        self.reward_scale = reward_scale
        self.step_reduced = False
        self.step_size = self.max_step

        if self.print_info:
            print('[INFO] Initializing game environment...')

        # Start client.
        if not def_instance:
            self.def_instance, self.learn_list = renderSceneData(source_list=self.blend_sources, dataset_dir=self.dataset_dir)
            self.def_instance.start_local_blender()
        else:
            self.def_instance = def_instance
            self.learn_list = learn_list

        # Generate starting scene vector.
        if self.start_pos == 'fixed':
            self.current_learning_sv = np.zeros(shape=(1, self.sv_elems), dtype=np.uint8)
        elif self.start_pos == 'random':
            self.current_learning_sv = np.random.uniform(-1, 1, size=(1, self.sv_elems))
        else:
            raise ValueError(f'start_pos must be "fixed", or "random". Received: {self.start_pos}')
        
        # Generate target scene vector.
        self.target_learning_sv = np.random.uniform(-1, 1, size=(1, self.sv_elems))

        # Verify current and target sv's are not equal.
        while (self.target_learning_sv == self.current_learning_sv).all():
            self.target_learning_sv = np.random.uniform(-1, 1, size=(1, self.sv_elems))
        
        self.svs = np.concatenate((self.current_learning_sv, self.target_learning_sv), axis=-1)
        
        # Convert learning vectors to render vectors.
        self.current_render_sv, _ = learnToRender(self.current_learning_sv, dataset_dir=self.dataset_dir)
        self.target_render_sv, _ = learnToRender(self.target_learning_sv, dataset_dir=self.dataset_dir)
        
        # Render current and target images.
        self.current_img = self.def_instance.render_vector_to_buffer_local(
            vector=self.current_render_sv, img_w=self.img_shape[0], img_h=self.img_shape[1])
        self.target_img = self.def_instance.render_vector_to_buffer_local(
            vector=self.target_render_sv, img_w=self.img_shape[0], img_h=self.img_shape[1])
        
        # Remove alpha channels.
        self.current_img = self.current_img[:, :, :, :-1]
        self.target_img = self.target_img[:, :, :, :-1]

        # Calculate Euclidean distance of learning vectors.
        self.dist = np.linalg.norm((self.target_learning_sv - self.current_learning_sv))

        # Initialize score.
        self.score = 0

        # Initialize done bool.
        self.done = False

        self.win = 0

        # Take img difference.
        self.state = self.current_img - self.target_img
        self.mae = np.mean(np.abs(self.state))
        self.best_mae = self.mae

        # Initialize best score.
        self.start_mae = self.mae
        self.best_mae = 9999
        
        # Set maximum number of moves and available moves.
        self.max_moves = num_moves
        self.num_moves = self.max_moves

        # Set MAE threhold for finishing game.
        self.mae_threshold = mae_threshold

        # Distribution plot x-axis.
        self.x = np.linspace(-10, 10, 1000)
        self.dist_labels = ['x', 'y']

        if self.render_bool:
            plt.ion()

            self.fig = plt.figure(figsize=(6, 3))
            self.ax1 = plt.subplot(1, 2, 1)
            self.im1 = self.ax1.imshow(self.state[0])
            self.ax1.set_title(f'Moves Left: {self.num_moves}, Score: {self.score}')
            
            self.ax2 = plt.subplot(1, 2, 2)
            self.ax2.set_title(f'Action Probabilities')
            self.ax2.set_xlim(-1.2, 1.2)
            self.ax2.set_ylim(0, 1.05)

        if self.print_info:
            print('[INFO] Game environment initialized!')

    def step(self, action):

        # Update states with action.
        self.current_learning_sv = self.current_learning_sv + action * self.max_step
        self.sv_render_state, _ = learnToRender(self.current_learning_sv, dataset_dir=self.dataset_dir)
        
        # Concatenate.
        self.svs = np.concatenate((self.current_learning_sv, self.target_learning_sv), axis=-1)

        new_mae = np.mean(np.abs(self.current_learning_sv - self.target_learning_sv))

        # Render new img.
        new_img = self.def_instance.render_vector_to_buffer_local(
            vector=self.sv_render_state, img_w=self.img_shape[0], img_h=self.img_shape[1])
        
        # Remove alpha channel.
        new_img = new_img[:, :, :, :-1]

        # Get new state.
        new_state = new_img - self.target_img

        # Reduce number of available moves.
        self.num_moves -= 1

        # Define rewards.
        mae_new = np.mean(np.abs(new_state))

        # -1 for step.
        reward = -1

        # Update step scale.
        if self.step_size_schedule and self.step_size > 0.05:
            self.step_size *= self.step_scale

        # Win state.
        if mae_new < self.mae_threshold:
            reward += 100
            self.done = True
            self.win = 1
            # if self.mae_threshold > 0.009:
            #     self.mae_threshold *= 0.96
        
        # Game over state.
        if (np.abs(self.current_learning_sv[0]) > 1.5).any():
            reward -= 100
            self.done = True
        if self.num_moves <= 0:
            self.done = True
                 
        # Update score.
        self.score += reward
        
        # Update img state.
        self.current_img = new_img
        self.state = new_state
        self.mae = mae_new

        # Update best MAE if beaten.
        if self.mae < self.best_mae:
            self.best_mae = self.mae
        
        # Set placeholder for info.
        info = {'win': self.win}
        
        if self.print_info:
            print(f'Current SV State: {self.current_learning_sv}')
            print(f'Step Size: {self.step_size}')
            print(f'Curent MAE: {self.mae}')
        
        self.current_img = tf.convert_to_tensor(self.current_img)
        self.target_img = tf.convert_to_tensor(self.target_img)
        self.state = [self.current_img, self.target_img]

        return self.state, self.svs, reward, self.done, info
    
    def render(self, action_mus, action_vars):
        action_mus = tf.squeeze(action_mus)
        action_vars = tf.squeeze(action_vars)
        # Real-time visualization.
        self.im1.set_data(np.abs(self.current_img[0] - self.target_img[0]))
        self.fig.canvas.flush_events()
        self.ax1.set_title(f'Moves Left: {self.num_moves}, Score: {self.score}')
        self.ax2.clear()
        self.ax2.set_title('Action Distributions')

        for i in range(self.sv_elems):
            if i == 1:
                action_dist = scipy.stats.norm(
                    action_mus.numpy()[i], action_vars.numpy()[i])

                self.ax2.plot(self.x, action_dist.pdf(self.x) + 1e-16,
                    label=self.dist_labels[i], alpha=0.5)
                
                if np.max(action_dist.pdf(self.x)) < 0.0:
                    print(f'Variance: {action_vars[i]}')
                    print(f'Mean: {action_mus[i]}')
                    print(f'Maximum density: {np.max(action_dist.pdf(self.x))}')
                    print(f'Distribution Sum: {np.sum(np.max(action_dist.pdf(self.x)))}')
                    raise ValueError('Value out of range for probability distribution calculation!')
            else:
                action_dist = scipy.stats.norm(
                    action_mus.numpy()[i] * -1, action_vars.numpy()[i])
                self.ax2.plot(self.x, action_dist.pdf(self.x) + 1e-16,
                    label=self.dist_labels[i], alpha=0.5)

        self.ax2.legend()
        self.ax2.set_xlim(-2, 2)
        # self.ax2.set_yscale('log')
        self.ax2.set_ylim(0,1.5)
        self.ax2.set_xlabel('Action')
        self.ax2.set_ylabel('p')

    def reset(self):
        # Reset starting state.
        if self.print_info:
            print('[INFO] Restarting game!')
        # Generate starting scene vector.
        if self.start_pos == 'fixed':
            self.current_learning_sv = np.zeros(shape=(1, self.sv_elems), dtype=np.uint8)
        elif self.start_pos == 'random':
            self.current_learning_sv = np.random.uniform(-1, 1, size=(1, self.sv_elems))
        else:
            raise ValueError(f'start_pos must be "fixed", or "random". Received: {self.start_pos}')
        
        # Generate target scene vector.
        self.target_learning_sv = np.random.uniform(-1, 1, size=(1, self.sv_elems))

        # Verify current and target sv's are not equal.
        while (self.target_learning_sv == self.current_learning_sv).all():
            self.target_learning_sv = np.random.uniform(-1, 1, size=(1, self.sv_elems))
        
        # Concatenate.
        self.svs = np.concatenate((self.current_learning_sv, self.target_learning_sv), axis=-1)

        # Convert learning vectors to render vectors.
        self.current_render_sv, _ = learnToRender(self.current_learning_sv, dataset_dir=self.dataset_dir)
        self.target_render_sv, _ = learnToRender(self.target_learning_sv, dataset_dir=self.dataset_dir)
        
        # Render current and target images.
        self.current_img = self.def_instance.render_vector_to_buffer_local(
            vector=self.current_render_sv, img_w=self.img_shape[0], img_h=self.img_shape[1])
        self.target_img = self.def_instance.render_vector_to_buffer_local(
            vector=self.target_render_sv, img_w=self.img_shape[0], img_h=self.img_shape[1])
        
        # Remove alpha channels.
        self.current_img = self.current_img[:, :, :, :-1]
        self.target_img = self.target_img[:, :, :, :-1]

        # Calculate Euclidean distance of learning vectors.
        self.dist = np.linalg.norm((self.target_learning_sv - self.current_learning_sv))

        self.score = 0
        self.step_size = self.max_step
        self.step_reduced = False

        self.done = False
        self.win = 0

        # Take img difference.
        self.state = self.current_img - self.target_img
        self.mae = np.mean(np.abs(self.state))
        self.best_mae = self.mae

        # Initialize best score.
        self.start_mae = self.mae
        self.best_mae = 9999
        
        # Set maximum number of moves and available moves.
        self.num_moves = self.max_moves

        # Real-time visualization.
        if self.render_bool:
            self.im1.set_data(np.abs(self.state[0]))
            self.fig.canvas.flush_events()
            self.im1.set_data(np.abs(self.state[0]))
            self.fig.canvas.flush_events()
            self.ax1.set_title(f'Moves Left: {self.num_moves}, Score: {self.score}')
            self.ax2.set_title(f'Action Probabilities')
            self.ax2.clear()
        
        self.current_img = tf.convert_to_tensor(self.current_img)
        self.target_img = tf.convert_to_tensor(self.target_img)
        self.state = [self.current_img, self.target_img]

        return self.state, self.svs 

    def close_im(self):
        plt.close()

if __name__ == "__main__":

    # Load config.
    with open(f'RL/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    locals().update(config)
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    env = GridImageEnv(
        sv_elems=SV_ELEMS, render_elems=RENDER_ELEMS,
        img_shape=IMG_SHAPE, num_moves=NUM_MOVES, max_step=MAX_STEP,
        mae_threshold=MAE_THRESHOLD, start_pos='fixed', render_bool=True,
        blend_sources=BLEND_SOURCES, dataset_dir=DATASET_DIR)
    
    for _ in range(20):
        random_action = np.random.randint(0,2)
        if random_action == 0:
            action = np.array([[1,0]])
        else:
            action = np.array([[0,1]])
        _, reward, _, _ = env.step(action)
        print(f'Reward: {reward}')

    print('Finished!')