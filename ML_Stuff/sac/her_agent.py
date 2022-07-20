import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from RL.SAC_V2.her_memory_v2 import ReplayBuffer
from RL.SAC_V2.networks import ActorNetwork, CriticNetwork, AsymmetricCriticNetwork


class Agent:
    def __init__(
        self,
        input_shape=(512, 512, 3),
        alpha=0.0003,
        beta=0.0003,
        max_step=0.1,
        gamma=0.99,
        temperature=1.0,
        n_actions=2,
        max_size=1000,
        tau=0.005,
        batch_size=256,
        reward_scale=2,
        output_dir='',
        input_type='image_diff',
        strategy='future',
        her_ratio=0.8,
        hidden_kernel=True,
        hidden_bias=False,
        output_kernel=True,
        output_bias=False,
        seed=None,
        skip=False,
        activation='LeakyReLU',
        show_summary=True
    ):
        self.gamma = gamma
        self.log_temp = tf.Variable(np.log(temperature * np.ones(shape=(n_actions,))), trainable=True, dtype=tf.float32)
        self.temperature = tf.exp(self.log_temp)
        self.temperature = 1
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_shape, n_actions, K=4, her_ratio=her_ratio, strategy=strategy)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.temp_opt = tf.keras.optimizers.Adam(lr=alpha)

        # Set seed if provided.
        if seed:
            tf.random.set_seed(seed)

        self.actor = ActorNetwork(
            n_actions=n_actions,
            input_shape=input_shape,
            name='actor',
            lr=alpha,
            input_type=input_type,
            max_step=max_step,
            chkpt_dir=output_dir,
            hidden_kernel=hidden_kernel,
            hidden_bias=hidden_bias,
            output_kernel=output_kernel,
            output_bias=output_bias,
            skip=skip,
            activation=activation,
            show_summary=show_summary
        )

        self.critic_1 = AsymmetricCriticNetwork(
            n_actions=n_actions,
            input_shape=input_shape,
            name='critic_1',
            lr=beta,
            input_type=input_type,
            chkpt_dir=output_dir,
            hidden_kernel=hidden_kernel,
            hidden_bias=hidden_bias,
            output_kernel=output_kernel,
            output_bias=output_bias,
            show_summary=show_summary
        )

        self.critic_2 = AsymmetricCriticNetwork(
            n_actions=n_actions,
            input_shape=input_shape,
            name='critic_2',
            lr=beta,
            input_type=input_type,
            chkpt_dir=output_dir,
            hidden_kernel=hidden_kernel,
            hidden_bias=hidden_bias,
            output_kernel=output_kernel,
            output_bias=output_bias,
            show_summary=show_summary
        )

        self.target_critic_1 = AsymmetricCriticNetwork(
            n_actions=n_actions,
            input_shape=input_shape,
            name='target_critic_1',
            lr=beta,
            input_type=input_type,
            chkpt_dir=output_dir,
            hidden_kernel=hidden_kernel,
            hidden_bias=hidden_bias,
            output_kernel=output_kernel,
            output_bias=output_bias,
            show_summary=show_summary
        )

        self.target_critic_2 = AsymmetricCriticNetwork(
            n_actions=n_actions,
            input_shape=input_shape,
            name='target_critic_2',
            lr=beta,
            input_type=input_type,
            chkpt_dir=output_dir,
            hidden_kernel=hidden_kernel,
            hidden_bias=hidden_bias,
            output_kernel=output_kernel,
            output_bias=output_bias,
            show_summary=show_summary
        )

        self.target_entropy = -n_actions
        self.scale = reward_scale
        self.update_network_parameters(tau=1)


    def choose_action(self, observation):
        actions, log_probs, mu, var = self.actor.sample_normal(observation)

        return actions, mu, var

    def remember(self, state, action, reward, new_state, done, sv, new_sv):
        self.memory.store_transition(state, action, reward, new_state, done, sv, new_sv)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights_1 = []
        targets_1 = self.target_critic_1.weights

        for i, weight in enumerate(self.critic_1.weights):
            weights_1.append(weight * tau + targets_1[i]*(1-tau))
        
        weights_2 = []
        targets_2 = self.target_critic_2.weights

        for i, weight in enumerate(self.critic_2.weights):
            weights_2.append(weight * tau + targets_2[i]*(1-tau))

        self.target_critic_1.set_weights(weights_1)
        self.target_critic_2.set_weights(weights_2)

    def save_models(self):
        print('... saving models ...')
        out_dir = os.path.join(os.path.expanduser("~"), 'Datasets/RL/Output')
        self.actor.model.save_weights(os.path.join(out_dir, self.actor.checkpoint_file))
        self.critic_1.save_weights(os.path.join(out_dir, self.critic_1.checkpoint_file))
        self.critic_2.save_weights(os.path.join(out_dir, self.critic_2.checkpoint_file))
        self.target_critic_1.save_weights(os.path.join(out_dir, self.target_critic_1.checkpoint_file))
        self.target_critic_2.save_weights(os.path.join(out_dir, self.target_critic_2.checkpoint_file))

    def load_models(self):
        print('... loading models ...')
        out_dir = os.path.join(os.path.expanduser("~"), 'Datasets/RL/Output')
        self.actor.load_weights(os.path.join(out_dir, self.actor.checkpoint_file))
        self.critic_1.load_weights(os.path.join(out_dir, self.critic_1.checkpoint_file))

    def load_actor(self, path):
        self.actor.model.load_weights(path)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones, svs, new_svs = \
                self.memory.sample_buffer(self.batch_size)

        critic_inputs = [svs] + [actions]

        new_policy_actions, new_log_probs, _, _ = self.actor.sample_normal(new_states)
        new_critic_inputs = [new_svs] + [new_policy_actions]

        new_q1 = self.target_critic_1.model(new_critic_inputs)
        new_q2 = self.target_critic_2.model(new_critic_inputs)

        min_q_next = tf.minimum(new_q1, new_q2) - self.temperature * new_log_probs

        q_hat = rewards + (1 - dones) * self.gamma * min_q_next
        
        # Train critic network.
        with tf.GradientTape() as tape:
            curr_q_1 = self.critic_1.model(critic_inputs)

            critic_loss = keras.losses.MSE(curr_q_1, q_hat)

        critic_grad = tape.gradient(critic_loss, self.critic_1.model.trainable_variables)
        self.critic_1.optimizer.apply_gradients(
            zip(critic_grad, self.critic_1.model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            curr_q_2 = self.critic_2.model(critic_inputs)

            critic_loss = keras.losses.MSE(curr_q_2, q_hat)

        critic_grad = tape.gradient(critic_loss, self.critic_2.model.trainable_variables)
        self.critic_2.optimizer.apply_gradients(
            zip(critic_grad, self.critic_2.model.trainable_variables)
        )

        # Train actor network
        with tf.GradientTape() as tape:
        
            policy_actions, log_probs, _, _ = self.actor.sample_normal(states)

            critic_inputs = [svs, policy_actions]
            q_1 = self.critic_1.model(critic_inputs)
            q_2 = self.critic_2.model(critic_inputs)
            min_q = tf.minimum(q_1, q_2)

            actor_loss = tf.reduce_mean(self.temperature * log_probs - min_q)

        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.model.trainable_variables)
        )

        # Train entropy.
        inner_prod = (log_probs + self.target_entropy)
        with tf.GradientTape() as tape:
            entropy_loss = -tf.reduce_mean(self.log_temp * inner_prod)

        entropy_grad = tape.gradient(entropy_loss, self.log_temp)
        self.temp_opt.apply_gradients(zip([entropy_grad], [self.log_temp]))
        self.temperature = tf.reduce_mean(tf.exp(self.log_temp))

        self.update_network_parameters()