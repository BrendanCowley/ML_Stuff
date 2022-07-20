import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, strategy, K, her_ratio):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.her_mem_cntr = 0
        self.episode_final_ind = []
        self.strategy = strategy
        self.K = K
        self.her_ratio = her_ratio

        self.memory = {
            'states': np.zeros((self.mem_size, 2, *input_shape)),
            'new_states': np.zeros((self.mem_size, 2, *input_shape)),
            'svs': np.zeros((self.mem_size, 2 * n_actions)),
            'new_svs': np.zeros((self.mem_size, 2 * n_actions)),
            'actions': np.zeros((self.mem_size, n_actions)),
            'dones': np.zeros(self.mem_size, dtype=np.bool),
            'rewards': np.zeros((self.mem_size))
        }

        self.her_memory = {
            'goals': np.zeros((self.mem_size * self.K, *input_shape)),
            'states': np.zeros((self.mem_size * self.K, 2, *input_shape)),
            'new_states': np.zeros((self.mem_size * self.K, 2, *input_shape)),
            'svs': np.zeros((self.mem_size * self.K, 2 * n_actions)),
            'new_svs': np.zeros((self.mem_size * self.K, 2 * n_actions)),
            'actions': np.zeros((self.mem_size * self.K, n_actions)),
            'dones': np.zeros(self.mem_size * self.K, dtype=np.bool),
            'rewards': np.zeros((self.mem_size * self.K))
        }

    def store_transition(self, state, action, reward, new_state, done, sv, new_sv):
        ind = self.mem_cntr % self.mem_size

        self.memory['states'][ind][0] = state[0]
        self.memory['states'][ind][1] = state[1]
        self.memory['new_states'][ind][0] = new_state[0]
        self.memory['new_states'][ind][1] = new_state[1]
        self.memory['svs'][ind] = sv
        self.memory['new_svs'][ind] = new_sv
        self.memory['actions'][ind] = action
        self.memory['rewards'][ind] = reward
        self.memory['dones'][ind] = done

        if done:
            self.episode_final_ind.append(ind)
            
            if self.strategy.lower() == 'future':
                self.store_her_transition()
                self.her_mem_cntr = (self.mem_cntr + 1) * self.K
            elif self.strategy.lower() == 'final':
                self.store_her_transition_final()
                self.her_mem_cntr = self.mem_cntr + 1
            elif self.strategy.lower() == 'episode':
                self.store_her_transition_episode()
                self.her_mem_cntr = (self.mem_cntr + 1) * self.K
            else:
                raise ValueError(f'Sampling strategy {self.strategy} not supported.')

        self.mem_cntr += 1

    def store_her_transition_final(self):
        if len(self.episode_final_ind) > 1:
            start_ind = self.episode_final_ind[-2] + 1
            end_ind = self.episode_final_ind[-1]
        else:
            start_ind = 0
            end_ind = self.episode_final_ind[-1]

        if start_ind > end_ind:
            end_ind += self.mem_size

        ind = self.her_mem_cntr % (self.mem_size)

        for i in range(start_ind, end_ind + 1):
            if i >= self.mem_size:
                iter = i - self.mem_size
            else:
                iter = i

            cur_ind = ind + (i - start_ind)

            if cur_ind >= self.mem_size:
                cur_ind -= self.mem_size

            future = end_ind

            if future >= self.mem_size:
                future -= self.mem_size

            goal = self.memory['new_states'][future]
            state = self.memory['states'][iter]
            new_state = self.memory['new_states'][iter]
            sv = self.memory['svs'][iter]
            new_sv = self.memory['new_svs'][iter]
            action = self.memory['actions'][iter]
            done = np.array_equal(new_state, goal)
            reward = 100 if done else self.memory['rewards'][iter]
            self.her_memory['states'][cur_ind] = state
            self.her_memory['new_states'][cur_ind] = new_state
            self.her_memory['svs'][cur_ind] = sv
            self.her_memory['new_svs'][cur_ind] = new_sv
            self.her_memory['actions'][cur_ind] = action
            self.her_memory['rewards'][cur_ind] = reward
            self.her_memory['dones'][cur_ind] = done

    def store_her_transition(self):
        if len(self.episode_final_ind) > 1:
            start_ind = self.episode_final_ind[-2] + 1
            end_ind = self.episode_final_ind[-1]
        else:
            start_ind = 0
            end_ind = self.episode_final_ind[-1]

        if start_ind > end_ind:
            end_ind += self.mem_size

        ind = self.her_mem_cntr % (self.mem_size * self.K)

        for i in range(start_ind, end_ind + 1):
            for k in range(self.K):
                if i >= self.mem_size:
                    iter = i - self.mem_size
                else:
                    iter = i

                cur_ind = ind + (i - start_ind) * self.K + k

                if cur_ind >= self.mem_size * self.K:
                    cur_ind -= self.mem_size * self.K

                future = np.random.randint(iter, end_ind + 1)

                if future >= self.mem_size:
                    future -= self.mem_size

                goal = self.memory['new_states'][future]
                state = self.memory['states'][iter]
                new_state = self.memory['new_states'][iter]
                sv = self.memory['svs'][iter]
                new_sv = self.memory['new_svs'][iter]
                action = self.memory['actions'][iter]
                done = np.array_equal(new_state, goal)
                reward = 100 if done else self.memory['rewards'][iter]
                # reward = 100 if done else np.abs(100 - self.memory['rewards'][iter])
                # reward = 100

                self.her_memory['states'][cur_ind][0] = state[0]
                self.her_memory['states'][cur_ind][1] = state[1]
                self.her_memory['new_states'][cur_ind][0] = new_state[0]
                self.her_memory['new_states'][cur_ind][1] = new_state[1]
                self.her_memory['svs'][cur_ind] = sv
                self.her_memory['new_svs'][cur_ind] = new_sv
                self.her_memory['actions'][cur_ind] = action
                self.her_memory['rewards'][cur_ind] = reward
                self.her_memory['dones'][cur_ind] = done

    def store_her_transition_episode(self):
        if len(self.episode_final_ind) > 1:
            start_ind = self.episode_final_ind[-2] + 1
            end_ind = self.episode_final_ind[-1]
        else:
            start_ind = 0
            end_ind = self.episode_final_ind[-1]

        if start_ind > end_ind:
            end_ind += self.mem_size

        ind = self.her_mem_cntr % (self.mem_size * self.K)

        for i in range(start_ind, end_ind + 1):
            for k in range(self.K):
                if i >= self.mem_size:
                    iter = i - self.mem_size
                else:
                    iter = i

                cur_ind = ind + (i - start_ind) * self.K + k

                if cur_ind >= self.mem_size * self.K:
                    cur_ind -= self.mem_size * self.K

                future = np.random.randint(start_ind, end_ind + 1)

                if future >= self.mem_size:
                    future -= self.mem_size

                goal = self.memory['new_states'][future]
                state = self.memory['states'][iter]
                new_state = self.memory['new_states'][iter]
                sv = self.memory['svs'][iter]
                new_sv = self.memory['new_svs'][iter]
                action = self.memory['actions'][iter]
                done = np.array_equal(new_state, goal)
                reward = 100 if done else self.memory['rewards'][iter]
                # reward = 100 if done else np.abs(100 - self.memory['rewards'][iter])
                # reward = 100

                self.her_memory['states'][cur_ind] = state
                self.her_memory['new_states'][cur_ind] = new_state
                self.her_memory['svs'][cur_ind] = sv
                self.her_memory['new_svs'][cur_ind] = new_sv
                self.her_memory['actions'][cur_ind] = action
                self.her_memory['rewards'][cur_ind] = reward
                self.her_memory['dones'][cur_ind] = done

    def sample_buffer(self, batch_size):
        if self.her_mem_cntr > 0:
            max_mem = min(self.mem_cntr, self.mem_size)
            her_max_mem = min(self.her_mem_cntr, self.mem_size * self.K)

            her_batch =  np.random.choice(her_max_mem, int(batch_size * self.her_ratio))

            batch = np.random.choice(max_mem, batch_size - int(batch_size * self.her_ratio))

            states = self.memory['states'][batch]
            new_states = self.memory['new_states'][batch]
            svs = self.memory['svs'][batch]
            new_svs = self.memory['new_svs'][batch]
            actions = self.memory['actions'][batch]
            rewards = self.memory['rewards'][batch]
            dones = self.memory['dones'][batch]

            states = np.squeeze(states)
            new_states = np.squeeze(new_states)
            svs = np.squeeze(svs)
            new_svs = np.squeeze(new_svs)
            actions = np.squeeze(actions)
            rewards = np.squeeze(rewards)
            dones = np.squeeze(dones)

            her_states = self.her_memory['states'][her_batch]
            her_new_states = self.her_memory['new_states'][her_batch]
            her_svs = self.her_memory['svs'][her_batch]
            her_new_svs = self.her_memory['new_svs'][her_batch]
            her_actions = self.her_memory['actions'][her_batch]
            her_rewards = self.her_memory['rewards'][her_batch]
            her_dones = self.her_memory['dones'][her_batch]

            her_states = np.squeeze(her_states)
            her_new_states = np.squeeze(her_new_states)
            her_svs = np.squeeze(her_svs)
            her_new_svs = np.squeeze(her_new_svs)
            her_actions = np.squeeze(her_actions)
            her_rewards = np.squeeze(her_rewards)
            her_dones = np.squeeze(her_dones)

            states = np.concatenate((states, her_states), axis=0)
            np.random.shuffle(states)
            new_states = np.concatenate((new_states, her_new_states), axis=0)
            np.random.shuffle(new_states)
            svs = np.concatenate((svs, her_svs), axis=0)
            np.random.shuffle(svs)
            new_svs = np.concatenate((new_svs, her_new_svs), axis=0)
            np.random.shuffle(new_svs)
            actions = np.concatenate((actions, her_actions), axis=0)
            np.random.shuffle(actions)
            rewards = np.concatenate((rewards, her_rewards), axis=0)
            np.random.shuffle(rewards)
            dones = np.concatenate((dones, her_dones), axis=0)
            np.random.shuffle(dones)
        else:
            max_mem = min(self.mem_cntr, self.mem_size)

            batch = np.random.choice(max_mem, batch_size)

            states = self.memory['states'][batch]
            new_states = self.memory['new_states'][batch]
            svs = self.memory['svs'][batch]
            new_svs = self.memory['new_svs'][batch]
            actions = self.memory['actions'][batch]
            rewards = self.memory['rewards'][batch]
            dones = self.memory['dones'][batch]

            states = np.squeeze(states)
            new_states = np.squeeze(new_states)
            svs = np.squeeze(svs)
            new_svs = np.squeeze(new_svs)
            actions = np.squeeze(actions)
            rewards = np.squeeze(rewards)
            dones = np.squeeze(dones)
        
        # print(f'HER positive reward samples: {np.where(self.her_memory["rewards"] > 0)}')
        # print(f'Positive reward observations: {np.where(rewards > 0)}')

        return \
            [tf.convert_to_tensor(states[:, 0], dtype=tf.float32), tf.convert_to_tensor(states[:, 1], dtype=tf.float32)], \
            tf.convert_to_tensor(actions, dtype=tf.float32), \
            tf.convert_to_tensor(rewards, dtype=tf.float32),\
            [tf.convert_to_tensor(new_states[:, 0], dtype=tf.float32), tf.convert_to_tensor(new_states[:, 1], dtype=tf.float32)],\
            dones,\
            tf.convert_to_tensor(svs, dtype=tf.float32), \
            tf.convert_to_tensor(new_svs, dtype=tf.float32)