import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


class ReplayMemory_MPC:
    def __init__(self,size):
        df = pd.read_excel(r'C:\Users\Shiva\Desktop\RL_FRE\Time-varying-Setpoint-Tracking-for-Batch-Process-Control-using-Reinforcement-Learning-1\DQN_Implementation\Sample_data.xlsx')
        # Convert each column of the DataFrame into a NumPy array
        self.np_arrays = [df[col].to_numpy().reshape(-1, 1) for col in df.columns]
        # Stack the NumPy arrays horizontally
        self.memory_mpc = np.hstack(self.np_arrays)
        self.size_mpc = size
    def get_batch(self,size):
        indexes = np.random.choice(range(0,self.size_mpc), size=size, replace=False)
        return self.memory_mpc[indexes]

class ReplayMemory:
    def __init__(self, max_size, num_cols):
        self.memory = np.zeros((max_size, num_cols))
        self.size = 0
        self.set_size = max_size
        self.curr_index = 0
        self.is_full = False
    def push(self, item):
        if self.size >= self.set_size and self.is_full == False:
            self.is_full = True
            self.curr_index = 0
        elif self.curr_index >= self.set_size and self.is_full == True:
            self.curr_index = 0
        elif self.size < self.set_size:
            self.size += 1
        self.memory[self.curr_index] = item
        self.curr_index += 1
    def get_batch(self, size):
        end_index = self.set_size if self.is_full else self.curr_index
        indexes = np.random.choice(range(0, end_index), size=size, replace=False)
        return self.memory[indexes]

class DQNAgent:
    
    def __init__(self, environment, learning_rate=1e-3, decay_rate=1e-4, discount_factor=1, epsilon=0.05, batch_size=20, replay_memory_size= 50, nn_arch=[400, 300, 200], reset_steps=9):
        self.env = environment
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = ReplayMemory(replay_memory_size, 2*self.env.state_dim + self.env.action_dim + 1)
        self.Q = self._get_model(nn_arch)
        self.Q_t = self._get_model(nn_arch)
        self.Q_t.set_weights(self.Q.get_weights())
        self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate, decay=decay_rate)
        self.reset_steps = reset_steps
        self.state_index = np.arange(0, self.env.state_dim, 1, dtype=np.int16)
        self.action_index = np.arange(self.state_index[-1]+1, self.state_index[-1]+1+self.env.action_dim, 1, dtype=np.int16)
        self.next_state_index = np.arange(self.action_index[-1]+1, self.action_index+1+self.env.state_dim, 1, dtype=np.int16)
        self.reward_index = self.next_state_index[-1] + 1
        self.memory_mpc = ReplayMemory_MPC(200)
        
    def _get_model(self, nn_arch):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nn_arch[0], input_shape=(4,), activation="relu"))
        for num_neurons in nn_arch[1:]:
            model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.num_j_temp, activation="linear"))
        model.build()
        return model
    
    @tf.function
    def greedy_policy(self, state):
        return tf.argmax(self.Q(tf.reshape(state, (1, -1)))[0])
    
    def policy(self, state):
        if np.random.uniform() > self.epsilon: return self.greedy_policy(state)
        return np.random.randint(self.env.num_j_temp)
    
    @tf.function
    def opt_q_val_many_states(self, states):
        """ Calculates action based on Q matrix.
        Parameters: 
            model: the neural network that takes state as input and gives Q values for different actions as outputs
            states: a n x 3 numpy array"""
        
        q_action_arrays = self.Q_t(states)
        opt_q_values = tf.reduce_max(q_action_arrays, axis=1)
        terminal_states_indexes = tf.where(states[:, 0] == self.env.n_tf)
        # # opt_q_values[terminal_states_indexes] = 0
        # terminal_states_indexes = terminal_states_indexes[:, 0]
        # print(terminal_states_indexes)
        update = tf.zeros(shape=(tf.shape(terminal_states_indexes)[0],))
        opt_q_values = tf.tensor_scatter_nd_update(opt_q_values, 
                                                   terminal_states_indexes, 
                                                   update,
                                                   )
        return opt_q_values
    
    def get_action(self, state):
        action_index = self.greedy_policy(np.hstack([state.reshape(-1)]))
        return self.env.tj_list[action_index]
    
    @tf.function
    def update_q_weights(self, inputs, action_args, labels):
        
        # training model
        masks = tf.one_hot(action_args, self.env.num_j_temp)
        masks = tf.reshape(masks, (masks.shape[0], masks.shape[2]))
        with tf.GradientTape() as tape:
            Q_values = self.Q(inputs)
            Q_action = tf.reduce_sum(tf.multiply(Q_values, masks), axis=1)
            loss = tf.keras.losses.mse(labels, Q_action)
        gradients = tape.gradient(loss, self.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.Q.trainable_variables))
    
    def train(self, num_episodes):
        iter_num = 0
        episode_versus_reward = np.zeros((num_episodes, 2))
        #episode_versus_reward = np.zeros((num_episodes, 2))
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        line, = ax.plot([], [])  # Empty plot to be updated dynamically


        for episode_index in range(num_episodes):
            # initialize markov chain with initial state
            state = self.env.reset()
            cumulative_reward = 0
            while not self.env.done:
                # epsilon greedy action selection
                action_index = self.policy(state)
                action = self.env.tj_list[action_index]
                #print("action",action)
                # executing action, observing reward and next state to store experience in tuple
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += (self.discount_factor ** iter_num) * reward
                # store experience in replay memory
                self.memory.push(np.hstack([state.reshape(-1), action_index, next_state.reshape(-1), reward]))
                
                # get replay memory
                if self.memory.size < self.batch_size: 
                    state = next_state
                    continue
                # get replay memory from MPC
                rand_batch_mpc = self.memory_mpc.get_batch(size=int(self.batch_size/2))
                rand_batch_replay = self.memory.get_batch(size=int(self.batch_size/2))
                rand_batch = np.append(rand_batch_mpc,rand_batch_replay, axis=0)
                # print("------------------------------------------------------------------------------------------")
                # print(rand_batch)
                inputs = rand_batch[:, self.state_index]
                next_inputs = rand_batch[:, self.next_state_index]
                action_args  = tf.cast(rand_batch[:, self.action_index], dtype=tf.dtypes.int32)
                labels = tf.cast(rand_batch[:, self.reward_index], dtype=tf.dtypes.float32) + self.discount_factor * self.opt_q_val_many_states(next_inputs)
                # get inputs and labels for neural network
                self.update_q_weights(inputs, action_args, labels)
                # update state
                if np.rint(self.env.curr_state[0, 0]).astype(int) % self.reset_steps == 0:
                    self.Q_t.set_weights(self.Q.get_weights())
                iter_num += 1
                state = next_state
             
            
            if episode_index % 10000 == 0:
                self.Q.save(f"Q_{episode_index}.h5")
            if episode_index % 100 == 0:
                print(f"[episodes]: {episode_index}")

            if episode_index % 1 == 0:
                # Update the plot dynamically
                episode_versus_reward[episode_index] = np.array([episode_index, cumulative_reward])
                line.set_data(episode_versus_reward[:episode_index+1, 0], episode_versus_reward[:episode_index+1, 1])
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)  # Adjust the pause time as needed
            
        plt.ioff()
        plt.show()
        return episode_versus_reward
