import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_probability as tfp


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
        self.A = self.actor_model(nn_arch)
        self.Q = self.critic_model(nn_arch)
        self.Q_t = self.critic_model(nn_arch)
        self.Q_t.set_weights(self.Q.get_weights())
        self.A_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.Q_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate, decay=decay_rate)
        self.reset_steps = reset_steps
        self.state_index = np.arange(0, self.env.state_dim, 1, dtype=np.int16)
        self.action_index = np.arange(self.state_index[-1]+1, self.state_index[-1]+1+self.env.action_dim, 1, dtype=np.int16)
        self.next_state_index = np.arange(self.action_index[-1]+1, self.action_index+1+self.env.state_dim, 1, dtype=np.int16)
        self.reward_index = self.next_state_index[-1] + 1
        self.memory_mpc = ReplayMemory_MPC(200)
        self.returns = []
        
    def critic_model(self, nn_arch):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nn_arch[0], input_shape=(4,), activation="relu"))
        for num_neurons in nn_arch[1:]:
            model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.build()
        return model
    
    def actor_model(self, nn_arch):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nn_arch[0], input_shape=(4,), activation="relu"))
        for num_neurons in nn_arch[1:]:
            model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.num_j_temp, activation="softmax"))
        model.build()
        return model
    
    @tf.function
    def policy(self,state):
        prob = self.A(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.A.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.Q.trainable_variables)
        self.A_opt.apply_gradients(zip(grads1, self.A.trainable_variables))
        self.Q_opt.apply_gradients(zip(grads2, self.Q.trainable_variables))
        return a_loss, c_loss
    
    def preprocess(states, actions, rewards, done,values, returns, gamma):
        g = 0
        lmbda = 0.95
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
            g = delta + gamma * lmbda * done[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        return states, actions, returns, adv 
    
    
    def value_function(self,states):
        values = []
        for i in states:
           values.append(self.Q(np.array(states[i])))
        return values
    
    def get_action(self, state):
        action_index = self.greedy_policy(np.hstack([state.reshape(-1)]))
        return self.env.tj_list[action_index]
    
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
            state = self.env.reset()
            all_aloss = []
            all_closs = []
            rewards = []
            states = []
            actions = []
            probs = []
            dones = []
            values = []
            while not self.env.done:
                action_index = self.policy(state)
                value = self.critic_model(state)
                action = self.env.tj_list[action_index]
                next_state, reward, done, _ = self.env.step(action)
                dones.append(1-done)
                rewards.append(reward)
                states.append(state)
                #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
                actions.append(action)
                prob = self.A(np.array([state]))
                probs.append(prob[0])
                values.append(value[0][0])
                cumulative_reward += (self.discount_factor ** iter_num) * reward
                state = next_state
                iter_num += 1
            value = self.critic_model(np.array([state])).numpy()
            values.append(value[0][0])
            np.reshape(probs, (len(probs),self.env.num_j_temp))
            probs = np.stack(probs, axis=0)
            returns = []
            adv
            states, actions, returns, adv  = self.preprocess(states, actions, rewards, dones, values,returns, 1)
            al,cl = self.learn(states, actions, adv, probs, returns)
            
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
