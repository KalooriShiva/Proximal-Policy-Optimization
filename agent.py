import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls


# class ReplayMemory_MPC:
#     # def __init__(self,size):
#     #     df = pd.read_excel(r'C:\Users\Shiva\Desktop\RL_FRE\Time-varying-Setpoint-Tracking-for-Batch-Process-Control-using-Reinforcement-Learning-1\DQN_Implementation\Sample_data.xlsx')
#     #     # Convert each column of the DataFrame into a NumPy array
#     #     self.np_arrays = [df[col].to_numpy().reshape(-1, 1) for col in df.columns]
#     #     # Stack the NumPy arrays horizontally
#     #     self.memory_mpc = np.hstack(self.np_arrays)
#     #     self.size_mpc = size
#     # def get_batch(self,size):
#     #     indexes = np.random.choice(range(0,self.size_mpc), size=size, replace=False)
#     #     return self.memory_mpc[indexes]

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

class PPOAgent:
    
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
        self.A_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.Q_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate, decay=decay_rate)
        self.reset_steps = reset_steps
        self.state_index = np.arange(0, self.env.state_dim, 1, dtype=np.int16)
        self.action_index = np.arange(self.state_index[-1]+1, self.state_index[-1]+1+self.env.action_dim, 1, dtype=np.int16)
        self.next_state_index = np.arange(self.action_index[-1]+1, self.action_index+1+self.env.state_dim, 1, dtype=np.int16)
        self.reward_index = self.next_state_index[-1] + 1
        self.clip_pram = 0.2
        # self.memory_mpc = ReplayMemory_MPC(200)
        self.returns = []
        
    def critic_model(self, nn_arch):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nn_arch[0], input_shape=(3,), activation="relu"))
        for num_neurons in nn_arch[1:]:
            model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.build()
        return model
    
    def actor_model(self, nn_arch):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nn_arch[0], input_shape=(3,), activation="relu"))
        for num_neurons in nn_arch[1:]:
            model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.num_j_temp, activation="softmax"))
        model.build()
        return model
    
    def policy_probs(self,state):
        prob = self.A(tf.reshape(state, (1, -1)))
        return prob
    
    def policy(self,state):
        prob = self.policy_probs(state)
        dist = tfp.distributions.Categorical(probs=prob,dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy())
    
    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op, a  in zip(probability, adv, old_probs, actions):
            t =  tf.constant(t)
            #op =  tf.constant(op)
            #print(f"t{t}")
            #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb[a],op[a])
            #print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio,t)
            #print(f"s1{s1}")
            s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
            #print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss
    
    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),self.env.num_j_temp))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.A(states)
            v =  self.Q(states)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.A.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.Q.trainable_variables)
        self.A_opt.apply_gradients(zip(grads1, self.A.trainable_variables))
        self.Q_opt.apply_gradients(zip(grads2, self.Q.trainable_variables))
        return a_loss, c_loss
    
    def preprocess(self, states, actions, rewards, done,values, returns, gamma):
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
    
    
    def train(self, num_episodes):
        iter_num = 0
        episode_versus_reward = np.zeros((num_episodes, 2))
        # #episode_versus_reward = np.zeros((num_episodes, 2))
        # plt.ion()  # Turn on interactive mode
        # fig, ax = plt.subplots()
        # line, = ax.plot([], [])  # Empty plot to be updated dynamically


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
                scaled_state = state.copy()
                scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
                scaled_state[2,0] = (scaled_state[2,0])*80
                # print(scaled_state.reshape(-1))
                action_index = self.policy(scaled_state)
                value = self.Q(tf.reshape(scaled_state, (1, -1)))
                action = self.env.tj_list[action_index]
                next_state, reward, done, _ = self.env.step(action)
                dones.append(1-done)
                rewards.append(reward)
                states.append(scaled_state)
                #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
                actions.append(action_index)
                prob = self.policy_probs(scaled_state)
                probs.append(np.array(prob[0]))
                values.append(np.array(value[0][0]))
                cumulative_reward += (self.discount_factor ** iter_num) * reward
                state = next_state
                iter_num += 1
            value = self.Q(tf.reshape(scaled_state, (1, -1))).numpy()
            values.append(value[0][0])
            np.reshape(probs, (len(probs),self.env.num_j_temp))
            probs = np.stack(probs, axis=0)
            returns = []
            states, actions, returns, adv  = self.preprocess(states, actions, rewards, dones, values,returns, 1)
            al,cl = self.learn(states, actions, adv, probs, returns)
            
            if episode_index % 10000 == 0:
                self.Q.save(f"Q_{episode_index}.h5")
            if episode_index % 100 == 0:
                print(f"[episodes]: {episode_index}")

            if episode_index % 1 == 0:
                # Update the plot dynamically
                episode_versus_reward[episode_index] = np.array([episode_index, cumulative_reward])
        #         line.set_data(episode_versus_reward[:episode_index+1, 0], episode_versus_reward[:episode_index+1, 1])
        #         ax.relim()
        #         ax.autoscale_view()
        #         plt.draw()
        #         plt.pause(0.001)  # Adjust the pause time as needed
            
        # plt.ioff()
        # plt.show()
        return episode_versus_reward
