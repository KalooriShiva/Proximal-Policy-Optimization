import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

# class ReplayMemory_MPC:
#     def __init__(self,size):
#         df = pd.read_excel(r'C:\Users\Shiva\Desktop\RL_FRE\Time-varying-Setpoint-Tracking-for-Batch-Process-Control-using-Reinforcement-Learning-1\DQN_Implementation\Sample_data.xlsx')
#         # Convert each column of the DataFrame into a NumPy array
#         self.np_arrays = [df[col].to_numpy().reshape(-1, 1) for col in df.columns]
#         # Stack the NumPy arrays horizontally
#         self.memory_mpc = np.hstack(self.np_arrays)
#         self.size_mpc = size
#     def get_batch(self,size):
#         indexes = np.random.choice(range(0,self.size_mpc), size=size, replace=False)
#         return self.memory_mpc[indexes]

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
    
    def __init__(self, environment, learning_rate=1e-3, decay_rate=1e-4, discount_factor=1, epsilon=0.05, batch_size=2000, replay_memory_size= 5000, nn_arch=[400, 300, 200], reset_steps=9):
        
        # self.memory_mpc = ReplayMemory_MPC(200)
        self.env = environment
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = ReplayMemory(replay_memory_size, 2*self.env.state_dim + self.env.action_dim + 2)
        self.A = self.actor_model(nn_arch)
        self.Q = self.critic_model(nn_arch)
        self.Q_t = self.critic_model(nn_arch)
        self.Q_t.set_weights(self.Q.get_weights())
        self.A_opt = tf.keras.optimizers.Adam(learning_rate)
        self.Q_opt = tf.keras.optimizers.Adam(learning_rate)
        self.reset_steps = reset_steps
        self.state_index = np.arange(0, self.env.state_dim, 1, dtype=np.int16)
        self.action_index = np.arange(self.state_index[-1]+1, self.state_index[-1]+1+self.env.action_dim, 1, dtype=np.int16)
        self.next_state_index = np.arange(self.action_index[-1]+1, self.action_index+1+self.env.state_dim, 1, dtype=np.int16)
        self.reward_index = self.next_state_index[-1] + 1
        self.clip_pram = 0.2
        self.gae_lambda = 0.5
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
    
    def policy_probs(self,state):
        prob = self.A(tf.reshape(state, (1, -1)))
        return prob
    
    def policy(self,state):
        prob = self.policy_probs(state)
        dist = tfp.distributions.Categorical(probs=prob,dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy())
    
    def actor_loss(self, probs, actions, adv, old_probs):
        
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
            ratio = tf.math.divide(pb[a],op)
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
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)))
        #print(loss)
        return loss
    
    def update_actor(self, states, actions,  adv , old_probs):
        
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),self.env.num_j_temp))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.policy_probs(states)
            a_loss = self.actor_loss(p, actions, adv, old_probs)
            
        grads1 = tape1.gradient(a_loss, self.A.trainable_variables)
        self.A_opt.apply_gradients(zip(grads1, self.A.trainable_variables))
        return a_loss
    


    # Step 2: Compute Advantages using GAE
    def compute_gae_advantages(self,rewards, critic_values, next_critic_values):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
    
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.discount_factor * next_critic_values[t] - critic_values[t]
            gae = td_error + self.discount_factor * self.gae_lambda * gae
            advantages[t] = gae
        
        return advantages

    # Step 3: Normalize Advantages (optional)
    def normalize_advantages(advantages):
        mean_advantage = np.mean(advantages)
        std_advantage = np.std(advantages)
        normalized_advantages = (advantages - mean_advantage) / (std_advantage + 1e-8)  # Add small value to avoid division by zero
        return normalized_advantages
    
    def update_critic(self,states, next_states,rewards):
        # Compute critic values for current states
        critic_values = []
        next_critic_values = []
        for st in states:
            critic_values.append(self.Q(tf.reshape(st, (1, -1))))
        for nst in next_states:
            next_critic_values.append(self.Q(tf.reshape(nst, (1, -1))))

        # Convert lists to NumPy arrays
        critic_values = np.array(critic_values).reshape(-1)
        next_critic_values = np.array(next_critic_values).reshape(-1)

        
        with tf.GradientTape() as tape:
            
            target_values = rewards + self.discount_factor * next_critic_values
                # Compute TD errors
            td_errors = target_values - critic_values
                # Compute critic loss (e.g., mean squared error)
            critic_loss = tf.reduce_mean(tf.square(td_errors))
    
        # Compute gradients
        grads = tape.gradient(critic_loss, self.Q.trainable_variables)
        print(critic_loss)
        # Apply gradients to update critic parameters
        self.Q_opt.apply_gradients(zip(grads, self.Q.trainable_variables))
    
        return critic_values,next_critic_values
    def scaling_state(self,state):
        scaled_state = state.copy()
        scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
        scaled_state[2,0] = (scaled_state[2,0])*80
        scaled_state[3,0] = (scaled_state[3,0] - (-10))*(0 - 40)/(10 - (-10)) + 0
        #scaled_values = ((original_values - original_min) * (new_max - new_min)) / (original_max - original_min) + new_min

        return scaled_state
    
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
            scaled_state = self.scaling_state(state)
            cumulative_reward = 0
            while not self.env.done:
                # epsilon greedy action selection
                action_index = self.policy(scaled_state)
                action = self.env.tj_list[action_index]
                prob = self.policy_probs(scaled_state)
                prob = np.stack(prob, axis=0)
                print(prob)
                #print("action",action)
                # executing action, observing reward and next state to store experience in tuple
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += (self.discount_factor ** iter_num) * reward
                # store experience in replay memory
                self.memory.push(np.hstack([scaled_state.reshape(-1), action_index, next_state.reshape(-1), reward, prob[0][action_index]]))
                
                # get replay memory
                if self.memory.size < self.batch_size: 
                    print(self.memory.size, self.batch_size)
                    state = next_state
                    continue

                # rand_batch_mpc = self.memory_mpc.get_batch(size=int(4000))
                rand_batch = self.memory.get_batch(size=self.batch_size)
                # rand_batch = np.append(rand_batch_mpc,rand_batch_replay, axis=0)
                inputs = rand_batch[:, self.state_index]
                print("See----------------------------------------------------------------------------------------------------------------------------------------------------")
                print(inputs)
                next_inputs = rand_batch[:, self.next_state_index]
                action_args  = tf.cast(rand_batch[:, self.action_index], dtype=tf.dtypes.int32)
                old_probs = rand_batch[:,-1]
                rewards = rand_batch[:,-2]
                # Caluclation of TdLamda and update critic
                critic_values,next_critic_values = self.update_critic(inputs,next_inputs,rewards)

                # advantage calculations
                advantages = self.compute_gae_advantages(rewards,critic_values,next_critic_values)

                # Calculation of ratio of probs and update of actor

                al = self.actor_loss(inputs,action_args,advantages,old_probs)

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

