import numpy as np
import pandas as pd
import seaborn as sns
from reactor_environment import Environment
from agent import DQNAgent
import time 
import matplotlib.pyplot as plt
import time 
from datetime import date
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
if __name__ == "__main__":
    env = Environment(timesteps=40, num_j_temp=40)
    agent = DQNAgent(learning_rate=1e-3, decay_rate=1e-4, batch_size=20, replay_memory_size=50, environment=env, reset_steps=9, nn_arch=[400, 300, 200])
    start_time = time.perf_counter()
    episode_versus_reward = agent.train(5000)
    # agent.Q = tf.keras.models.load_model(r"C:\Users\Dr Nabil\Desktop\Shiva_DQN_Random_Trajectory\Q_50000.h5")
    agent.Q.save("Q_network.h5")
    cpu_time = time.perf_counter() - start_time
    state_arr = np.zeros_like(env.time_list)
    conc_arr = np.zeros_like(env.time_list)
    action_arr = np.zeros_like(env.time_list)
    reward_arr = np.zeros_like(env.time_list)
    episode_vs_reward_df = pd.DataFrame({"Episodes": episode_versus_reward[:, 0], "Reward": episode_versus_reward[:, 1]})
    env.curr_state = np.vstack([0, 298, 0.6, 298-env.Tref[0]])
    state = env.curr_state
    for i in range(env.n_tf):
        state_arr[i] = state[env.T, 0]
        conc_arr[i] = state[env.Ca, 0]
        action_arr[i] = agent.get_action(state)
        #reward_computation
        next_state, reward, done, info = env.step(action_arr[i])
        reward_arr[i] = reward
        state = next_state
    df = pd.DataFrame({"Temperature": state_arr, "Time": env.time_list, "Reference": env.Tref, "Jacket Temperature": action_arr, "Concentration [A]": conc_arr, "Reward": reward_arr})
    sns.set_theme()
    fig1 = plt.figure(1)
    sns.lineplot(
        data=df,
        x="Time",
        y="Temperature",
        label="Reactor Temperature",
    )
    sns.lineplot(
        data=df,
        x="Time",
        y="Reference",
        legend="full",
        label="Reference Temperature",
    )
    fig2 = plt.figure(2)
    sns.lineplot(
        data=df,
        x="Time",
        y="Jacket Temperature",
        legend="full",
        label="Action",
    )
    
    # concentration plot
    fig3 = plt.figure(3)
    sns.lineplot(data=df, x="Time", y="Concentration [A]")
    window_size = 100
    rolling_avg = pd.Series(episode_versus_reward[:,1]).rolling(window=window_size, min_periods=1).mean()
    data = pd.DataFrame({'Episode': range(1, episode_versus_reward.shape[0] + 1),
                          'Reward':episode_versus_reward[:,1] ,
                          'Rolling Average': rolling_avg})
    fig4 = plt.figure(4)
    sns.set_style("darkgrid")
    sns.lineplot(data=data, x='Episode', y = 'Reward', label='Episode Reward', color='blue')
    sns.lineplot(data=data, x='Episode', y = 'Rolling Average', label='Rolling Average', color='red')
    #sns.lineplot(episode_vs_reward_df, x="Episodes", y="Reward")
    plt.title("Episode_Vs_Reward")
    
    pp = PdfPages(f"Plots{date.today()}.pdf")
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.show()
    # concentration plot
    # sns.lineplot(data=df, x="Time", y="Concentration [A]")
    # control signal plot
    # sns.lineplot(x=env.time_list, y=action_arr, drawstyle='steps-pre', label="Jacket Temperature")
    # episode versus reward plot
    # sns.lineplot(episode_vs_reward_df, x="Episodes", y="Reward")
    MAE = np.sum(np.abs(state_arr-env.Tref))/len(state_arr)
    RMSE = (np.sum((state_arr-env.Tref)**2)/len(state_arr))**0.5
    days = int(cpu_time // 86400)
    hrs = int((cpu_time -  86400 * days)// 3600)
    mins = int((cpu_time - 3600 * hrs - 86400 * days) // 60)
    seconds = int((cpu_time - 60 * mins - 3600 * hrs - 86400 * days) // 1)
    print(f"{days = }\n{hrs = }\n{mins = }\n{seconds = }")  
    #$env:TF_USE_LEGACY_KERAS = "True"
