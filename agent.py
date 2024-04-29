import numpy as np
import pandas as pd
import seaborn as sns
from reactor_environment import Environment
from agent import PPOAgent
import time 
import matplotlib.pyplot as plt
import time 
from datetime import date
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
    
    
if __name__ == "__main__":
    env = Environment(timesteps=40, num_j_temp=40)
    agent = PPOAgent(learning_rate=1e-6, decay_rate=1e-7, environment=env, nn_arch=[400, 300, 200])
    #agent.A = tf.keras.models.load_model(r"C:\Users\Dr Nabil\Downloads\Policy gradient\Pg_final\P_network.h5")
    env.testing = True
    start_time = time.perf_counter()
    episode_versus_reward = agent.train(70000)
    cpu_time = time.perf_counter() - start_time
    agent.A.save("A_network.h5")
    state_arr = np.zeros_like(env.time_list)
    conc_arr = np.zeros_like(env.time_list)
    action_arr = np.zeros_like(env.time_list)
    reward_arr = np.zeros_like(env.time_list)
    episode_vs_reward_df = pd.DataFrame({"Episodes": episode_versus_reward[:, 0], "Reward": episode_versus_reward[:, 1]})
    env.curr_state = np.vstack([0, 298, 0.6])
    state = env.curr_state
    for i in range(env.n_tf):
        state_arr[i] = state[env.T, 0]
        conc_arr[i] = state[env.Ca, 0]
        scaled_state = state.copy()
        scaled_state[1,0] = (scaled_state[1,0] - env.min_j_temp)*(80/(env.max_j_temp-env.min_j_temp))
        scaled_state[2,0] = (scaled_state[2,0])*80
        action_arr[i] = env.tj_list[agent.policy(scaled_state)]
        #reward_computation
        next_state, reward, done, info = env.step(action_arr[i])
        reward_arr[i] = reward
        state = next_state
    df = pd.DataFrame({"Temperature": state_arr, "Time": env.time_list, "Reference": env.Tref, "Jacket Temperature": action_arr, "Concentration [A]": conc_arr, "Reward": reward_arr})
#    df.to_excel("PG.xlsx")
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
        drawstyle = "steps-pre"
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
    # control signal plot
    # sns.lineplot(x=env.time_list, y=action_arr, drawstyle='steps-pre', label="Jacket Temperature")
    # episode versus reward plot
    # sns.lineplot(episode_vs_reward_df, x="Episodes", y="Reward")
    # plt.show()
    MAE = np.sum(np.abs(state_arr-env.Tref))/len(state_arr)
    RMSE = (np.sum((state_arr-env.Tref)**2)/len(state_arr))**0.5
    days = int(cpu_time // 86400)
    hrs = int((cpu_time -  86400 * days)// 3600)
    mins = int((cpu_time - 3600 * hrs - 86400 * days) // 60)
    seconds = int((cpu_time - 60 * mins - 3600 * hrs - 86400 * days) // 1)
    print(f"{days = }\n{hrs = }\n{mins = }\n{seconds = }")  
