import sys

import numpy as np
import paddle
import pandas as pd
import traci
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from DQN_paddle import DQN_Agent
from LPRB import LightPriorReplayBuffer, device
from utils import evaluate_policy, str2bool, LinearSchedule, set_sumo, set_train_path
from training_simulation1 import Simulation

def initialize_parser():
    """Initialize and return the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=250, help='Which model to load')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--Max_train_steps', type=int, default=3600, help='Max training steps')
    parser.add_argument('--buffer_size', type=int, default=40000, help='Size of the replay buffer')
    parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
    parser.add_argument('--warmup', type=int, default=600, help='Steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=3600, help='Training frequency')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
    parser.add_argument('--lr_init', type=float, default=1.5e-4, help='Initial Learning rate')
    parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
    parser.add_argument('--lr_decay_steps', type=int, default=6000, help='Learning rate decay steps')
    parser.add_argument('--batch_size', type=int, default=50, help='Length of sliced trajectory')
    parser.add_argument('--exp_noise_init', type=float, default=0.6, help='Initial explore noise')
    parser.add_argument('--exp_noise_end', type=float, default=0.03, help='Final explore noise')
    parser.add_argument('--noise_decay_steps', type=int, default=5000, help='Decay steps of explore noise')
    parser.add_argument('--DDQN', type=str2bool, default=True, help='True: DDQN; False: DQN')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha for PER')
    parser.add_argument('--beta_init', type=float, default=0.4, help='Beta for PER')
    parser.add_argument('--beta_gain_steps', type=int, default=6000, help='Steps of beta from beta_init to 1.0')
    parser.add_argument('--replacement', type=str2bool, default=False, help='Sample method')
    args, unknown = parser.parse_known_args()  # 使用 parse_known_args 忽略未知参数
    return args

def setup_environment(opt, sumocfg_file_name):
    """Setup and return the simulation environment and model."""
    gui = False
    sumo_cmd = set_sumo(gui, sumocfg_file_name, step_length=1, time_teleport=60)

    model = DQN_Agent(opt)
    model.load('DDQN', 'vsl_ddqn', 1)

    Env = Simulation(model, sumo_cmd, opt.gamma, opt.Max_train_steps, opt.state_dim, opt.action_dim)
    return model, Env
def run_simulation(Env, episodes, use_model=False):
    """Run simulation and return the results."""
    total_reward = []
    total_Average_Queuelength = []
    total_Fuelconsumption = []
    total_CO2emission = []
    total_Average_Acceleration = []
    total_Waitingtime = []

    for episode in range(episodes):
        timestamp_start = datetime.now()
        print(f'\n----- Episode {episode + 1} of {episodes} start time: {timestamp_start}')

        count, episode_reward = 0, 0
        Queuelength_sum = Fuelconsumption_sum = CO2emission_sum = Acceleration_sum = Waiting_time_sum = 0
        done = False

        if use_model:
            state, done, total_steps, v1_old, v2_old, v3_old = Env._reset()
        else:
            done, total_steps, v1_old, v2_old, v3_old = Env._reset_none()

        while not done:
            if use_model:
                a = model.select_action(state, deterministic=True)
                s_next, r, dw, done, total_steps, v1_new, v2_new, v3_new = Env._run(a, v1_old, v2_old, v3_old)
                state = s_next
                v1_old, v2_old, v3_old = v1_new, v2_new, v3_new
            else:
                r, dw, done = Env._run_none()

            Queuelength, Fuelconsumption, CO2emission, Acceleration, Waiting_time = Env.get_from_traci()
            Queuelength_sum += Queuelength
            Fuelconsumption_sum += Fuelconsumption
            CO2emission_sum += CO2emission
            Acceleration_sum += Acceleration
            Waiting_time_sum += Waiting_time
            count += 1
            episode_reward += r

        Average_Queuelength = Queuelength_sum / count
        Average_Acceleration = Acceleration_sum / count

        total_reward.append(episode_reward)
        total_Average_Queuelength.append(Average_Queuelength)
        total_Fuelconsumption.append(Fuelconsumption_sum)
        total_CO2emission.append(CO2emission_sum)
        total_Average_Acceleration.append(Average_Acceleration)
        total_Waitingtime.append(Waiting_time_sum)

        Env._close()

        end_time = datetime.now()
        print(f"simulation time: {end_time - timestamp_start} ---- eps_reward: {episode_reward}")

    return {
        "total_reward": total_reward,
        "total_Average_Queuelength": total_Average_Queuelength,
        "total_Fuelconsumption": total_Fuelconsumption,
        "total_CO2emission": total_CO2emission,
        "total_Average_Acceleration": total_Average_Acceleration,
        "total_Waitingtime": total_Waitingtime
    }

def calculate_average_results(results, episodes):
    """Calculate and return the average results of the simulation."""
    average_results = {}
    for key in results:
        average_results[key] = sum(results[key]) / episodes
    return average_results

def save_results_to_csv(results, file_path):
    """Save results to CSV file."""
    df = pd.DataFrame([results])
    df.to_csv(file_path, index=False)

def plot_results(eps_reward_list, eps_reward_list_none):
    """Plot the results of the simulations."""
    x_range = list(range(len(eps_reward_list)))
    plt.plot(x_range, eps_reward_list, label='DDQN control')
    plt.plot(x_range, eps_reward_list_none, label='base')
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    opt = initialize_parser()
    opt.state_dim = 17
    opt.action_dim = 27
    episodes = 20

    sumocfg_file_name = sys.argv[1]  # 从命令行获取sumocfg文件路径
    output_directory = sys.argv[2]  # 从命令行获取输出目录

    model, Env = setup_environment(opt, sumocfg_file_name)

    # Run simulation without model control
    print('\n测试无模型控制')
    results_none = run_simulation(Env, episodes, use_model=False)
    average_results_none = calculate_average_results(results_none, episodes)
    save_results_to_csv(average_results_none, os.path.join(output_directory, "Model_Output_none0.csv"))

    # Run simulation with DQN control
    print('\n测试DQN模型')
    results = run_simulation(Env, episodes, use_model=True)
    average_results = calculate_average_results(results, episodes)
    save_results_to_csv(average_results, os.path.join(output_directory, "Model_Output_compare.csv"))

    # Plot results
    plot_results(results['total_reward'], results_none['total_reward'])