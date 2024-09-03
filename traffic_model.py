
import numpy as np
import paddle
import pandas as pd
import traci

# import gymnasium as gym
from DQN_paddle import DQN_Agent
from LPRB import LightPriorReplayBuffer, device
# from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy, str2bool, LinearSchedule,set_sumo, set_train_path
from training_simulation1 import Simulation
import matplotlib.pyplot as plt

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
# parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=250, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(3600), help='Max training steps')  #
parser.add_argument('--buffer_size', type=int, default=int(40000), help='size of the replay buffer')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--warmup', type=int, default=int(600), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=3600, help='training frequency')  # 5 steps

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr_init', type=float, default=1.5e-4, help='Initial Learning rate')
parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
parser.add_argument('--lr_decay_steps', type=float, default=int(6000), help='Learning rate decay steps')  #3000  *5
parser.add_argument('--batch_size', type=int, default=50, help='lenth of sliced trajectory')  #
parser.add_argument('--exp_noise_init', type=float, default=0.6, help='init explore noise')
parser.add_argument('--exp_noise_end', type=float, default=0.03, help='final explore noise')
parser.add_argument('--noise_decay_steps', type=int, default=int(5000), help='decay steps of explore noise')  #2500   *5
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')

parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
parser.add_argument('--beta_gain_steps', type=int, default=int(6000), help='steps of beta from beta_init to 1.0')   #3000   *5
parser.add_argument('--replacement', type=str2bool, default=False, help='sample method')
opt = parser.parse_args()

opt.state_dim = 17
opt.action_dim = 27
episodes = 20 # training eps
step_length = 1
time_teleport = 60
sumocfg_file_name = "WUHAN\\wuhandata\\7.16.sumocfg"
gui = False
sumo_cmd = set_sumo(gui, sumocfg_file_name, step_length,time_teleport)






#创建模型
model = DQN_Agent(opt)

#加载模型参数
model.load('DDQN', 'vsl_ddqn', 1)

#加载环境
Env = Simulation(
    model,
    sumo_cmd,

    opt.gamma,
    opt.Max_train_steps,
    opt.state_dim,
    opt.action_dim,
)

#无模型控制
timestamp_start = datetime.now()
print('\n测试无模型控制', 'start time:', timestamp_start)

#初始化数据容器
total_reward = []
total_Average_Queuelength = []
total_Fuelconsumption = []
total_CO2emission = []
total_Average_Acceleration = []
total_Waitingtime = []
eps_reward_list_none = []

for episode in range(episodes):
    timestamp_start = datetime.now()
    print('\n----- Episode', str(episode + 1), 'of', str(episodes), 'start time: ', timestamp_start)

    count = 0
    episode_reward = 0
    Queuelength_sum = 0
    Fuelconsumption_sum = 0
    CO2emission_sum = 0
    Acceleration_sum = 0
    Waiting_time_sum = 0
    done = False
    done, total_steps,v1_old,v2_old,v3_old = Env._reset_none()

    while not done:

        r, dw, done = Env._run_none()

        #获取执行一个动作（60s）后的指标，然后累加到x_sum上（x_sum为一个episode内部所有动作周期的累加指标）
        Queuelength, Fuelconsumption, CO2emission, Acceleration, Waiting_time = Env.get_from_traci()
        Queuelength_sum += Queuelength
        Fuelconsumption_sum += Fuelconsumption
        CO2emission_sum += CO2emission
        Acceleration_sum += Acceleration
        Waiting_time_sum += Waiting_time
        count += 1
        # 获取一步动作（60秒）内的平均排队长度，总燃油消耗，总CO2排放，平均车辆加速度，总累计等待时间
        episode_reward += r

    #计算当前episode的平均排队长度和平均车辆加速度
    Average_Queuelength = Queuelength_sum / count
    Average_Acceleration = Acceleration_sum / count

    #将当前episode内的指标存入数组之中
    total_reward.append(episode_reward)
    total_Average_Queuelength.append(Average_Queuelength)
    total_Fuelconsumption.append(Fuelconsumption_sum)
    total_CO2emission.append(CO2emission_sum)
    total_Average_Acceleration.append(Average_Acceleration)
    total_Waitingtime.append(Waiting_time_sum)

    Env._close()

    end_time = datetime.now()
    print("simulation time: ", end_time - timestamp_start, '----eps_reward: ', episode_reward,)
    eps_reward_list_none.append(episode_reward)
#计算所有episodes平均下来的指标
average_reward = sum(total_reward)/episodes
average_queuelength = sum(total_Average_Queuelength)/episodes
average_fuelconsumption = sum(total_Fuelconsumption) / episodes
average_co2emission = sum(total_CO2emission) / episodes
average_acceleration = sum(total_Average_Acceleration) / episodes
average_waitingtime = sum(total_Waitingtime) / episodes

#输出
df_all_none = pd.DataFrame(
    columns=['Total_Waitingtime', 'Average_Queuelength', 'Total_CO2emission', 'Total_Fuelall', 'Average_Acceleration', 'Oflow'])
new_data = {
    'Total_Waitingtime': [average_waitingtime],
    'Average_Queuelength': [average_queuelength],
    'Total_CO2emission': [average_co2emission],
    'Total_Fuelall': [average_fuelconsumption],
    'Average_Acceleration': [average_acceleration],
    'Oflow': [average_reward]
}

new_df = pd.DataFrame(new_data)
df_all_none = pd.concat([df_all_none,new_df],ignore_index=True)
# df_all_none = df_all_none.append(
#     {'Total_Waitingtime': average_waitingtime,
#      'Average_Queuelength': average_queuelength,
#      'Total_CO2emission': average_co2emission,
#      'Total_Fuelall': average_fuelconsumption,
#      'Average_Acceleration': average_acceleration,
#      'Oflow': average_reward}, ignore_index=True)
df_all_none.to_csv("output\\Model_Output_none0.csv", index=True)
#DQN控制
timestamp_start = datetime.now()
print('\n测试DQN模型', 'start time:', timestamp_start)

#初始化数据容器
total_reward = []
total_Average_Queuelength = []
total_Fuelconsumption = []
total_CO2emission = []
total_Average_Acceleration = []
total_Waitingtime = []
eps_reward_list = []

for episode in range(episodes):
    timestamp_start = datetime.now()
    print('\n----- Episode', str(episode + 1), 'of', str(episodes), 'start time: ', timestamp_start)

    count = 0
    episode_reward = 0
    Queuelength_sum = 0
    Fuelconsumption_sum = 0
    CO2emission_sum = 0
    Acceleration_sum = 0
    Waiting_time_sum = 0
    done = False
    state, done, total_steps,v1_old,v2_old,v3_old = Env._reset()

    while not done:
        a = model.select_action(state, deterministic=True)
        s_next, r, dw, done, total_steps, v1_new, v2_new, v3_new = Env._run(a, v1_old, v2_old, v3_old)
        print(v1_new, v2_new, v3_new)

        #获取执行一个动作（60s）后的指标，然后累加到x_sum上（x_sum为一个episode内部所有动作周期的累加指标）
        Queuelength, Fuelconsumption, CO2emission, Acceleration, Waiting_time = Env.get_from_traci()
        Queuelength_sum += Queuelength
        Fuelconsumption_sum += Fuelconsumption
        CO2emission_sum += CO2emission
        Acceleration_sum += Acceleration
        Waiting_time_sum += Waiting_time
        count += 1
        # 获取一步动作（60秒）内的平均排队长度，总燃油消耗，总CO2排放，平均车辆加速度，总累计等待时间
        episode_reward += r
        state = s_next
        v1_old = v1_new
        v2_old = v2_new
        v3_old = v3_new

    #计算当前episode的平均排队长度和平均车辆加速度
    Average_Queuelength = Queuelength_sum / count
    Average_Acceleration = Acceleration_sum / count

    #将当前episode内的指标存入数组之中
    total_reward.append(episode_reward)
    total_Average_Queuelength.append(Average_Queuelength)
    total_Fuelconsumption.append(Fuelconsumption_sum)
    total_CO2emission.append(CO2emission_sum)
    total_Average_Acceleration.append(Average_Acceleration)
    total_Waitingtime.append(Waiting_time_sum)

    Env._close()

    end_time = datetime.now()
    print("simulation time: ", end_time - timestamp_start, '----eps_reward: ', episode_reward,)
    #print(v1_new, v2_new, v3_new)
    eps_reward_list.append(episode_reward)
#计算所有episodes平均下来的指标
average_reward = sum(total_reward)/episodes
average_queuelength = sum(total_Average_Queuelength)/episodes
average_fuelconsumption = sum(total_Fuelconsumption) / episodes
average_co2emission = sum(total_CO2emission) / episodes
average_acceleration = sum(total_Average_Acceleration) / episodes
average_waitingtime = sum(total_Waitingtime) / episodes

#输出
df_all = pd.DataFrame(
    columns=['Total_Waitingtime', 'Average_Queuelength', 'Total_CO2emission', 'Total_Fuelall', 'Average_Acceleration', 'Oflow'])
new_data = {
    'Total_Waitingtime': [average_waitingtime],
    'Average_Queuelength': [average_queuelength],
    'Total_CO2emission': [average_co2emission],
    'Total_Fuelall': [average_fuelconsumption],
    'Average_Acceleration': [average_acceleration],
    'Oflow': [average_reward]
}

new_df = pd.DataFrame(new_data)
df_all = pd.concat([df_all,new_df],ignore_index=True)
# df_all_none = df_all_none.append(
#     {'Total_Waitingtime': average_waitingtime,
#      'Average_Queuelength': average_queuelength,
#      'Total_CO2emission': average_co2emission,
#      'Total_Fuelall': average_fuelconsumption,
#      'Average_Acceleration': average_acceleration,
#      'Oflow': average_reward}, ignore_index=True)
df_all.to_csv("output\\Model_Output_compare.csv", index=True)

#绘图
x_range = list(range(len(eps_reward_list)))
#num=eps_reward_list
plt.plot(x_range, eps_reward_list) # 每个回合return
plt.plot(x_range, eps_reward_list_none)
plt.xlabel('episode')
plt.ylabel('return')
plt.legend(['DDQN control', 'base'], loc='upper left')
plt.show()


#mctf控制方法
