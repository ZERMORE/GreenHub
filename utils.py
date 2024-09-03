import configparser
import os
import sys
import argparse
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import os
import sys
import optparse
import traci



def evaluate_policy(env, model, turns = 3):
    scores = 0
    for j in range(turns):
        s, info, total_steps = env._reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_next, r, dw, tr,total_steps = env._run(a) # dw: terminated; tr: truncated
            done = dw + tr
            scores += r
            s = s_next
    return int(scores/turns)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, initial_p, final_p):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_sumo(gui, sumocfg_file_name, step_length,time_teleport):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # setting the cmd command to run sumo at simulation time  读取 intersection 文件夹中的 sumo文件
    sumo_cmd = [sumoBinary, "-c", os.path.join('maps', sumocfg_file_name), "--no-step-log", "true", "--step-length",
                str(step_length), "--time-to-teleport", str(time_teleport)]

    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]  # 获得编号的 列表
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options
def start_sumo():
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    sumocfgfile = "maps\\WUHAN\\wuhandata\\7.16.sumocfg"  # sumocfg文件的位置
    traci.start([sumoBinary, '-c', sumocfgfile]) # 打开sumocfg文件
    for step in range(0, 3600):  # 仿真时间
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
    traci.close()
