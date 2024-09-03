import numpy as np
import random
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

'''
The Simulation class handles the simulation. 
In particular, the function run allows the simulation of one episode. 
Also, other functions are used during run to interact with SUMO, for example: 
retrieving the state of the environment (get_state), set the next green light phase (_set_green_phase) or 
preprocess the data to train the neural network (_replay). 
Two files contain a slightly different Simulation class: training_simulation.py and testing_simulation.py. 
Which one is loaded depends if we are doing the training phase or the testing phase.
'''


# phase codes based on environment.net.xml




class Simulation:
    def __init__(self, Model,  sumo_cmd, gamma, max_steps,  num_states, num_actions):
        self._Model = Model
        # self._Memory = Memory
        # self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        # self._green_duration = green_duration
        # self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions  # 9  注意action = -1 0 1 ， 代表+-5 mph
        self._action_dim = 1  # or 3 or  5
        # self._training_epochs = training_epochs
        self._step_length = 1
        self._reward_store = []
        # self._cumulative_wait_store = []
        # self._avg_queue_length_store = []
        # # create the outputs in here first
        # self._cumulative_co2_store = []
        # self._cumulative_fuel_store = []
        self.control_horizon = 60  # action time step 60s
        '''
        Network Parameters
        '''
        self.control_section = 'R2'
        self.state_detector = ['rlight1_0loop', 'R1_1loop', 'R1_2loop', 'R1_3loop', 'R1_4loop', 'R1_5loop',
                               'R4_0loop', 'R4_1loop', 'R4_2loop', 'R4_3loop', 'R4_4loop', 'R4_5loop']
        #self.state_lane = ['rin3_0','R2_0', 'R2_1', 'R2_2', 'R2_3', 'R2_4',
        #                   'R4_0', 'R4_1', 'R4_2', 'R4_3', 'R4_4','R4_5'] # 12
        self.state_edge = ['R3','RL1', 'RL2', 'R2'] #4
        self.state_edge_length = [159.64,88.23,125.19,213.86]  # 4
        self.VSLlist = ['R2_0','R2_1', 'R2_2','R1_0','R1_1','R1_2']
        self.VSLlist1 = ['R2_0','R1_0']
        self.VSLlist2 = ['R2_1','R1_1']
        self.VSLlist3 = ['R2_2','R1_2']
        self.inID = [ 'R1_0loop', 'R1_1loop', 'R1_2loop','RL1_loop']
        self.outID = ['R4_0loop', 'R4_1loop', 'R4_2loop', 'R4_3loop']
        self.bottleneck_detector = ['R4_0loop', 'R4_1loop', 'R4_2loop', 'R4_3loop']
        self.action_sets = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
         [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
        [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]]


    def _reset(self,):
        # first, generate the route file for this simulation and set up sumo
        # self._TrafficGen.generate_routefile(seed=episode)    #  seed=episode 随机种子是变动的
        # self._TrafficGen.writenewtrips(episode)
        traci.start(self._sumo_cmd)  # a new round simulation
        print("start new Simulating...")
        self._step = 0

        self._reward = 0
        # self._warmuptime_steps = 10 # 600/60
        self._warmuptime(600)
        self.done = False
        # inits

        self._ep_reward = 0
        self._outflow = 0
        self._bottle_speed = 0
        # for step in range(self._warmuptime_steps):
            # v = 29.06 * np.ones(self._action_dim)  # run the first step
        v1,v2,v3 =  self.from_a_to_mlv(self._random_action(),12.5,12.5,12.5)   # ---> m/s
            # self.state = np.zeros(self._num_states, )
        state, reward,  = self.run_step(13,v1,v2,v3)  # 初始速度 执行一个动作周期 60s
        # self.state = state  # 更新状态
        # state = np.ones()
        # self._outflow = self._outflow + outflow_temp  # 更新outflow

        # current_state = self.get_step_state()  # get state_
        self.old_action = 4 # 初始化的动作

        # initials
        self._reward = reward
        self.sum_pos_reward = 0


        return state,  self.done, self._step,v1,v2,v3


    def _warmuptime(self,time):
        # steps = 0
        if self._step  < time:  # do not do more steps than the maximum allowed number of steps
            traci.simulationStep()
            self._step += 1




    def _run(self, act,v1_old,v2_old,v3_old):
        """
        Runs an episode of simulation, then starts a training session
        """
        # self._reset()  #start new simulation,   do on the training main file

        current_state = self.get_step_state(act)   # current_state
        #epsilon = 0.001
        #action = self._choose_action(current_state, act,epsilon)
        action = act     # 获取当前状态对应的动作  old_action   [7,7,7]
        #print('action: ',action)
        v1,v2,v3= self.from_a_to_mlv(action,v1_old,v2_old,v3_old)
        # get current state 环境更新 # current_state = self.get_step_state()
        state_updated, reward,    = self.run_step(act,v1,v2,v3)  # 初始速度 执行一个动作周期 60s
        # state_next = state_updated  #state = next_state
        # reward_revised = reward - self._reward
        # reward_revised = (reward - self._reward)*50
        reward_revised = reward
        #if reward_revised ==0:
        #    reward_revised = 1
        #elif reward_revised>0:
        #    reward_revised = 1+reward_revised*2

        self._reward = reward # update reward_

        # saving the data into the memory
        # print('action:  ',action, 'actionarray:',actionindex)
        # self._Memory.add_sample((current_state, action, reward_revised,  state_updated, actionindex))
        # transition = []
        # self._Memory.add_sample((current_state, action, reward_revised,  state_updated, ))
        # saving variables for later & accumulate reward
        self.state = state_updated  # old state
        # self.old_action = action  # output
        if reward_revised > 0:
            self.sum_pos_reward += reward_revised
        # saving only the meaningful reward to better see if the agent is behaving correctly
        # self._save_episode_stats()

        dw, done = self.done, self.done

        return state_updated, reward_revised, dw, done, self._step,v1,v2,v3

    #####################  run one action step: reward is outflow  ####################
    # 跑一个动作长度  60s， 并且 输出结果
    def run_step(self,act, v1,v2,v3):
        """
        Execute steps in sumo while gathering statistics
        """
        state_overall = 0
        reward = 0

        oflow = 0
        bspeed = 0
        bn_speed = 0

        self.set_vsl(v1,v2,v3)
        steps = self.control_horizon
        steptodo = steps
        if (self._step + self.control_horizon) > self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps = self._max_steps - self._step
            steptodo  = steps

        while steptodo > 0:
            traci.simulationStep()  # simulate 1 step in sumo   注意这里结果是0.1s 所以下面计算的结果 需要注意！
            # state_overall = state_overall + self.get_step_state()  # 建议更改state

            oflow = oflow + self.calc_outflow()
            #
            #bn_speed =bn_speed + traci.edge.getLastStepMeanSpeed('R4')  # bottleneck speed


            self._step += 1  # update the step counter
            if self._step == self._max_steps:
                self.done = True
                break
            steptodo -= 1

        state_overall =  self.get_step_state(act)  # 建议更改state
        #reward =  oflow / 60 * 0.2 + bn_speed / (30 * steps) * 0.8   # 这个是原始的奖励函数
        # reward =  oflow / 60 * 0.2 + bspeed / (30 * steps) * 0.8   #
        # reward = bn_speed / steps       # the reward is defined as the outflow
        reward = oflow
        # reward = bn_speed/steps

        # state, reward, oflow_temp,  =
        return state_overall, reward,

    # def _bottleneck_speed(self):
    #     return traci.edge.getLastStepMeanSpeed('R4')



    ### action  ###
    def _random_action(self,):
        # action = np.zeros(self._action_dim, ) #[0,0,0,0,0]

        action_array = random.randint(0, 26)


        # print('random action_array', action_array_)
        # print('random action_INDEX', action_ind)
        return action_array

        #     # action[i] = random.randint(0, self.actionbound - 1)
        #     if a_old[i] >0 and a_old[i] < 8:
        #         action_rand = random.randint(-1, 1)
        #         action[i] = a_old[i] + action_rand
        #     if a_old[i] == 0:
        #         action_rand = random.randint(0, 1)
        #         action[i] = a_old[i] + action_rand
        #     if a_old[i] == 8:
        #         action_rand = random.randint(-1, 0)
        #         action[i] = a_old[i] + action_rand
        #     # action_array[i] = action[i]
        # action_index = self._get_action_index(action)  # 这部分逻辑有问题，因为action index是由两个位置的随机数共同确定的
        # # 为什么不是随机选择 0-8 中的一个index， 然后生成动作的约束即可呢？
        # # print('random action index: ', action_index)
        # return action, action_index

    def _choose_action(self, state,act, epsilon, ):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        # old_act_index = self._get_action_index(a_old)
        if random.random() < epsilon:
            return self._random_action() # random action  注意 yellow time 不算action
        else:
            action = np.rint(self._Model.predict_one(state))
            action_arry = [0, 0, 0]
            action_index = np.argmax(self._Model.predict_one(state))  # 0 TO 87 GET ONE

            # print('NN action index: ',action_index)

            # action_arry = np.ones(self._action_dim, )

            # print('nn action:',action_arry, type(action_arry))

            # print('DQN action_INDEX', action_ind)
            return action_index #这部分要大改
            # the best action given the current state
        # 值得注意的是 虽然神经网络输入一直是 state， 但是网络的参数的更新， 是根据reward 变化的，
        # 在已经训练好网络之后，模型只需要输入 state 便可以预测action


    def from_a_to_mlv(self, a,v1_old,v2_old,v3_old):
        action_sel = self.action_sets[a]
        v1_new=v1_old + action_sel[0]*1.3889
        v2_new=v2_old + action_sel[1]*1.3889
        v3_new=v3_old + action_sel[2]*1.3889
        v1=self.maxmin(v1_new,12.5,22.2222)
        v2=self.maxmin(v2_new,12.5,22.2222)
        v3=self.maxmin(v3_new,12.5,22.2222)
        return v1,v2,v3
    # 1.3889 = 5 km/h     [40,45,50,55,60,65,70,75]   [0,1,2,3,4,5,6,7] num =8
    # a*2   [40,50,60,70,80]>[0,1,2,3,4]
    def maxmin(self,v,v_min,v_max):
        return max(min(v, max(v_min,v_max)), min(v_min,v_max))


    #####################  set speed limit  ####################
    def set_vsl(self, v1,v2,v3):
        number_of_lane1 = len(self.VSLlist1)
        number_of_lane2 = len(self.VSLlist2)
        number_of_lane3 = len(self.VSLlist3)
        # for j in range(number_of_lane):
        #     traci.lane.setMaxSpeed(self.VSLlist[j],float(v[j]))
        for j in range(number_of_lane1):
            traci.lane.setMaxSpeed(self.VSLlist1[j],float(v1))
        for j in range(number_of_lane2):
            traci.lane.setMaxSpeed(self.VSLlist2[j],float(v2))
        for j in range(number_of_lane3):
            traci.lane.setMaxSpeed(self.VSLlist3[j],float(v3))

    #####################  obtain state  ####################
    def get_step_state(self,act):
        # revised when with cav v2i  state = 23 lanes *2  46
        speed_list = []
        veh_lane_list = []
        vsl_list = []
        state = []
        occ = 0
        occ_list = []
        act_list=[]
        #act_list.append(act)
        act1 =self.action_sets[act]
        act_list.append(act1[0]+1)
        act_list.append(act1[1]+1)
        act_list.append(act1[2]+1)
        #for bn_lane in self.VSLlist:
             #speed_l = traci.lane.getLastStepMeanSpeed(bn_lane)
             #vsl_list.append(speed_l)

        # for edge in self.state_edge:
        for i in range(len(self.state_edge)):
            edge = self.state_edge[i]
            mean_speed_l = traci.edge.getLastStepMeanSpeed(edge)
            if mean_speed_l < 0:
                mean_speed_l = 0
            elif mean_speed_l > 0 and mean_speed_l <= 12.5:  #40
                mean_speed_l = 1
            elif mean_speed_l>12.5 and mean_speed_l<= 15.2778: # 50
                mean_speed_l = 2
            elif mean_speed_l >15.2778 and mean_speed_l <= 18.0556:  # 60
                mean_speed_l = 3
            elif mean_speed_l > 18.0556 and mean_speed_l <= 20.8333:  # 70
                mean_speed_l = 4
            elif mean_speed_l > 20.8333:
                mean_speed_l = 5                 
            speed_list.append(mean_speed_l)
            if edge=='R2':
                action_sel = self.action_sets[act]
                for j in range(len(self.VSLlist)):
                    lane=self.VSLlist[j]
                    vsl_speed=traci.lane.getLastStepMeanSpeed(lane)
                    if j==0 :
                        m=0
                    elif j==1 or j==2:
                        m=1
                    elif j==3 or j==4:
                        m=2
                    vsl_speed+=action_sel[m]*5
                    if vsl_speed < 0:
                        vsl_speed = 0
                    elif vsl_speed > 0 and vsl_speed <= 12.5:  #40
                        vsl_speed = 1
                    elif vsl_speed>12.5 and vsl_speed<= 15.2778: # 50
                        vsl_speed = 2
                    elif vsl_speed > 15.2778 and vsl_speed <= 18.0556:  # 60
                        vsl_speed = 3
                    elif vsl_speed > 18.0556 and vsl_speed <= 20.8333:  # 70
                        vsl_speed = 4
                    elif vsl_speed > 20.8333:
                        vsl_speed = 5
                    vsl_list.append(vsl_speed)

            veh_num_l = traci.edge.getLastStepVehicleNumber(edge)
            if veh_num_l < 0:
                veh_num_l = 0
            occ_edge = veh_num_l/self.state_edge_length[i]
            if occ_edge == 0:
                occ = 0
            if occ_edge>0 and occ_edge <=0.025:
                occ = 1
            elif occ_edge > 0.025 and occ_edge <= 0.05:  #40
                occ = 2
            elif occ_edge>0.05 and occ_edge<= 0.075: # 50
                occ = 3
            elif occ_edge > 0.075 and occ_edge <= 0.1:  # 60
                occ = 4
            elif occ_edge > 0.1 and occ_edge <= 0.125:  # 70
                occ = 5
            elif occ_edge > 0.125 and occ_edge <= 0.15:  # 70
                occ = 6
            elif occ_edge > 0.15 :
                occ = 7
            #if i == 'R2':
                #if act ==0:
                #    occ-=2
               # elif act == 1 or act==3 :
                #    occ-=1
                #elif act==2 or act==4 or act==6:
                #    occ+=0
                #elif act==5 or act ==7 :
                #    occ+=1
                #elif act== 8:
                #   occ+=2  
            occ_list.append(occ)
        '''mean_speed_2 = traci.edge.getLastStepMeanSpeed('E0')
        if mean_speed_2 < 0:
            mean_speed_2 = 0
        elif mean_speed_2 > 0 and mean_speed_2 <= 17.8816:  #40
            mean_speed_2 = 1
        elif mean_speed_2>17.8816 and mean_speed_2<= 22.352: # 50
            mean_speed_2 = 2
        elif mean_speed_2 > 22.352 and mean_speed_2 <= 26.8824:  # 60
            mean_speed_2 = 3
        elif mean_speed_2 > 26.8824 and mean_speed_2 <= 31.2928:  # 70
            mean_speed_2 = 4
        elif mean_speed_2 > 31.2928:
            mean_speed_2 = 5                 
        speed_list.append(mean_speed_2)'''
        # state = speed_list + veh_lane_list
        #state =  speed_list+ occ_list +act_list +vsl_list # 4+4+3+6=17
        state =  speed_list+ occ_list  +vsl_list + act_list #4+4+5+3=8
        #print(np.array(state))
        return np.array(state)


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)
        #      self._Memory.add_sample((current_state, action, reward_revised,  state_updated, ))c
        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, s_,  = b[0], b[1], b[2], b[3],  # extract data from one sample
                #  self._Memory.add_sample((current_state, action, self._reward,  state_updated, ))
                # print(action,a_index)
                # action_index = self._get_action_index(action)
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN

    def _close(self):
        traci.close()



    #####################  the out flow ####################
    def calc_outflow(self):
        state = []
        statef = []
        for detector in self.outID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            state.append(veh_num)
        for detector in self.inID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            statef.append(veh_num)
        return np.sum(np.array(state)) - np.sum(np.array(statef))
    #####################  the bottleneck speed ####################
    def calc_bottlespeed(self):
        speed = []
        for detector in self.bottleneck_detector:
            dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
            if dspeed < 0:
                dspeed = 5
                # The value of no-vehicle signal will affect the value of the reward
            speed.append(dspeed)
        return np.mean(np.array(speed))

    def get_from_traci(self):
        #获取一步动作（60秒）内的平均排队长度，总燃油消耗，总CO2排放，平均车辆加速度，总累计等待时间
        Queuelength = traci.edge.getLastStepHaltingNumber('R2') + traci.edge.getLastStepHaltingNumber('RL1')
        Fuelconsumption = traci.edge.getFuelConsumption('R2') + traci.edge.getFuelConsumption('RL1')
        C02emission = traci.edge.getCO2Emission('R2') + traci.edge.getCO2Emission('RL1')
        vehicle_list = traci.vehicle.getIDList()
        Waiting_time_store = []
        Acceleration_store = []
        for vehicle_id in vehicle_list:
                Acceleration = traci.vehicle.getAcceleration(vehicle_id)
                Waitingtime = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                Acceleration_store.append(Acceleration)
                Waiting_time_store.append(Waitingtime)

        Acceleration = np.mean(Acceleration_store)
        Waiting_time = sum(Waiting_time_store)


        return Queuelength, Fuelconsumption, C02emission, Acceleration, Waiting_time

    def _reset_none(self,):

        traci.start(self._sumo_cmd)
        print("start new Simulating...")
        self._step = 0
        self._reward = 0
        self._warmuptime(300)
        self.done = False

        self._ep_reward = 0
        self._outflow = 0
        self._bottle_speed = 0
        v1,v2,v3 = [22.352, 22.352, 22.352]
        reward = self.run_step_none()  # 初始速度 执行一个动作周期 60s


        self._reward = reward
        self.sum_pos_reward = 0


        return  self.done, self._step,v1,v2,v3

    def _run_none(self):

        reward = self.run_step_none()  # 初始速度 执行一个动作周期 60s
        reward_revised = reward

        self._reward = reward

        if reward_revised > 0:
            self.sum_pos_reward += reward_revised

        dw, done = self.done, self.done

        return reward_revised, dw, done

    def run_step_none(self):

        oflow = 0
        steps = self.control_horizon
        step_length=1
        steptodo = steps
        if (self._step + self.control_horizon) > self._max_steps:
            steps = self._max_steps - self._step
            steptodo  = steps

        while steptodo > 0:
            for i in range(int(1 / step_length)):
                traci.simulationStep()

            oflow = oflow + self.calc_outflow()

            self._step += 1
            if self._step == self._max_steps:
                self.done = True
                break
            steptodo -= 1

        reward = oflow

        return reward
