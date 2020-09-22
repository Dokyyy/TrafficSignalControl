import os
import sys
import pickle
from sumo.send_get_state import *
from xml.dom.minidom import parse
from multiprocessing import Process, Pool

# Carinfo_api_key = '2jHSIjomFCS=28ZDPlD0GO06c=o='
# Carinfo_device_ID = '606107785'
# record_api_key = '2jHSIjomFCS=28ZDPlD0GO06c=o='
# record_device_ID = '606108100'

Carinfo_api_key = 'eC36b1SgLIoXVWBIBYfy3m48kxU='
Carinfo_device_ID = '600643323'
record_api_key = 'IPR8QILg5Wz=MMs7Au9yf8i2EvU='
record_device_ID = '605651933'

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import numpy as np
import pandas as pd

from sumo.traffic_signal import TrafficSignal


class SumoEnv:
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, dic_env_conf, record_path=None, num_seconds=20000, start_ep=0):

        self.dic_env_conf = dic_env_conf
        self.node_file = dic_env_conf["DATA_PATH"]+'/grid.nod.xml'
        self._net = dic_env_conf["DATA_PATH"]+'/grid.net.xml'
        if dic_env_conf["route"] == 1:
            self._route_car = dic_env_conf["DATA_PATH"]+'/grid.rou3.car.xml'
            self._route_firetruck = dic_env_conf["DATA_PATH"]+'/grid.rou1.firetruck.xml'
            self._route = self._route_car+' , '+self._route_firetruck
        elif dic_env_conf["route"] == 2:
            self._route = dic_env_conf["DATA_PATH"] + '/rou2.xml'
        elif dic_env_conf["route"] == 3:
            self._route_car = dic_env_conf["DATA_PATH"] + '/grid.rou3.car.xml'
            self._route_firetruck = dic_env_conf["DATA_PATH"] + '/grid.rou3.firetruck.xml'
            self._route = self._route_car + ' , ' + self._route_firetruck
        elif dic_env_conf["route"] == 4:
            self._route_car = dic_env_conf["DATA_PATH"] + '/grid.rou4.car.xml'
            self._route_firetruck = dic_env_conf["DATA_PATH"] + '/grid.rou4.firetruck.xml'
            self._route = self._route_car + ' , ' + self._route_firetruck
        if record_path is not None:
            self.path_to_log = record_path
        else:
            self.path_to_log = dic_env_conf["LOG_PATH"]
        self.list_state_features = dic_env_conf["list_state_features"]
        self.use_gui = dic_env_conf["use_gui"]
        self.car_weight = dic_env_conf["car_weight"]
        self.firetruck_weight = dic_env_conf["firetruck_weight"]
        self.alpha = dic_env_conf["alpha"]
        self.avg_wait_time = dic_env_conf["avg_wait_time"]
        self.time_to_load_vehicles = dic_env_conf["time_to_load_vehicles"]  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = dic_env_conf["delta_time"]  # seconds on sumo at each step
        self.max_depart_delay = dic_env_conf["max_depart_delay"]  # Max wait time to insert a vehicle
        self.min_green = dic_env_conf["min_green"]
        self.max_green = dic_env_conf["max_green"]
        self.yellow_time = dic_env_conf["yellow_time"]
        self.single_agent = dic_env_conf["single_agent"]
        self.adjacency_use_distance = dic_env_conf["adjacency_use_distance"]
        self.reward_function = dic_env_conf["reward_function"]
        self.use_all_avg = dic_env_conf["use_all_avg"]

        self.sim_max_time = num_seconds
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.ts_ids = traci.trafficlight.getIDList()
        #traffic signal id to index
        self.ts_id_2_index = {}
        self.ts_index_2_id = {}
        counter_ts = 0
        for ts in self.ts_ids:
            self.ts_id_2_index[ts] = counter_ts
            self.ts_index_2_id[counter_ts] = ts
            counter_ts += 1
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_ids[0])))
        self.ts_position = self.get_ts_position()
        self.traffic_signals = dict()

        self.vehicles = dict()
        self.last_measure = dict()  # used to reward function remember last measure
        self.last_measure_car = dict()
        self.last_measure_firetruck = dict()
        self.last_measure_car_all = dict()
        self.last_measure_firetruck_all = dict()
        self.last_reward = [0]*9

        self.phases = self.convert_phases(dic_env_conf["phases"])
        self.num_green_phases = len(self.phases)

        self.top_k = dic_env_conf["top_k"]
        self.adjacency_matrix = {}

        self.run = start_ep
        self.metrics = []

        traci.close()

    def convert_phases(self, phases_list):
        # phases = []
        # for phase in phases_list:
        #     phase_str = ''
        #     for s in phase:
        #         phase_str += 'G' if s==1 else 'r'
        #     p = traci.trafficlight.Phase(self.max_green, phase_str)
        #     phases.append(p)
        # yellow = traci.trafficlight.Phase(self.yellow_time, 'yyyyyyyyyyyyyyyy')
        # phases.append(yellow)
        phases = [traci.trafficlight.Phase(self.max_green, 'GGGgrrrrGGGgrrrr'),
                  traci.trafficlight.Phase(self.yellow_time, 'yyygrrrryyygrrrr'),
                  traci.trafficlight.Phase(self.max_green, 'rrrGrrrrrrrGrrrr'),
                  traci.trafficlight.Phase(self.yellow_time, 'rrryrrrrrrryrrrr'),
                  traci.trafficlight.Phase(self.max_green, 'rrrrGGGgrrrrGGGg'),
                  traci.trafficlight.Phase(self.yellow_time, 'rrrryyygrrrryyyg'),
                  traci.trafficlight.Phase(self.max_green, 'rrrrrrrGrrrrrrrG'),
                  traci.trafficlight.Phase(self.yellow_time, 'rrrrrrryrrrrrrry')]
        return phases

    def get_ts_position(self):
        domTree = parse(self.node_file)
        #文档根元素
        rootNode = domTree.documentElement

        nodes = rootNode.getElementsByTagName('node')
        dic_ts_position = {}
        for node in nodes:
            if node.getAttribute('type') == 'traffic_light':
                dic_ts_position[node.getAttribute('id')]={'x': float(node.getAttribute('x')),
                                                          'y': float(node.getAttribute('y'))}
        return dic_ts_position

    def get_adjacency_maxtrix(self):
        ts_adjacency_matrix = {}
        for ts1 in self.ts_ids:
            location1 = self.ts_position[ts1]
            if self.adjacency_use_distance == True: #使用距离
                row = np.array([0]*len(self.ts_ids))
                for ts2 in self.ts_ids:
                    location2 = self.ts_position[ts2]
                    dist = traci.simulation.getDistance2D(x1=location1['x'], y1=location1['y'],
                                                          x2=location2['x'], y2=location2['y'])#通过道路连接的距离
                    row[self.ts_id_2_index[ts2]] = dist
                if len(row) == self.top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:self.top_k].tolist()
                elif len(row) > self.top_k:
                    adjacency_row_unsorted = np.argpartition(row, self.top_k)[:self.top_k].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(len(self.ts_ids))]
                adjacency_row_unsorted.remove(self.ts_id_2_index[ts1])
                ts_adjacency_matrix[ts1] = [self.ts_id_2_index[ts1]] + adjacency_row_unsorted
            else: # use connection information
                ts_adjacency_matrix[ts1] = [self.ts_id_2_index[ts1]]
                for j in self.traffic_signals[ts1].get_neighbor():
                    if j is not None:
                        ts_adjacency_matrix[ts1].append(self.ts_id_2_index[j])
                    else:
                        ts_adjacency_matrix[ts1].append(-1)
        return ts_adjacency_matrix

    def reset(self):
        self.run += 1
        self.metrics = []

        sumo_cmd = [self._sumo_binary,
                    '-n', self._net,
                    '-r', self._route,
                    '--max-depart-delay', str(self.max_depart_delay),
                    '--waiting-time-memory', '10000',
                    '--time-to-teleport', '-1',
                    '--random']
        if self.use_gui:
            sumo_cmd.append('--start')
        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green,
                                                     self.phases)
            self.last_measure[ts] = 0.0
            self.last_measure_car[ts] = 0.0
            self.last_measure_firetruck[ts] = 0.0
            self.last_measure_car_all[ts] = 0.0
            self.last_measure_firetruck_all[ts] = 0.0
        self.adjacency_matrix = self.get_adjacency_maxtrix()

        self.vehicles = dict()

        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()

        if self.single_agent:
            return self._compute_obs_inlanes()[self.ts_ids[0]]
        else:
            return self._compute_obs_inlanes()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()  # milliseconds to seconds

    def step(self, action, send_carinfo):
        # act
        self._apply_actions(action)

        for i in range(self.yellow_time):
            self._sumo_step()
        for ts in self.ts_ids:
            self.traffic_signals[ts].update_phase()
        for j in range(self.delta_time - self.yellow_time):
            self._sumo_step()

        observation = self._compute_obs_inlanes(send_carinfo)
        reward = self._compute_rewards()
        self.last_reward = reward
        total_reward = sum(reward)
        done = {'__all__': self.sim_step > self.sim_max_time}

        info = self._compute_step_info()
        # 上传性能数据
        Send_RecordInfo(info, record_api_key, record_device_ID)
        self.metrics.append(info)

        if self.single_agent:
            return observation[self.ts_ids[0]], reward[self.ts_ids[0]], done['__all__'], total_reward
        else:
            return observation, reward, done['__all__'], total_reward

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for i in range(len(actions)):
                self.traffic_signals[self.ts_index_2_id[i]].set_next_phase(actions[i])

    def _compute_obs_inlanes(self, send_carinfo = 0):
        observations = []
        Car_Info = [[] for i in range(6)]
        for ts in self.ts_ids:
            obs = {}
            for feature in self.list_state_features:
                if feature == 'cur_phase':
                    obs[feature] = [self.traffic_signals[ts].phase]
                elif feature == 'lane_num_vehicle':
                    obs[feature] = self.traffic_signals[ts].get_lane_num_vehicle()
                elif feature == 'adjacency_matrix':
                    obs[feature] = self.adjacency_matrix[ts]
                elif feature == 'map_inlanes':
                    p_state = np.zeros((12, 43, 2))
                    lanes = getattr(self.traffic_signals[ts], 'lanes')
                    for p in range(self.traffic_signals[ts].num_green_phases):
                        veh_list = self.traffic_signals[ts].get_veh_list(p)
                        for veh in veh_list:
                            veh_Type = traci.vehicle.getTypeID(veh)
                            lane_id = traci.vehicle.getLaneID(veh)
                            veh_lane = self.traffic_signals[ts].get_edge_id(lane_id)
                            lane_index = lanes.index(lane_id)
                            acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                            # 计算的是车辆在某一个道路上的等待时间
                            if veh not in self.vehicles:
                                waiting_time = acc
                                self.vehicles[veh] = {veh_lane: acc}
                            else:
                                waiting_time = acc - sum(
                                    [self.vehicles[veh][lane] for lane in self.vehicles[veh].keys() if
                                     lane != veh_lane])
                                self.vehicles[veh][veh_lane] = waiting_time
                            lane_pos = traci.lane.getLength(lane_id)-traci.vehicle.getLanePosition(veh)
                            if send_carinfo == 1:
                                Car_Info[0].append(veh)
                                Car_Info[1].append(veh_Type)
                                Car_Info[2].append(veh_lane)
                                Car_Info[3].append(lane_index)
                                Car_Info[4].append(waiting_time)
                                Car_Info[5].append(lane_pos)

                            if int(lane_pos / 7) >= 0 and int(lane_pos / 7) <= 42:
                                if traci.vehicle.getTypeID(veh) == 'FireTruck':
                                    p_state[lane_index, int(lane_pos / 7)] = [self.firetruck_weight, self.vehicles[veh][veh_lane]]
                                else:
                                    p_state[lane_index, int(lane_pos / 7)] = [self.car_weight, self.vehicles[veh][veh_lane]]
                    p_state = np.reshape(p_state, [-1, 12*43*2])
                    obs[feature] = p_state
            observations.append(obs)

        if send_carinfo == 1: # 上传车辆数据
            Send_CarInfo(Car_Info, Carinfo_api_key, Carinfo_device_ID)
        return observations

    def _compute_observations(self, with_queue_length=False):
        """
        Return the current observation for each traffic signal
        """
        observations = []
        for ts in self.ts_ids:
            obs = {}
            for feature in self.list_state_features:
                if feature=='cur_phase':
                    obs[feature] = [self.traffic_signals[ts].phase]
                elif feature == 'lane_num_vehicle':
                    obs[feature] = self.traffic_signals[ts].get_lane_num_vehicle()
                elif feature == 'adjacency_matrix':
                    obs[feature] = self.adjacency_matrix[ts]
            if with_queue_length:
                obs['lane_num_vehicle_been_stopped_thres1'] = self.traffic_signals[ts].get_lane_stop_vehicle_num()
            observations.append(obs)
        return observations

    def _compute_rewards(self):
        if self.avg_wait_time:
            return self._avg_waiting_time_reward()
        else:
            return self._waiting_time_reward()

    def _queue_average_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            new_average = np.mean(self.traffic_signals[ts].get_stopped_vehicles_num())
            rewards[ts] = self.last_measure[ts] - new_average
            self.last_measure[ts] = new_average
        return rewards

    def _queue_reward(self):
        rewards = []
        for ts in self.ts_ids:
            rewards.append(- (sum(self.traffic_signals[ts].get_stopped_vehicles_num())) * 0.25)
        return rewards

    def _avg_waiting_time_reward(self):
        rewards = list()
        avg_wt_car_pre_all = np.mean([self.last_measure_car[ts] for ts in self.ts_ids])
        avg_wt_firetruck_pre_all = np.mean([self.last_measure_firetruck[ts] for ts in self.ts_ids])
        for ts in self.ts_ids:
            avg_wt_car_all, avg_wt_firetruck_all = self.traffic_signals[ts].get_avg_wait_time_priority_all()
            avg_wt_car, avg_wt_firetruck = self.traffic_signals[ts].get_avg_wait_time_priority()
            if self.reward_function==0:
                if self.use_all_avg:
                    r = -(1-self.alpha) * avg_wt_car_all - self.alpha * avg_wt_firetruck_all
                else:
                    r = -(1-self.alpha) * avg_wt_car - self.alpha * avg_wt_firetruck
            elif self.reward_function==1:
                if self.use_all_avg:
                    r = (1-self.alpha) * (self.last_measure_car_all[ts]-avg_wt_car_all) + \
                          self.alpha * (self.last_measure_firetruck_all[ts]-avg_wt_firetruck_all)
                else:
                    r = (1-self.alpha) * (self.last_measure_car[ts]-avg_wt_car) + \
                              self.alpha * (self.last_measure_firetruck[ts]-avg_wt_firetruck)

            self.last_measure[ts] = r
            self.last_measure_car[ts] = avg_wt_car
            self.last_measure_firetruck[ts] = avg_wt_firetruck
            self.last_measure_car_all[ts] = avg_wt_car_all
            self.last_measure_firetruck_all[ts] = avg_wt_firetruck_all
            rewards.append(r)
        return rewards

    def _waiting_time_reward(self):
        rewards = list()
        for ts in self.ts_ids:
            ts_wait_time_car, ts_wait_time_firetruck = self.traffic_signals[ts].get_waiting_time_priority()
            sum_wt_car = sum(ts_wait_time_car)
            sum_wt_firetruck = sum(ts_wait_time_firetruck)
            if self.reward_function==0:
                r = -(1-self.alpha)*sum_wt_car - self.alpha*sum_wt_firetruck
            elif self.reward_function==1:
                r = (1-self.alpha)*(self.last_measure_car[ts]-sum_wt_car) + self.alpha*(self.last_measure_firetruck[ts]-sum_wt_firetruck)
            self.last_measure[ts] = r
            self.last_measure_car[ts] = sum_wt_car
            self.last_measure_firetruck[ts] = sum_wt_firetruck
            rewards.append(r)
        return rewards

    def _waiting_time_reward2(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            self.last_measure[ts] = ts_wait
            if ts_wait == 0:
                rewards[ts] = 1.0
            else:
                rewards[ts] = 1.0 / ts_wait
        return rewards

    def _waiting_time_reward3(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards[ts] = -ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        if self.avg_wait_time:
            return {
                'step_time': self.sim_step,
                'reward': sum(self.last_reward),
                'total_stopped': sum([sum(self.traffic_signals[ts].get_stopped_vehicles_num()) for ts in self.ts_ids]),
                'avg_wait_time_car': np.mean([self.last_measure_car[ts] for ts in self.ts_ids]),
                'avg_wait_time_firetruck': np.mean([self.last_measure_firetruck[ts] for ts in self.ts_ids]),
                'avg_all_wt_car': np.mean([self.last_measure_car_all[ts] for ts in self.ts_ids]),
                'avg_all_wt_firetruck': np.mean([self.last_measure_firetruck_all[ts] for ts in self.ts_ids]),
            }
        else:
            return {
                'step_time': self.sim_step,
                'reward': sum(self.last_reward),
                'total_stopped': sum([sum(self.traffic_signals[ts].get_stopped_vehicles_num()) for ts in self.ts_ids]),
                'sum_wait_time_car': np.mean([self.last_measure_car[ts] for ts in self.ts_ids]),
                'sum_wait_time_firetruck': np.mean([self.last_measure_firetruck[ts] for ts in self.ts_ids])
            }

    def close(self):
        traci.close()
        # print('sumo environment closed')

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def save_csv_file(self):
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
        self._save_csv(self.path_to_log, self.run)

    def _save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '/run{}'.format(run) + '.csv', index=False)


if __name__=='__main__':
    dataPath = os.path.join("data", "grid_3_3")
    logPath = os.path.join('records')
    list_state_features = ['map_inlanes', 'adjacency_matrix']
    phases = [[1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0],
            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]]
    env = SumoEnv(logpath=logPath, list_state_features=list_state_features, datapath=dataPath, use_gui=True,
                  phases=phases, top_k=5, car_weight=1, firetruck_weight=5, avg_wait_time=True)
    state = env.reset()
    state, reward, done, _ = env.step([2,2,2,2,2,2,2,2,0])
    state, reward, done, _ = env.step([3, 2, 2, 2, 3, 2, 2, 2, 2])
    state, reward, done, _ = env.step([3, 3, 3, 3, 3, 3, 3, 3, 3])
    print(reward)

