from sumo.sumo_env import SumoEnv
from sumo.send_get_state import *
from sumo.send_get_action import *
import os
import numpy as np
import re

# action_api_key = '2jHSIjomFCS=28ZDPlD0GO06c=o='
# action_device_ID = '605460850'
# state_api_key = '2jHSIjomFCS=28ZDPlD0GO06c=o='
# state_device_ID = '605460941'

state_api_key = 'd3z94YsdZbFmo1So4dOscjDHb2w='
state_device_ID = '603595007'
action_api_key = 'VknSLNCjwhPXJg13IPRxxaaunzU='
action_device_ID = '603264724'

def run_with_static_logic(DIC_ENV_CONF, recordPath, max_epLen, start_ep=0):
    record_path = os.path.join(recordPath, "static_logic")
    max_len_per_epi = max_epLen
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    config_file = open(os.path.join(record_path, 'config.txt'), 'w')
    config_file.write("ENV_CONFIG:\n")
    for key in DIC_ENV_CONF.keys():
        config_file.write(key + " : " + str(DIC_ENV_CONF.get(key))+'\n')
    config_file.write("\n\n")
    config_file.close()

    env = SumoEnv(dic_env_conf=DIC_ENV_CONF, record_path=record_path, start_ep=start_ep)
    total_steps = 0
    j = 0
    rAll = 0
    done = False
    state = env.reset()
    action1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    action2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    action3 = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    action4 = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    action_list = [action1, action1, action1,
                   action2, action2, action2,
                   action3, action3, action3,
                   action4, action4, action4,]
    while j < max_len_per_epi and not done:
        state_, reward, done, total_reward = env.step(action_list[j%len(action_list)])
        rAll += total_reward
        j += 1
        total_steps += 1
        state = state_
    env.save_csv_file()
    env.close()
    print("total reward for static logic: %f-----"%(rAll))


def run_with_network3(memo, start_ep=0, reward_function=0, alpha=0.5, rou=1, avg_wt=True, use_avg_all=True, double=True,
                      lr=0.0001, copy_cloud_data = 0, send_carinfo = 1):
    DIC_EXP_CONF = {
        "model_path": os.path.join("saved_model", memo),
        "record_path": os.path.join("records", memo),
        "num_episode": 1,
        "max_len_per_episode": 64,
    }
    DIC_ENV_CONF = {
        "car_weight": 1,
        "firetruck_weight": 10,
        "alpha": alpha,
        "reward_function": reward_function,
        "avg_wait_time": avg_wt,
        "use_all_avg": use_avg_all,
        "len_feature": 12 * 43 * 2,
        "num_intersections": 9,
        "route": rou,
        "DATA_PATH": os.path.join("sumo", "data", "grid_3_3"),
        "LOG_PATH": os.path.join(DIC_EXP_CONF["record_path"], "csv_record"),
        "list_state_features": ['map_inlanes', 'adjacency_matrix'],
        "use_gui": True,
        "time_to_load_vehicles": 0,
        "delta_time": 10,
        "max_depart_delay": 1000,
        "min_green": 5,
        "max_green": 60,
        "yellow_time": 4,
        "single_agent": False,
        "adjacency_use_distance": False,
        "top_k": 5,
        "phases": [[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]],
    }
    DIC_AGENT_CONF = {
        "MLP_layers": [256, 64, 32],
        "double": double,
        "dueling": False,
        "LEARNING_RATE": lr,
        "BATCH_SIZE": 64,
        "gamma": 0.8,
        "MAX_MEMORY_LEN": 2000,
        "epsilon_max": 1,
        "epsilon_decay": 0.95,
        "epsilon_min": 0.01,
        "LOSS_FUNCTION": "_huber_loss",
        # "LOSS_FUNCTION": "mse",
        "NORMAL_FACTOR": 1,
    }

    if not os.path.exists(DIC_EXP_CONF["record_path"]):
        os.makedirs(DIC_EXP_CONF["record_path"])
    config_file = open(os.path.join(DIC_EXP_CONF["record_path"], 'config.txt'), 'w')
    config_file.write("run_with_DQN_network\n\n")
    config_file.write("EXP_CONFIG:\n")
    for key in DIC_EXP_CONF.keys():
        config_file.write(key + " : " + str(DIC_EXP_CONF.get(key)) + '\n')
    config_file.write("\n\n")
    config_file.write("AGENT_CONFIG:\n")
    for key in DIC_AGENT_CONF.keys():
        config_file.write(key + " : " + str(DIC_AGENT_CONF.get(key)) + '\n')
    config_file.write("\n\n")
    config_file.write("ENV_CONFIG:\n")
    for key in DIC_ENV_CONF.keys():
        config_file.write(key + " : " + str(DIC_ENV_CONF.get(key)) + '\n')
    config_file.write("\n\n")
    config_file.close()

    # if start_ep==0:
    #     run_with_static_logic(DIC_ENV_CONF, recordPath=DIC_EXP_CONF["record_path"],
    #                           max_epLen=DIC_EXP_CONF["max_len_per_episode"])

    env = SumoEnv(dic_env_conf=DIC_ENV_CONF,start_ep=start_ep)

    for i in range(start_ep, DIC_EXP_CONF["num_episode"]):
        state = env.reset()
        state_temp = [[] for i in range(2)]
        for i in range(len(state)):
            state_temp[0].append(state[i]['adjacency_matrix'])
            state_temp[1].append(list(state[i]['map_inlanes'][0]))
        send_state(str(state_temp), state_device_ID, state_api_key)  # sumo端发送state至onenet
        done = False
        rAll = 0
        step = 0

        # print("Episode: %d---------" % (i))
        while step < DIC_EXP_CONF["max_len_per_episode"] and not done:
            # print('------------------------------sumo {}------------------------------'.format(step))
            datastream_id, action_sumo_str = get_action(step, action_device_ID, action_api_key,
                                                    copy_cloud_data)  # onenet下发action至sumo端
            while len(action_sumo_str) == 0:  # 如果接收的action是空，则重复申请接收
                # print('get action again!')
                datastream_id, action_sumo_str = get_action(step, action_device_ID, action_api_key, copy_cloud_data)
            delete_action(datastream_id, action_device_ID, action_api_key)
            pattern = re.compile(r'\d{1}', re.S)
            action_sumo = pattern.findall(action_sumo_str)
            for i in range(len(action_sumo)):
                action_sumo[i] = int(action_sumo[i])
            action_sumo = np.array(action_sumo)

            # print(action_sumo)

            state_, reward, done, total_reward = env.step(action_sumo, send_carinfo)
            state_temp = [[] for i in range(2)]
            for i in range(len(state_)):
                state_temp[0].append(state_[i]['adjacency_matrix'])
                state_temp[1].append(list(state_[i]['map_inlanes'][0]))

            datastream_id, state = get_state(step, state_device_ID, state_api_key, copy_cloud_data)  # 控制端从onenet获取state
            while len(state) != 0:  # 如果上一次的state还没接收，则暂时不发送
                datastream_id, state = get_state(step, state_device_ID, state_api_key, copy_cloud_data)
            send_state(str(state_temp), state_device_ID, state_api_key)  # sumo端发送state至onenet

            rAll += total_reward
            step += 1

        env.save_csv_file()
        env.close()

def run_sumo():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datastream_id, action_sumo = get_action(-1, action_device_ID, action_api_key, copy_cloud_data=0)
    delete_action(datastream_id, action_device_ID, action_api_key)
    run_with_network3("0616net3_double_rou4_reward1_alpha0.6_avgwt", start_ep=0, reward_function=1,
                      alpha=0.6, rou=4, avg_wt=True, use_avg_all=False, double=True, copy_cloud_data = 1, send_carinfo = 1)

if __name__=="__main__":
    run_sumo()