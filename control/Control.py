from control.network3 import MultiLightAgent3
from control.send_get_action import *
from control.send_get_state import *
import os

# state_device_ID = "605460941"
# state_api_key = "2jHSIjomFCS=28ZDPlD0GO06c=o="
# action_device_ID = "605460850"
# action_api_key = "2jHSIjomFCS=28ZDPlD0GO06c=o="

state_device_ID = "603595007"
state_api_key = "d3z94YsdZbFmo1So4dOscjDHb2w="
action_device_ID = "603264724"
action_api_key = "VknSLNCjwhPXJg13IPRxxaaunzU="

def control(memo, start_ep=0, reward_function=0, alpha=0.5, rou=1, avg_wt=True, use_avg_all=True, double=True,
                      lr=0.0001, copy_cloud_data = 0):
    DIC_EXP_CONF = {
        "model_path": os.path.join("saved_model", memo),
        "record_path": os.path.join("records", memo),
        "num_episode": 1,
        "max_len_per_episode": 36,
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
        "DATA_PATH": os.path.join("data", "grid_3_3"),
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

    # env = SumoEnv(dic_env_conf=DIC_ENV_CONF, start_ep=0)
    controller = MultiLightAgent3(len_feature=DIC_ENV_CONF["len_feature"], num_actions=4,
                                  num_intersections=9, num_lanes=12, dic_agent_conf=DIC_AGENT_CONF,
                                  path_to_model=DIC_EXP_CONF["model_path"], start_ep=start_ep)
    # controller.load(DIC_EXP_CONF["model_name"])
    # controller.load_bar(DIC_EXP_CONF['target_model_name'])
    action_list = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    step = 0
    while True:
        # 控制端从onenet获取state
        # print('------------------------------control {}------------------------------'.format(step))
        datastream_id, state = get_state(step, state_device_ID, state_api_key, copy_cloud_data)  # 控制端从onenet获取state
        while len(state) == 0: # 如果接收的state是空，则重复申请接收
            # print('get state again!')
            datastream_id, state = get_state(step, state_device_ID, state_api_key, copy_cloud_data)
        delete_state(datastream_id, state_device_ID, state_api_key)

        # # static logic
        # action_control = action_list[step%len(action_list)]

        # 控制端生成action
        action_control = controller.act(state)
        # controller.memorize(state, action_control, reward, state_, done)

        # print(action_control)


        # 控制端上传action至onenet
        datastream_id, action_sumo = get_action(step, action_device_ID, action_api_key)
        while len(action_sumo) !=0: # 如果上一次的action还没处理，则暂时不发送
            datastream_id, action_sumo = get_action(step, action_device_ID, action_api_key)
        send_action(str(action_control), action_device_ID, action_api_key)  # 控制端传输action至onenet

        if len(controller.memory) > DIC_AGENT_CONF["BATCH_SIZE"]:
            controller.replay(DIC_AGENT_CONF["BATCH_SIZE"])

        step += 1

def run_control():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    control("0616net3_double_rou4_reward1_alpha0.6_avgwt", start_ep=0, reward_function=1,
                      alpha=0.6, rou=4, avg_wt=True, use_avg_all=False, double=True, copy_cloud_data = 1)

if __name__ == '__main__':
    run_control()