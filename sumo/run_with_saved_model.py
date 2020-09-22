import os
from sumo.sumo_env import SumoEnv
from control.network3 import MultiLightAgent3
from keras.models import load_model
from keras import backend as K
import tensorflow as tf


def run_with_network3(memo, model_name, start_ep=0, reward_function=0, alpha=0.5, rou=1, avg_wt=True, use_avg_all=True, double=True,
                      lr=0.0001):
    DIC_EXP_CONF = {
        "model_path": os.path.join("saved_model", memo),
        "record_path": os.path.join("saved_model", memo, "records"),
        "saved_model": os.path.join("saved_model", memo) + "/"+model_name,
        "num_episode": 1,
        "max_len_per_episode": 360,
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
        "epsilon_max": 0.01,
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

    env = SumoEnv(dic_env_conf=DIC_ENV_CONF,start_ep=start_ep)
    controller = MultiLightAgent3(len_feature=DIC_ENV_CONF["len_feature"], num_actions=4,
                                  num_intersections=9, num_lanes=12, dic_agent_conf=DIC_AGENT_CONF,
                                  path_to_model=DIC_EXP_CONF["model_path"], start_ep=start_ep)
    controller.load(DIC_EXP_CONF["saved_model"])

    for i in range(start_ep, DIC_EXP_CONF["num_episode"]):
        state = env.reset()
        done = False
        rAll = 0
        j = 0

        print("Episode: %d---------" % (i))
        while j < DIC_EXP_CONF["max_len_per_episode"] and not done:
            action = controller.act(state)
            state_, reward, done, total_reward = env.step(action)
            # controller.memorize(state, action, reward, state_, done)
            rAll += total_reward
            j += 1
            state = state_
            # if len(controller.memory) > DIC_AGENT_CONF["BATCH_SIZE"]:
            #     controller.replay(DIC_AGENT_CONF["BATCH_SIZE"])
        env.save_csv_file()
        env.close()
        # controller.update_target_model()
        # print("episode: {}, e: {}"
        #       .format(i, controller.epsilon))
        # if i > start_ep and i%5==0:
        #     controller.save_network("round_{0}".format(i))
        #     controller.save_network_bar("round_{0}_target".format(i))
        print("total reward: %f-----" % (rAll))
    print("experiment for %s end!" % (memo))

def _huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta

    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

if __name__=='__main__':
    run_with_network3("0616net3_double_rou4_reward1_alpha0.6_avgwt", "round_200.h5", rou=4)