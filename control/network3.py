import tensorflow as tf
import numpy as np
import time
import os
import random
from collections import deque
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge, Subtract, Add
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model, model_from_json, load_model
from keras.layers.core import Activation
from keras.utils import np_utils, to_categorical
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, TensorBoard


#without neighbor communication
class MultiLightAgent3:
    def __init__(self, len_feature=12*30*2, num_actions=4, num_intersections=9,  num_lanes=12,
                 dic_agent_conf=None, path_to_model=None, start_ep=0):
        self.dic_agent_conf = dic_agent_conf
        self.path_to_model = path_to_model
        self.len_feature = len_feature
        self.num_actions = num_actions
        self.num_agents = num_intersections
        self.num_lanes = num_lanes

        self.double = dic_agent_conf['double']
        self.dueling = dic_agent_conf["dueling"]
        self.epsilon = dic_agent_conf["epsilon_max"]
        self.epsilon_min = dic_agent_conf["epsilon_min"]
        self.epsilon_decay = dic_agent_conf["epsilon_decay"]
        self.gamma = dic_agent_conf["gamma"]

        self.memory = deque(maxlen=self.dic_agent_conf["MAX_MEMORY_LEN"])
        self.model = self.build_network(MLP_layers=self.dic_agent_conf["MLP_layers"])
        self.target_model = self.build_network(MLP_layers=self.dic_agent_conf["MLP_layers"])
        if start_ep!=0:
            self.model.load_weights(os.path.join(self.path_to_model, "round_{}.h5".format(start_ep)))
            self.target_model.load_weights(os.path.join(self.path_to_model, "round_{}_target.h5".format(start_ep)))
            self.epsilon = max(self.epsilon_min, self.epsilon * pow(self.epsilon_decay, 360 * (start_ep - 1)))
        else:
            self.update_target_model()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def MLP(self, In_0, layers=[256, 64]):
        """
                Currently, the MLP layer
                -input: [batch,#agents,feature_dim]
                -outpout: [batch,#agents,128]
                """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(h)

        return h

    """Huber loss for Q Learning
        References: https://en.wikipedia.org/wiki/Huber_loss
                    https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
        """
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def build_network(self, MLP_layers=[32, 32]):
        start_time = time.time()
        In = Input(shape=[self.num_agents, self.len_feature], name="feature")
        """
                #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
                ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
                """
        """
                Currently, the MLP layer 
                -input: [batch,agent,feature_dim]
                -outpout: [#agent,batch,128]
                """
        feature = self.MLP(In, MLP_layers)

        # action prediction layer
        # [batch,agent,32]->[batch,agent,action]
        if self.dic_agent_conf["dueling"]:
            value = Dense(MLP_layers[-1], activation="relu")(feature)
            value = Dense(1, activation="relu")(value)
            advantage = Dense(MLP_layers[-1], activation="relu")(feature)
            advantage = Dense(self.num_actions, activation="relu")(advantage)
            advantage_mean = Lambda(lambda x: K.mean(x, axis=-1))(advantage)
            advantage = Subtract()([advantage, advantage_mean])
            out = Add()([value, advantage])
            # h = Dense(self.num_actions + 1, activation='linear')(feature)
            # out = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
            #              output_shape=((self.num_agents, self.num_actions),))(h)
        else:
            out = Dense(self.num_actions, kernel_initializer='random_normal', name='action_layer')(feature)
        # out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model = Model(inputs=In, outputs=out)

        if self.dic_agent_conf["LOSS_FUNCTION"]=='mse':
            model.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]), loss='mse')
        else:
            model.compile(
                    optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                    loss=self._huber_loss,)

        # model.summary()
        network_end = time.time()
        # print('total time:', network_end - start_time)
        return model

    def act(self, state):
        feature, adj = self.prepare_state([state])
        action = self.model.predict(feature)
        max_action = np.expand_dims(np.argmax(action, axis=-1), axis=-1)
        random_action = np.reshape(np.random.randint(self.num_actions, size=1 * self.num_agents),
                                   (1, self.num_agents, 1))
        # [batch,agent,2]
        possible_action = np.concatenate([max_action, random_action], axis=-1)
        selection = np.random.choice(
            [0, 1],
            size=self.num_agents,
            p=[1 - self.epsilon, self.epsilon])
        act = possible_action.reshape((self.num_agents, 2))[
            np.arange(self.num_agents), selection]
        # act = np.reshape(act, (1, self.num_agents))
        return act
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(self.num_actions, size=1 * self.num_agents)
        # else:
        #     feature, adj = self.prepare_state([state])
        #     action = self.model.predict(feature)
        #     # max_action = np.expand_dims(np.argmax(action, axis=-1), axis=-1)
        #     max_action = np.argmax(action, axis=-1)[0]
        #     return max_action

    def prepare_state(self, state):
        total_features = list()
        total_adjs = list()
        for i in range(len(state)):
            feature = []
            adj = []
            for j in range(self.num_agents):
                observation = []
                for feature_name in ['map_inlanes', 'adjacency_matrix']:
                    if 'adjacency' in feature_name:
                        continue
                    if feature_name == "cur_phase":
                        if len(state[i][j][feature_name]) == 1:
                            # choose_action
                            observation.extend(
                                self.dic_traffic_env_conf['PHASE'][state[i][j][feature_name][0]])
                        else:
                            observation.extend(state[i][j][feature_name])
                    elif feature_name == "map_inlanes":
                        observation.extend(state[i][j][feature_name])
                feature.append(observation)
                adj.append(state[i][j]['adjacency_matrix'])
            total_features.append(feature)
            total_adjs.append(adj)
        total_features = np.reshape(np.array(total_features), [len(state), self.num_agents, -1])
        total_adjs = self.adjacency_index2matrix(np.array(total_adjs))
        return total_features, total_adjs

    def replay2(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            _feature, q_values = self.action_att_predict([state])
            _next_features, q_eval4target = self.action_att_predict([next_state])
            # target_q_values:[batch,agent,action]
            _, target_q_values = self.action_att_predict(
                [next_state],
                total_features=_next_features,
                bar=True)
            for j in range(self.num_agents):
                if self.double:
                    max_act4next = np.argmax(q_eval4target[0][j])
                    select_q_target = target_q_values[0][j][max_act4next]
                else:
                    select_q_target = np.max(target_q_values[0][j])
                q_values[0][j][action[j]] = reward[j]/ self.dic_agent_conf["NORMAL_FACTOR"] + self.gamma * select_q_target
            self.model.fit(_feature, q_values, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        _state = []
        _next_state = []
        _action = []
        _reward = []
        i = 0
        for state, action, reward, next_state, done in minibatch:
            _state.append([])
            _action.append([])
            _next_state.append([])
            _reward.append([])
            for j in range(self.num_agents):
                _state[i].append(state[j])
                _action[i].append(action[j])
                _next_state[i].append(next_state[j])
                _reward[i].append(reward[j])
            i += 1
        _features, q_values= self.action_att_predict(_state)
        _next_features, q_eval4target = self.action_att_predict(_next_state)
        # target_q_values:[batch,agent,action]
        _, target_q_values = self.action_att_predict(
            _next_state,
            total_features=_next_features,
            bar=True)
        # _, q_eval4target = self.action_att_predict(
        #     _next_state,
        #     total_features=_next_features
        # )

        for i in range(batch_size):
            for j in range(self.num_agents):
                if self.double:
                    max_act4next = np.argmax(q_eval4target[i][j])
                    select_q_target = target_q_values[i][j][max_act4next]
                else:
                    select_q_target = np.max(target_q_values[i][j])
                q_values[i][j][_action[i][j]] = _reward[i][j]/ self.dic_agent_conf["NORMAL_FACTOR"] + self.gamma * select_q_target

        # self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features]
        self.Y = q_values.copy()
        self.Y_total = [q_values.copy()]
        self.model.fit(self.Xs, self.Y_total, epochs=1, verbose=0)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def action_att_predict(self, state, total_features=[], bar=False):
        # state:[batch,agent,features and adj]
        # return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size = len(state)
        if total_features == [] :
            total_features= list()
            for i in range(batch_size):
                feature = []
                for j in range(self.num_agents):
                    observation = []
                    for feature_name in ['map_inlanes', 'adjacency_matrix']:
                        if 'adjacency' in feature_name:
                            continue
                        elif feature_name == "map_inlanes":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                total_features.append(feature)
            # feature:[agents,feature]
            total_features = np.reshape(np.array(total_features), [batch_size, self.num_agents, -1])
            # adj:[agent,neighbors]
        if bar:
            action = self.target_model.predict(total_features)
        else:
            action = self.model.predict(total_features)

        # out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        return total_features, action

    def adjacency_index2matrix(self, adjacency_index):
        # adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """
        # [batch,agents,neighbors]
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        l = to_categorical(adjacency_index_new, num_classes=self.num_agents)  # 转化为独热码
        return l

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def save_network(self, file_name):
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)
        self.model.save(os.path.join(self.path_to_model, "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)
        self.target_model.save(os.path.join(self.path_to_model, "%s.h5" % file_name))

