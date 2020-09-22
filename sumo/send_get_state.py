import requests
import json
import os
import re
import numpy as np

#state_api_key = 'd3z94YsdZbFmo1So4dOscjDHb2w='
#state_device_ID = '603595007'
#action_api_key = 'VknSLNCjwhPXJg13IPRxxaaunzU='
#action_device_ID = '603264724'
action_api_key = '2jHSIjomFCS=28ZDPlD0GO06c=o='
action_device_ID = '605460850'
state_api_key = '2jHSIjomFCS=28ZDPlD0GO06c=o='
state_device_ID = '605460941'

class Car():
    veh_id = 0
    veh_type= 0
    veh_lane= 0
    lane_index= 0
    waiting_time= 0
    lane_pos= 0
    def __init__(self,veh_id,veh_type,veh_lane,lane_index,waiting_time,lane_pos):
        self.veh_id = veh_id
        self.veh_type = veh_type
        self.veh_lane = veh_lane
        self.lane_index = lane_index
        self.waiting_time = waiting_time
        self.lane_pos = lane_pos
    
    def __repr__(self):
        return((self.veh_id,self.veh_type, self.veh_lane,self.lane_index,self.waiting_time,self.lane_pos))

    def get_info(self):
        return str("{'veh_id':"+str(self.veh_id)+",veh_type':"+str(self.veh_type)+",'veh_lane':"+str(self.veh_lane)+",'lane_index':"+str(self.lane_index)+",'waiting_time':"+str(self.waiting_time)+",'lane_pos':"+str(self.waiting_time)+"}")

def generate_car(Car_Info):
    length = len(Car_Info[0])
    cars = list()
    fire_truck = list()
    for i in range(length):
        car = Car(Car_Info[0][i],Car_Info[1][i],Car_Info[2][i],Car_Info[3][i],Car_Info[4][i],Car_Info[5][i])
        if(car.veh_type == "Car"):
            cars.append(car)
        else:
            fire_truck.append(car)
    cars.sort(key=lambda car: car.waiting_time,reverse=True)
    fire_truck.sort(key=lambda car: car.waiting_time,reverse=True)
    cars = cars[:6]
    cars = json.dumps(cars, default=lambda o: o.__dict__)
    fire_truck = json.dumps(fire_truck, default=lambda o: o.__dict__)
    return cars,fire_truck


# 上传汽车信息
def Send_CarInfo(Car_Info, api_key, device_ID):
    # type: (object, object, object) -> object
    url_post = "https://api.heclouds.com/devices/" + device_ID + "/datapoints"  # 数据点
    headers = {'api-key': api_key}
    # veh_id  veh_Type  veh_lane  lane_index  waiting_time  lane_pos
    cars,fire_truck = generate_car(Car_Info)
    data = {'datastreams': [
        {"id": "veh_id", "datapoints": [{"value": Car_Info[0]}]},
        {"id": "veh_Type", "datapoints": [{"value": Car_Info[1]}]},
        {"id": "veh_lane", "datapoints": [{"value": Car_Info[2]}]},
        {"id": "lane_index", "datapoints": [{"value": Car_Info[3]}]},
        {"id": "waiting_time", "datapoints": [{"value": Car_Info[4]}]},
        {"id": "lane_pos", "datapoints": [{"value": Car_Info[5]}]},
        {"id": "total","datapoints":[{"value": cars}]},
        {"id": "fire_truck","datapoints":[{"value": fire_truck}]},
        {"id": "num","datapoints": [{"value": len(Car_Info[0])}]}
    ]}
    jdata = json.dumps(data).encode("utf-8")
    r = requests.post(url=url_post, headers=headers, data=jdata)


# 上传性能数据
def Send_RecordInfo(record_Info, api_key, device_ID):
    url_post = "https://api.heclouds.com/devices/" + device_ID + "/datapoints"  # 数据点
    headers = {'api-key': api_key}
    data = {'datastreams': [
        {"id": "record_Info", "datapoints": [{"value": record_Info}]}
    ]}
    jdata = json.dumps(data).encode("utf-8")
    r = requests.post(url=url_post, headers=headers, data=jdata)

# 上传数据流
def send_state(state, device_ID, api_key):
    url_post = "https://api.heclouds.com/devices/" + device_ID + "/datapoints"  # 数据点
    headers = {'api-key': api_key}
    data = {'datastreams': [
        {"id": "state", "datapoints": [{"value": state}]}
    ]}
    jdata = json.dumps(data).encode("utf-8")
    r = requests.post(url=url_post, headers=headers, data=jdata)
    # params = json.loads(r.text)
    # print("上传：", params['error'])

# 接收数据流
def get_state(step, device_ID, api_key, copy_cloud_data):
    data_path = os.path.join('cloud_data/')
    url_get = "https://api.heclouds.com/devices/" + device_ID + "/datastreams"  # 数据流
    headers = {'api-key': api_key}
    r = requests.get(url=url_get, headers=headers)
    t = r.text
    params = json.loads(t)
    x = params['data']

    datastream_id = []
    datastream = []
    for index, values in enumerate(x):
        datastream_id.append(str(values['id']))
        current_value = str(values.get('current_value', ''))
        datastream.append(current_value)

        if copy_cloud_data == 1:
            # 备份下载好的云端state文件
            folder = os.path.exists(data_path + str(values['id']))
            if not folder:
                os.makedirs(data_path + str(values['id']))
            data_path = os.path.join('cloud_data/' + str(values['id'] + '/'))
            fname = str(values['id']) + str(step) + '.txt'
            f = open(data_path + fname, 'w')
            f.write(current_value)
            f.close()

    if len(datastream)==0:
        return datastream_id, datastream

    state = []
    pattern = re.compile(r'[\[](.*?)[\]]', re.S)
    result = pattern.findall(current_value)
    for i in range(len(result)):
        result[i] = result[i].replace('[', '')
        result[i] = result[i].split(',')
        if i < 9:
            for j in range(len(result[i])):
                result[i][j] = int(result[i][j])
        else:
            for j in range(len(result[i])):
                result[i][j] = float(result[i][j])
    for i in range(9):
        dd = dict()
        dd['map_inlanes'] = np.array(result[i + 9])
        dd['adjacency_matrix'] = result[i]
        state.append(dd)

    return datastream_id, state

# 删除数据流
def delete_state(datastream_id, device_ID, api_key):
    url_get = "https://api.heclouds.com/devices/" + device_ID + "/datastreams"  # 数据流
    headers = {'api-key': api_key}
    for dsid in datastream_id:
        url_del = url_get + '/' + dsid
        r = requests.delete(url=url_del, headers=headers)
        params = json.loads(r.text)

def main():
    str = '[0 0 3 0 2 1 1 3 2]'
    pattern = re.compile(r'\d{1}', re.S)
    result = pattern.findall(str)
    for i in range(len(result)):
        result[i] = int(result[i])
    result = np.array(result)
    print(result)
    datastream_id, state = get_state(0, state_device_ID, state_api_key, 0)  # 控制端从onenet获取state
    print(state)

if __name__ == '__main__':
    main()