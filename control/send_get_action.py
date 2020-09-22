import requests
import json
import os

def send_action(action, device_ID, api_key):
    send_action_dis(action)
    url_post = "https://api.heclouds.com/devices/" + device_ID + "/datapoints"  # 数据点
    headers = {'api-key': api_key}
    data = {'datastreams': [
        {"id": "action", "datapoints": [{"value": action}]}
    ]}
    jdata = json.dumps(data).encode("utf-8")
    r = requests.post(url=url_post, headers=headers, data=jdata)
    params = json.loads(r.text)
    # print('send action:', params['error'])
def send_action_dis(action):
    url_post = "https://api.heclouds.com/devices/610862356/datapoints"  # 数据点
    headers = {'api-key': '80hx6dOnn=DhnSoKFUmU3DaoECA='}
    data = {'datastreams': [
        {"id": "action", "datapoints": [{"value": action}]}
    ]}
    jdata = json.dumps(data).encode("utf-8")
    r = requests.post(url=url_post, headers=headers, data=jdata)
    # params = json.loads(r.text)
# 接收数据流
def get_action(step, device_ID, api_key):
    data_path = os.path.join('cloud_data/')
    url_get = "https://api.heclouds.com/devices/" + device_ID + "/datastreams"  # 数据流
    headers = {'api-key': api_key}
    r = requests.get(url=url_get, headers=headers)
    t = r.text
    params = json.loads(t)
    # print('get action:', params['error'])
    x = params['data']

    datastream_id = []
    datastream = []
    for index, values in enumerate(x):
        datastream_id.append(str(values['id']))
        current_value = str(values.get('current_value', ''))
        datastream.append(current_value)
    if len(datastream)==0:
        return datastream_id, datastream

    return datastream_id, datastream[0]

# 删除数据流
def delete_action(datastream_id, device_ID, api_key):
    url_get = "https://api.heclouds.com/devices/" + device_ID + "/datastreams"  # 数据流
    headers = {'api-key': api_key}
    for dsid in datastream_id:
        url_del = url_get + '/' + dsid
        r = requests.delete(url=url_del, headers=headers)
        params = json.loads(r.text)
        # print(params)

def main():
    action_api_key = 'VknSLNCjwhPXJg13IPRxxaaunzU='
    action_device_ID = '603264724'
    action = '2'
    send_action(action, action_device_ID, action_api_key)
    datastream_id, datastream = get_action(1, action_device_ID, action_api_key)
    print(datastream_id, datastream)
    delete_action(datastream_id, action_device_ID, action_api_key)

if __name__ == '__main__':
    main()