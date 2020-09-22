import requests
import json
import os


def send_action(action, device_ID, api_key):
    # send_action_dis(action)
    url_post = "https://api.heclouds.com/devices/" + device_ID + "/datapoints"  # 数据点
    headers = {'api-key': api_key}
    data = {'datastreams': [
        {"id": "action", "datapoints": [{"value": action}]}
    ]}
    jdata = json.dumps(data).encode("utf-8")
    r = requests.post(url=url_post, headers=headers, data=jdata)
    # params = json.loads(r.text)



# 接收数据流
def get_action(step, device_ID, api_key, copy_cloud_data):
    data_path = os.path.join('cloud_data/')
    url_get = "https://api.heclouds.com/devices/" + device_ID + "/datastreams"  # 数据流
    headers = {'api-key': api_key}
    r = requests.get(url=url_get, headers=headers)
    t = r.text
    params = json.loads(t)
    # print(params)
    x = params['data']

    datastream_id = []
    datastream = []
    for index, values in enumerate(x):
        datastream_id.append(str(values['id']))
        current_value = str(values.get('current_value', ''))
        datastream.append(current_value)

        if copy_cloud_data == 1:
            # 备份下载好的云端action文件
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