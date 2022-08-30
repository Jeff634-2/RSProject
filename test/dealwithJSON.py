import json
import os

new_JsonFile = open('./comment.json', 'w', encoding='utf-8')

with open('./comment.json', 'r', encoding='utf-8')as fp:
    json_data = json.load(fp)
    id_num = 1
    for i in range(0, len(json_data)):
        # 添加index行
        new_data = {}
        new_data['index'] = {}
        new_data['index']['_index'] = "index_1"
        new_data['index']['_id'] = str(id_num)
        id_num = id_num + 1
        temp = json.dumps(new_data).encode("utf-8").decode('utf-8')
        new_JsonFile.write(temp)
        new_JsonFile.write('\n')
        # 原json对象处理为1行
        old_data = {}
        old_data["scenicName"] = json_data[i]["scenicName"]
        old_data["city"] = json_data[i]["city"]
        old_data["theme"] = json_data[i]["theme"]
        old_data["address"] = json_data[i]["address"]
        old_data["price"] = json_data[i]["price"]
        old_data["scenicAddress"] = json_data[i]["scenicAddress"]
        old_data["businessHours"] = json_data[i]["businessHours"]
        old_data["scenicCharacteristic"] = json_data[i]["scenicCharacteristic"]
        old_data["scenicScore"] = json_data[i]["scenicScore"]
        old_data["imageUrl"] = json_data[i]["imageUrl"]
        old_data[" latitudeeAndLongitude"] = json_data[i][" latitudeeAndLongitude"]
        temp = json.dumps(old_data).encode("utf-8").decode('unicode_escape')
        new_JsonFile.write(temp)
        new_JsonFile.write('\n')

    new_JsonFile.close()