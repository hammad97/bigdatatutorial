#!/usr/bin/python3
import sys
from collections import Counter 
from ast import literal_eval


arrDelay={}
apData = {}
for map_input in sys.stdin:
    map_input = literal_eval(map_input) 
    try:
        map_input = (map_input[0], float(map_input[1]), float(map_input[2]))
    except Exception as e:
        map_input=None
    if map_input:
        if map_input[0] not in apData.keys():
            apData[map_input[0]] = (map_input[0], float(map_input[1]), float(map_input[2]), 0 , 1, 0)
        else:
            temp = apData[map_input[0]]
            minn = temp[1]
            maxx = temp[2]
            if map_input[1] < temp[1]:
                minn = map_input[1]
            if map_input[1] > temp[2]:
                maxx = map_input[1]
            apData[map_input[0]]=(temp[0], minn, maxx, temp[3] + map_input[1], temp[4] + 1, temp[5] + map_input[2])

print ("Departure Delays :")
for key, value in apData.items():
    print("Airport : ", value[0] ,", Minimum : ", value[1], ", Maximum : ", value[2], ", Average : ", (value[3] / value[4]))
    arrDelay[key] = value[5] / value[4]
    
print("Arrival Delays :")
countArr = Counter(arrDelay)
for key, value in countArr.most_common(10):
    print("Airport :", key, ", Average : ", value)
