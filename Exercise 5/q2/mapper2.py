#!/usr/bin/python3
import sys

col_headers = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN" ,"DEST" ,"DEP_TIME" ,"DEP_DELAY" ,"ARR_TIME" ,"ARR_DELAY"]

for row in sys.stdin:
    origin = row.split(',')[3]
    depDelay = row.split(',')[6]
    arrDelay = row.split(',')[8]
    if depDelay == ' ' or not depDelay:
        depDelay = 0
    if arrDelay == ' ' or not arrDelay:
        arrDelay = 0
    print((origin, depDelay, arrDelay))       

