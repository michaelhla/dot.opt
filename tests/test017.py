import time
import socket
import traceback
import sys
from _thread import *
from threading import Lock
import re
import json
import os
import json
import select
import threading

# edge case , invalid flag

user_state_dictionary = {}
client_dictionary = {}
message_queue = {}
msg_db = {}

ADDR_1 = "127.0.0.1"
ADDR_2 = "127.0.0.1"
ADDR_3 = "127.0.0.1"


PORT_1 = 9080
PORT_2 = 9081
PORT_3 = 9082

CPORT_1 = 8080
CPORT_2 = 8081
CPORT_3 = 8082

ADDRS = [ADDR_1, ADDR_2, ADDR_3]
PORTS = [PORT_1, PORT_2, PORT_3]
CPORTS = [CPORT_1, CPORT_2, CPORT_3]

dict_lock = Lock()
replica_lock = Lock()

# replica connections, that are established, changed to the connection once connected
replica_connections = {"1": 0, "2": 0, "3": 0}

local_to_load = [user_state_dictionary, msg_db, message_queue]

is_Primary = False
USERFILEPATH = "user" + 'machine_idx' + ".json"
MSGFILEPATH = "sent" + 'machine_idx' + ".json"
MSGQPATH = "msg_queue" + 'machine_idx' + ".json"

files_to_expect = [USERFILEPATH, MSGFILEPATH, MSGQPATH]
user_state_dictionary = {'mike': 0}


def write(flag):
    if flag == 0:
        dict = user_state_dictionary
    elif flag == 1:
        dict = msg_db
    elif flag == 2:
        dict = message_queue
    else:
        print('TEST017 PASSED')
        exit()
    print('TEST017 FAILED')
    exit()
    file = files_to_expect[flag]
    try:
        print(dict, file)
        # with open(file, 'w') as f:
        #     json.dump(dict, f)
    except:
        print(f'ERROR: could not write to {file}')
        print('TEST016 FAILED')


write(3)
