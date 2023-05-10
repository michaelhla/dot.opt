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

# testing if response from primary is 1, server goes into backup behavior

# replica connections, that are established, changed to the connection once connected
replica_connections = {"1": 0, "2": 0, "3": 0}

local_to_load = [user_state_dictionary, msg_db, message_queue]

is_Primary = False


def backup_message_handling(machine_idx, res):
    global is_Primary
    global prim_conn
    while is_Primary == False:
        msg = None  # prim_conn.recv(2048)
        if msg:
            pass
        else:
            # handle leader election
            # if this doesnt work, use test sockets that are close
            is_Lowest = True
            for i in range(1, int(machine_idx)):
                try:
                    # ensures ordering of leader election, replacement for conn.active()
                    # time.sleep((int(machine_idx)-2)*0.2)
                    # test_socket = socket.socket(
                    #     socket.AF_INET, socket.SOCK_STREAM)
                    # test_socket.connect((ADDRS[i-1], PORTS[i-1]))
                    # # test_socket.settimeout(int(machine_idx))
                    # test_socket.sendall(int(machine_idx).to_bytes(1, "big"))

                    # test_socket.settimeout(None)
                    # replica_lock.acquire()
                    # if replica_connections[str(i)] != 0:
                    #     replica_connections[str(i)].close()
                    # replica_connections[str(i)] = test_socket
                    # replica_lock.release()

                    ret_tag = res
                    print(ret_tag)
                    if ret_tag == 1:
                        is_Lowest = False
                        prim_conn = replica_connections[str(i)]

                except ConnectionRefusedError:
                    replica_lock.acquire()
                    if replica_connections[str(i)] != 0:
                        replica_connections[str(i)].close()
                    replica_connections[str(i)] = 0
                    replica_lock.release()
                    continue
                except Exception as e:
                    print("B")
                    print(i)
                    print("-  -  -  -")
                    print(e)
                    replica_lock.acquire()
                    if replica_connections[str(i)] != 0:
                        replica_connections[str(i)].close()
                    replica_connections[str(i)] = 0
                    replica_lock.release()
                    continue
            if is_Lowest == True:
                is_Primary = True
            print("election done")
            print(is_Primary)
            break


backup_message_handling(2, 1)
if(not is_Primary):
    print('TEST013 PASSED')
else:
    print('TEST013 FAILED')
