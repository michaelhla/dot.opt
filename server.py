import struct
import threading
import os
import json
from threading import Lock
from _thread import *
import sys
import traceback
import socket
import time
import numpy as np
import pickle

NUM_MACHINES = 8

ADDR_1 = "172.31.22.5"
ADDR_2 = "172.31.20.103"
ADDR_3 = "172.31.16.178"


ADDRS = [ADDR_1, ADDR_2, ADDR_3, '172.31.23.163', '172.31.24.221',
         '172.31.18.185', '172.31.30.125', '172.31.26.203']
PUB_ADDRS = ['18.222.0.123', '3.138.193.5', '18.220.90.168']
PORTS = [9080+i for i in range(0, NUM_MACHINES)]

start, end = 0, 0


DIMENSION = 800
matrix1 = np.ones((DIMENSION, DIMENSION))
matrix2 = np.ones((DIMENSION, DIMENSION))
matshape = matrix1.shape


global product
product = np.zeros(matshape)
task_log = {}

# Product lock
prod_lock = Lock()


# Machine number
machine_idx = str(sys.argv[1])

# IP address
IP = ADDRS[int(machine_idx)-1]

# Server Port number
s_port = PORTS[int(machine_idx)-1]

# client username dictionary, with login status: 0 if logged off, corresponding address if logged in
queue = {}
products = {}

log_lock = Lock()


# replica dictionary, keyed by address and valued at machine id
replica_dictionary = {}
reverse_rep_dict = {}
for i in range(1, NUM_MACHINES+1):
    replica_dictionary[str(i)] = (ADDRS[i-1], PORTS[i-1])
    reverse_rep_dict[(ADDRS[i-1], PORTS[i-1])] = str(i)

# replica connections, that are established, changed to the connection once connected
replica_connections = {}
for i in range(1, NUM_MACHINES+1):
    replica_connections[str(i)] = 0

# lock on replica_connections, because incoming servers could affect accessing during leader election
replica_lock = Lock()

# defined global variable of whether replica is primary or backup
is_Primary = False

# lock for the message queue
dict_lock = Lock()


# Machine Availability Dictionary
availability = {}
for i in range(1, NUM_MACHINES+1):
    availability[str(i)] = 0


# Checks to see what subtasks are remaining
subtask_queue = []
for i in range(1, DIMENSION+1):
    subtask_queue.append(str(i))

# locks for changes to subtask_queue and availability dict
avail_lock = Lock()
subtask_lock = Lock()

# completed subtasks
completed = []


# DB OPERATIONS

dbfolder = "dbfolder/"

QPATH = dbfolder + "queue" + machine_idx + ".json"
PRODPATH = dbfolder + "prod" + machine_idx + ".json"

files_to_expect = [QPATH, PRODPATH]
local_to_load = [queue, products]


def load_db_to_state(path):
    try:
        with open(path, 'r') as f:
            res_dictionary = json.load(f)
    except Exception as e:
        res_dictionary = {}
        # print(e)

    return res_dictionary

# writes are correlated with a flag for each db to write to, pass in db ID to write to file


def write(flag):
    if flag == 0:
        dict = queue
    elif flag == 1:
        dict = products
    file = files_to_expect[flag]
    try:
        # print(dict, file)
        with open(file, 'w') as f:
            json.dump(dict, f)
    except:
        # print(f'ERROR: could not write to {file}')
        pass


def outer_product(a, b):
    return np.outer(a, b)

# for backup servers, updates server state as if it were interacting with the client
# server state only involves creation/deletion of account, plus additions/removals to message queue
# for each tag, we update the local state and then persist it to the db


def split(matrix):
    """
    Splits a given matrix into quarters.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
    """
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]


def strassen(x, y):
    """
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices x and y
    Output: nxn matrix, product of x and y
    """

    osize = x.shape[0]

    # Base case when size of matrices is 1x1
    if len(x) == 1:
        return x * y

    if x.shape[0] % 2 != 0 or y.shape[0] % 2 != 0:
        rowadd = [np.zeros(x.shape[0], dtype=int)]
        coladd = np.zeros(x.shape[0]+1, dtype=int)
        x = np.append(x, rowadd, axis=0)
        y = np.append(y, rowadd, axis=0)
        x = np.hstack((x, np.atleast_2d(coladd).T))
        y = np.hstack((y, np.atleast_2d(coladd).T))

    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)

    # Computing the 7 products, recursively (p1, p2...p7)
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11[:osize][:osize], c12[:osize][:osize])), np.hstack(
        (c21[:osize][:osize], c22[:osize][:osize]))))

    return c


# function that given the connections to other replicas, a machine sends a message to all replicas that are active and not itself
def send_to_replicas(message):
    for idx in replica_connections.keys():
        if idx != machine_idx and replica_connections[idx] != 0:
            try:
                replica_connections[idx].sendall(message)
            except Exception as e:
                # print(e)
                continue


# listening thread for backup servers


def backup_message_handling():
    # global state variable dictating whether the server is in primary or backup state
    global is_Primary
    # global varibale storing connection to the primary
    global prim_conn
    # is_Primary = False maintains a listening thread to the primary connection
    while is_Primary == False:
        # may change dep on wire protocol

        broken_conn = False

        met_dat = prim_conn.recv(8)

        if not met_dat:
            broken_conn = True
        else:

            task_num = met_dat[0:4]

            dim = met_dat[4:]
            dimension = int.from_bytes(dim, byteorder='big')

            size = dimension*8

            data = b''

            while len(data) < size+size:
                chunk = prim_conn.recv(2048)
                if not chunk:
                    broken_conn = True
                    break
                data += chunk

        if broken_conn == False:
            m1 = np.frombuffer(
                data[:size], dtype=np.float64)

            m2 = np.frombuffer(
                data[size:], dtype=np.float64)

            oshape = m1.shape[0]
            task_log[str(task_num)] = [m1, m2]

            # handles message sent by primary
            result = np.outer(m1, m2)
            # take the first n rows and columns of the result
            # result = result[:oshape, :oshape]

            print('finished', task_num)
            task_log[str(task_num)] = result

            result_bmsg = result.tobytes()

            header1 = len(result_bmsg).to_bytes(4, "big")

            # What if connection breaks here?: To Do

            prim_conn.sendall(header1)
            prim_conn.sendall(result_bmsg)
            # DELETE LATER
            # return
            # prim_conn.close()
            # backupserver.close()
            # sys.exit(0)

        else:
            # print("uh oh")
            # empty message means primary connection broken
            # save current backup state
            for i in range(len(files_to_expect)):
                write(i)

            # handle leader election
            # if this doesnt work, use test sockets that are close

            # variable marking if current replica is the lowest idx still running
            is_Lowest = True
            # try to connect to all machines with a lower index, as election is determined by lowest current running index
            for i in range(1, int(machine_idx)):
                try:
                    # ensures ordering of leader election, replacement for conn.active()
                    time.sleep((int(machine_idx)-2)*0.2)
                    # test connection; note that existing connections block, so need to replace existing connection to test if connection is acceptable
                    test_socket = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM)
                    # if this fails, goes to ConnectionRefusedError
                    test_socket.connect((ADDRS[i-1], PORTS[i-1]))
                    # test_socket.settimeout(int(machine_idx))
                    test_socket.sendall(int(machine_idx).to_bytes(1, "big"))
                    # convert task_log to bytes and send to test_socket with header of size and id
                    task_log_bmsg = pickle.dumps(task_log)
                    size = len(task_log_bmsg).to_bytes(4, "big")
                    bmsg = size + task_log_bmsg
                    test_socket.sendall(bmsg)

                    test_socket.settimeout(None)

                    # replaces any previous connection with the test_socket, for clarity
                    replica_lock.acquire()
                    if replica_connections[str(i)] != 0:
                        replica_connections[str(i)].close()
                    replica_connections[str(i)] = test_socket
                    replica_lock.release()

                    # ensures connection to primary, by reciving the return tag
                    ret_tag = test_socket.recv(1)[0]
                    if ret_tag == 1:
                        # there is a smaller machine index still runnning, so still backup
                        is_Lowest = False
                        prim_conn = replica_connections[str(i)]

                except ConnectionRefusedError:
                    # this means the connection to a lower index is down, so is_Lowest is still True
                    replica_lock.acquire()
                    if replica_connections[str(i)] != 0:
                        replica_connections[str(i)].close()
                    replica_connections[str(i)] = 0
                    replica_lock.release()
                    continue
                except Exception as e:
                    # print(e)
                    replica_lock.acquire()
                    if replica_connections[str(i)] != 0:
                        replica_connections[str(i)].close()
                    replica_connections[str(i)] = 0
                    replica_lock.release()
                    continue
            # if after connecting, backup is lowest running, elected as primary
            if is_Lowest == True:
                is_Primary = True
            # print("election done")
            # print(is_Primary)

            # TO DO: Reconstruction after leader fails


def catchup(conn):
    try:
        size = conn.recv(4)
        bytes_log = conn.recv(int.from_bytes(size, "big"))
        if bytes_log is not None:
            rec_log = pickle.loads(bytes_log)
            log_lock.acquire()
            for key in rec_log.keys():
                task_log[key] = rec_log[key]
            # print('new task log', task_log)
            log_lock.release()
            # print("caught up with", conn)

    except Exception as e:
        # print('ERROR with', conn, e)
        pass
# thread handling server interactions; all servers interact at backupserver addresses


def server_interactions():
    global is_Primary
    while True:
        # connections are always accepted
        conn, addr = backupserver.accept()
        # starts backup behavior according to global state
        if is_Primary == False:
            # backup behavior
            # tells other incoming connections that it is a backup replica, and it receives a machine index
            conn_type = conn.recv(1)
            index_of_connector = conn_type[0]
            print("------------------")
            print(index_of_connector, 'has connected as backup')
            print("------------------")
            key = str(index_of_connector)
            # is a connecting replica, so the machine index is sent:
            if key in replica_dictionary.keys():
                replica_lock.acquire()
                # incoming replica connection stored
                replica_connections[key] = conn
                replica_lock.release()
                bmsg = (0).to_bytes(1, "big")
                conn.sendall(bmsg)
        else:
            # primary behavior
            # still receives a connection, and gets an index of the connector
            conn_type = conn.recv(1)
            # catchup(conn)
            index_of_connector = conn_type[0]
            key = str(index_of_connector)
            # if other replica is connecting:
            if key in replica_dictionary.keys():
                replica_lock.acquire()
                replica_connections[key] = conn
                replica_lock.release()

                # marks machine index as available to compute
                avail_lock.acquire()
                availability[key] = 1
                avail_lock.release()
                # sends tag that this connection is the primary
                bmsg = (1).to_bytes(1, "big")
                conn.sendall(bmsg)
                print('acknowledge', key)

                ready_conns = [key for key in availability.keys()
                               if availability[key] == 1]
                if len(ready_conns) == NUM_MACHINES-1:
                    # if all nodes connected, then the primary can start sending tasks
                    global start
                    start = time.time()
                    for id in ready_conns:
                        start_new_thread(
                            task_scheduler, (replica_connections[id], addr, id))


def task_scheduler(conn, addr, key):
    global product
    worker_state = True
    while worker_state:
        subtask = None
        avail_lock.acquire()
        subtask_lock.acquire()
        if availability[key] == 1 and len(subtask_queue) > 0:
            # make unavailable
            availability[key] = 0
            avail_lock.release()
            subtask_lock.release()

            # pop subtask from queue
            subtask_lock.acquire()
            subtask = subtask_queue.pop(0)
            # print(subtask, subtask_queue)
            subtask_lock.release()

            # send subtask to computing machine

            send_task(subtask, conn, addr)

            # TO DO: how large?
            res_head = conn.recv(4)
            if not res_head:
                worker_state = False
            else:
                res_size = int.from_bytes(res_head, "big")
                data = b''

                while len(data) < res_size:
                    chunk = conn.recv(2048)
                    if not chunk:
                        worker_state = False
                        break
                    data += chunk

            if worker_state == True:
                # process result
                result = np.frombuffer(data, dtype=np.float64).reshape(
                    (int(DIMENSION), int(DIMENSION)))
                prod_lock.acquire()
                product += result
                prod_lock.release()

                completed.append(subtask)

                # make availability open again
                avail_lock.acquire()
                availability[key] = 1
                avail_lock.release()

            else:
                # connection broken, server no longer available
                avail_lock.acquire()
                availability[key] = 0
                avail_lock.release()

                # reappend task to queue to reassign
                subtask_lock.acquire()
                subtask_queue.append(subtask)
                subtask_lock.release()

                replica_connections[key] = 0
                worker_state = False
                # TO DO: remove connection?
        else:
            avail_lock.release()
            subtask_lock.release()
            if len(completed) == DIMENSION:
                print(product)
                end = time.time()

                print(str(end-start))
                with open('times.txt', 'a') as f:
                    # Write data to the file
                    f.write(str(end-start)+"\n")
                    f.close()
            break

# Sends the task (according to strassen's breakdown of matrix recursion)


def send_task(subtask, conn, addr):
    tag = (int(subtask)).to_bytes(4, "big")
    dim = (int(DIMENSION)).to_bytes(4, "big")
    column = matrix1[:, int(subtask)-1]
    row = matrix2[int(subtask)-1, :]

    column_bmsg = column.tobytes()
    row_bmsg = row.tobytes()

    # maybe change becau
    # header1 = len(column_bmsg).to_bytes(4, "big")
    # header2 = len(row_bmsg).to_bytes(4, "big")
    meta = tag+dim

    conn.sendall(meta)
    conn.sendall(column_bmsg+row_bmsg)


# FULL INITIALIZATION
# catch up on logs, and determine primary by connections
# init process:
global prim_conn
prim_conn = None

# binds server to server only address/ports
backupserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
backupserver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
backupserver.bind((IP, s_port))
backupserver.listen()


inputs = [backupserver]


# loads from persistent memory for all servers
for i in range(len(local_to_load)):
    if i == 0:
        queue = load_db_to_state(
            files_to_expect[i])  # persistence for the primary
    elif i == 1:
        products = load_db_to_state(
            files_to_expect[i])

# reaching out
# only while a server is_Primary=True can it accept connections
primary_exists = False
for idx in replica_dictionary.keys():
    # print(replica_dictionary, idx)
    if idx != machine_idx:
        try:
            conn_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          # print(replica_dictionary[idx])
            conn_socket.connect(replica_dictionary[idx])
            mtag = int(machine_idx).to_bytes(1, "big")
            conn_socket.sendall(mtag)

            # store connection in replica_connections
            replica_lock.acquire()
            # ONLY GETS HERE ONCE FOR SECOND SERVER
            replica_connections[idx] = conn_socket
            replica_lock.release()

            # received tag from other replicas, 0 implies backup, 1 implies primary
            tag = conn_socket.recv(1)
            if tag[0] == 1:
                primary_exists = True
                prim_conn = conn_socket
                # knows it is backup
            if tag == 0:
                # reached out to backup, so nothing to change here, other than replica connection
                pass

        except ConnectionRefusedError:
            pass
        except Exception as e:
            traceback.print_exc()


# if no primary exists, default primary
if primary_exists == False:
    is_Primary = True

# print('is primary:', is_Primary)

# list of threads that are always concurrent
thread_list = []
(threading.Thread(target=backup_message_handling)).start()
(threading.Thread(target=server_interactions)).start()


# # TO DO: Reconstruction after leader fails
# sudo yum install python-pip
# ssh -i "dot.pem" ec2-user@ec2-52-14-182-200.us-east-2.compute.amazonaws.com
# scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py  ec2-user@ec2-52-14-182-200.us-east-2.compute.amazonaws.com:~/.
#
# ssh -i "dot.pem" ec2-user@ec2-18-191-66-11.us-east-2.compute.amazonaws.com
# scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py  ec2-user@ec2-18-191-66-11.us-east-2.compute.amazonaws.com:~/.
#
# ssh -i "dot.pem" ec2-user@ec2-3-19-221-58.us-east-2.compute.amazonaws.com
# scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py ec2-user@ec2-3-19-221-58.us-east-2.compute.amazonaws.com:~/.
#
# ssh -i "dot.pem" ec2-user@ec2-3-21-39-6.us-east-2.compute.amazonaws.com
# scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py  ec2-user@ec2-3-21-39-6.us-east-2.compute.amazonaws.com:~/.
#
# ssh -i "dot.pem" ec2-user@ec2-3-142-141-230.us-east-2.compute.amazonaws.com
# scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py ec2-user@ec2-3-142-141-230.us-east-2.compute.amazonaws.com:~/.
#
# ssh -i "dot.pem" ec2-user@ec2-18-223-168-179.us-east-2.compute.amazonaws.com
# scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py ec2-user@ec2-18-223-168-179.us-east-2.compute.amazonaws.com:~/.

# ssh -i "dot.pem" ec2-user@ec2-3-17-12-45.us-east-2.compute.amazonaws.com
# # scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py ec2-user@ec2-3-17-12-45.us-east-2.compute.amazonaws.com:~/.
# python3 server.py 1
# ssh -i "dot.pem" ec2-user@ec2-18-217-90-20.us-east-2.compute.amazonaws.com
# # scp -i /Users/michaelh40/Code/dot.opt/dot.pem /Users/michaelh40/Code/dot.opt/server.py ec2-user@ec2-18-217-90-20.us-east-2.compute.amazonaws.com:~/.
