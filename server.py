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

NUM_MACHINES = 3

ADDR_1 = "localhost"
ADDR_2 = "localhost"
ADDR_3 = "localhost"


PORT_1 = 9080
PORT_2 = 9081
PORT_3 = 9082

CPORT_1 = 8080
CPORT_2 = 8081
CPORT_3 = 8082


ADDRS = [ADDR_1, ADDR_2, ADDR_3]
PORTS = [PORT_1, PORT_2, PORT_3]
CPORTS = [CPORT_1, CPORT_2, CPORT_3]

# Machine number
machine_idx = str(sys.argv[1])

# IP address
IP = ADDRS[int(machine_idx)-1]

# Server Port number
s_port = PORTS[int(machine_idx)-1]

# Client Port number
c_port = CPORTS[int(machine_idx)-1]

# client username dictionary, with login status: 0 if logged off, corresponding address if logged in
queue = {}
products = {}

# replica dictionary, keyed by address and valued at machine id
replica_dictionary = {"1": (ADDR_1, PORT_1), "2": (
    ADDR_2, PORT_2), "3": (ADDR_3, PORT_3)}
reverse_rep_dict = {(ADDR_1, PORT_1): "1", (ADDR_2, PORT_2)                    : "2", (ADDR_3, PORT_3): "3"}

# replica connections, that are established, changed to the connection once connected
replica_connections = {"1": 0, "2": 0, "3": 0}

# lock on replica_connections, because incoming servers could affect accessing during leader election
replica_lock = Lock()

# defined global variable of whether replica is primary or backup
is_Primary = False

# lock for the message queue
dict_lock = Lock()


# DB OPERATIONS

dbfolder = "dbfolder/"

QPATH = dbfolder + "user" + machine_idx + ".json"
PRODPATH = dbfolder + "sent" + machine_idx + ".json"

files_to_expect = [QPATH, PRODPATH]
local_to_load = [queue, products]


def load_db_to_state(path):
    try:
        with open(path, 'r') as f:
            res_dictionary = json.load(f)
    except Exception as e:
        res_dictionary = {}
        print(e)

    return res_dictionary

# writes are correlated with a flag for each db to write to, pass in db ID to write to file


def write(flag):
    if flag == 0:
        dict = queue
    elif flag == 1:
        dict = products
    file = files_to_expect[flag]
    try:
        print(dict, file)
        with open(file, 'w') as f:
            json.dump(dict, f)
    except:
        print(f'ERROR: could not write to {file}')


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
                print(e)
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
        size = prim_conn.recv(4)
        dim1 = prim_conn.recv(4)
        dim2 = prim_conn.recv(4)
        msg = prim_conn.recv(size)
        if msg:
            m1 = np.frombuffer(
                msg[:size/2], dtype=np.uint8).reshape((dim1, dim2))
            m2 = np.frombuffer(
                msg[size/2:], dtype=np.uint8).reshape((dim1, dim2))
            # handles message sent by primary
            strassen(m1, m2)
        else:
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
                    print(e)
                    replica_lock.acquire()
                    if replica_connections[str(i)] != 0:
                        replica_connections[str(i)].close()
                    replica_connections[str(i)] = 0
                    replica_lock.release()
                    continue
            # if after connecting, backup is lowest running, elected as primary
            if is_Lowest == True:
                is_Primary = True
            print("election done")
            print(is_Primary)


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
            print(index_of_connector, 'has connected as backup')
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
            index_of_connector = conn_type[0]
            key = str(index_of_connector)
            # if other replica is connecting:
            if key in replica_dictionary.keys():
                replica_lock.acquire()
                replica_connections[key] = conn
                replica_lock.release()
                # sends tag that this connection is the primary
                bmsg = (1).to_bytes(1, "big")
                conn.sendall(bmsg)

                # sends logs of client dict, sent messages, and message queue, for catchup
                for i in range(len(files_to_expect)):
                    file = files_to_expect[i]
                    filesize = os.path.getsize(file)
                    id = (i).to_bytes(4, "big")
                    size = (filesize).to_bytes(8, "big")
                    conn.sendall(id)
                    conn.sendall(size)
                    try:
                        with open(file, 'rb') as sendafile:
                            # Send the file over the connection in chunks
                            bytesread = sendafile.read(1024)
                            if not bytesread:
                                break
                            conn.sendall(bytesread)
                    except:
                        print('file error')


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

# HANDLES CLIENT SIDE, binds server to client facing address/ports
clientserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientserver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
clientserver.bind((IP, c_port))
clientserver.listen()
inputs.append(clientserver)
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
                # knows it is backup, catches up on server state
                try:
                    for i in range(len(files_to_expect)):
                        id = conn_socket.recv(4)
                        id = int.from_bytes(id, byteorder='big')
                        file_size = conn_socket.recv(8)
                        file_size = int.from_bytes(file_size, byteorder='big')
                        byteswritten = 0
                        with open(f'{files_to_expect[id]}', 'wb') as f:
                            # receive the file contents
                            while byteswritten < file_size:
                                buf = min(file_size - byteswritten, 1024)
                                data = conn_socket.recv(buf)
                                f.write(data)
                                byteswritten += len(data)
                        if local_to_load is not None and byteswritten != 0:
                            if i == 0:
                                queue = load_db_to_state(
                                    files_to_expect[i])
                                # persistence for the primary
                            elif i == 1:
                                products = load_db_to_state(
                                    files_to_expect[i])

                except Exception as e:
                    print('init error', e)
                    traceback.print_exc()

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

print('is primary:', is_Primary)

# list of threads that are always concurrent
thread_list = []
(threading.Thread(target=backup_message_handling)).start()
(threading.Thread(target=server_interactions)).start()


# todo: leader thread for job coordination
# coordinated job scheduling and sending
