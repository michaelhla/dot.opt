import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal
import time
from threading import Lock


# Checking to ensure there are no race conditions on client dictionary; tested for multiple create accounts

NUM_THREADS = 10
markers = []

marker_lock = Lock()


def thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if len(sys.argv) != 3:
        print("Correct usage: script, IP address, port number")
        exit()
    IP_address = str(sys.argv[1])
    Port = int(sys.argv[2])

    server.connect((IP_address, Port))

    client_logged_in = False

    create_account = test_client.process("Create Account", client_logged_in, username="yush")
    server.send(create_account)
    response, client_logged_in = test_client.receive(server)

    if response == "The account yush already exists. Please try again.":
        marker_lock.acquire()
        markers.append(1)
        marker_lock.release()
    else:
        marker_lock.acquire()
        markers.append(0)
        marker_lock.release()
        

thread_list = []

for i in range(0, NUM_THREADS):
    thread_list.append(threading.Thread(target=thread(), daemon=True))

for thread in thread_list:
    thread.start()

for thread in thread_list:
    thread.join()


if sum(markers) != NUM_THREADS-1:
    print('TEST010 FAILED: race condition')
else:
    print('TEST010 PASSED')


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if len(sys.argv) != 3:
    print("Correct usage: script, IP address, port number")
    exit()
IP_address = str(sys.argv[1])
Port = int(sys.argv[2])

server.connect((IP_address, Port))

client_logged_in = False

login_account = test_client.process("Login", client_logged_in, username='yush')
server.send(login_account)
response, client_logged_in = test_client.receive(server)


delete_account = test_client.process("Delete Account", client_logged_in)
server.send(delete_account)
response, client_logged_in = test_client.receive(server)

exit()