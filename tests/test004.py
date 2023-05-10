import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal

# Test ensuring cannot login into nonexisting account

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if len(sys.argv) != 3:
    print("Correct usage: script, IP address, port number")
    exit()
IP_address = str(sys.argv[1])
Port = int(sys.argv[2])

# process = subprocess.Popen(["python3", "test_server.py"] + [IP_address, str(Port)])


server.connect((IP_address, Port))

global client_logged_in
client_logged_in = False

login_account = test_client.process("Login", client_logged_in, username='yush')
server.send(login_account)
response, client_logged_in = test_client.receive(server)

if client_logged_in != False or response != "Username not found. Please try again.":
    print('TEST004 FAILED: Login into nonexisting username.')
    exit()
else:
    print('TEST004 PASSED')
    exit
    