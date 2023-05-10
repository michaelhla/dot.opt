import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal

# Tests the restriction that accounts cannot be created twice


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

create_account = test_client.process("Create Account", client_logged_in, username="yush")
server.send(create_account)
response, client_logged_in = test_client.receive(server)

if client_logged_in != True:
    print('TEST003 FAILED: creation error')
    exit()

logout_account = test_client.process("Logout", client_logged_in)
server.send(logout_account)
response, client_logged_in = test_client.receive(server)

if client_logged_in != False:
    print('TEST003 FAILED: logout error')
    exit()

create_account = test_client.process("Create Account", client_logged_in, username="yush")
server.send(create_account)
response, client_logged_in = test_client.receive(server)

if client_logged_in != False or response != "The account yush already exists. Please try again.":
    print('TEST003 FAILED: double creation failure')
    exit()

print('TEST003 PASSED')


login_account = test_client.process("Login", client_logged_in, username='yush')
server.send(login_account)
response, client_logged_in = test_client.receive(server)


delete_account = test_client.process("Delete Account", client_logged_in)
server.send(delete_account)

exit()
