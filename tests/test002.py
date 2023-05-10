import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal

# Tests basic functionality of being able to Create Account, List Accounts, Logout, Login, or Delete Accounts


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if len(sys.argv) != 3:
    print("Correct usage: script, IP address, port number")
    exit()
IP_address = str(sys.argv[1])
Port = int(sys.argv[2])

# process = subprocess.Popen(["python3", "test_server.py"] + [IP_address, str(Port)])


server.connect((IP_address, Port))

# os.kill(process.pid, signal.SIGINT)
# process.wait()

# exit()

global client_logged_in
client_logged_in = False

create_account = test_client.process("Create Account", client_logged_in, username="yush")
server.send(create_account)
response, client_logged_in = test_client.receive(server)
find_account = test_client.process("List Accounts", client_logged_in, query="yush")
server.send(find_account)
#ancilla to capture nonsignificant state change return for listing accounts
response, ancilla = test_client.receive(server)

if client_logged_in != True or response.split() != "Users matching yush:\nyush".split():
    print('TEST002 FAILED: creation error')
    exit()



logout_account = test_client.process("Logout", client_logged_in)
server.send(logout_account)
response, client_logged_in = test_client.receive(server)

if client_logged_in != False:

    print('TEST002 FAILED: logout error')
    exit()



login_account = test_client.process("Login", client_logged_in, username='yush')
server.send(login_account)
response, client_logged_in = test_client.receive(server)

if client_logged_in != True:
    print('TEST002 FAILED: login error')
    exit()

delete_account = test_client.process("Delete Account", client_logged_in)
server.send(delete_account)
response, client_logged_in = test_client.receive(server)


if client_logged_in != False:
    print('TEST002 FAILED: deletion error')
    print(response.split())
    exit()


print('TEST002 PASSED')



exit()

