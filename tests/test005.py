import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal

# Tests List Accounts Function

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

usernames = ["yush", "aayush", "aauh", "blue"]
for username in usernames:
    create_account = test_client.process("Create Account", client_logged_in, username=username)
    server.send(create_account)
    response, client_logged_in = test_client.receive(server)


    logout_account = test_client.process("Logout", client_logged_in)
    server.send(logout_account)
    response, client_logged_in = test_client.receive(server)


find_account = test_client.process("List Accounts", client_logged_in, query="*u*")
server.send(find_account)
#ancilla to capture nonsignificant state change return for listing accounts
response, ancilla = test_client.receive(server)

if response.split() != "Users matching *u*:\n yush aayush aauh blue".split():
    print('TEST005 FAILED: list error')

else:
    print('TEST005 PASSED')

# cleanup server

for username in usernames:
    login_account = test_client.process("Login", client_logged_in, username=username)
    server.send(login_account)
    response, client_logged_in = test_client.receive(server)

    delete_account = test_client.process("Delete Account", client_logged_in)
    server.send(delete_account)
    response, client_logged_in = test_client.receive(server)

exit()












