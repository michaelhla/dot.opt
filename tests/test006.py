import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal

# Tests Message Dump Function

TEXT = "Is the DOW up today?"

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


usernames = ["yush", "mhla"]
for username in usernames:
    create_account = test_client.process("Create Account", client_logged_in, username=username)
    server.send(create_account)
    response, client_logged_in = test_client.receive(server)


    logout_account = test_client.process("Logout", client_logged_in)
    server.send(logout_account)
    response, client_logged_in = test_client.receive(server)

login_account = test_client.process("Login", client_logged_in, username="yush")
server.send(login_account)
response, client_logged_in = test_client.receive(server)

send_message = test_client.process("Send", client_logged_in, username="yush", recipient="mhla", data=TEXT)
server.send(send_message)
response, client_logged_in = test_client.receive(server)

logout_account = test_client.process("Logout", client_logged_in)
server.send(logout_account)
response, client_logged_in = test_client.receive(server)

login_account = test_client.process("Login", client_logged_in, username="mhla")
server.send(login_account)
response, client_logged_in = test_client.receive(server)

dump_request = test_client.process("Open Undelivered Messages", client_logged_in)
server.send(dump_request)
response, client_logged_in = test_client.receive(server)

if response != "<yush>: " + TEXT:
    print('TEST006 FAILED: dump error')
else:
    print('TEST006 PASSED')


logout_account = test_client.process("Logout", client_logged_in)
server.send(logout_account)
response, client_logged_in = test_client.receive(server)

for username in usernames:
    login_account = test_client.process("Login", client_logged_in, username=username)
    server.send(login_account)
    response, client_logged_in = test_client.receive(server)

    delete_account = test_client.process("Delete Account", client_logged_in)
    server.send(delete_account)
    response, client_logged_in = test_client.receive(server)

exit()

