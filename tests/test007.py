import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal

# Sending to nonexistent user test

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

create_account = test_client.process("Create Account", client_logged_in, username="yush")
server.send(create_account)
response, client_logged_in = test_client.receive(server)

send_message = test_client.process("Send", client_logged_in, username="yush", recipient="mhla", data=TEXT)
server.send(send_message)
response, client_logged_in = test_client.receive(server)

if response != "Sorry, message recipient not found. Please try again.":
    print('TEST007 FAILED: send error')
else:
    print('TEST007 PASSED')



delete_account = test_client.process("Delete Account", client_logged_in)
server.send(delete_account)
response, client_logged_in = test_client.receive(server)

exit()