import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal
import time


# Checking to ensure no logins from two different places

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

    time.sleep(2)


    delete_account = test_client.process("Delete Account", client_logged_in)
    server.send(delete_account)
    response, client_logged_in = test_client.receive(server)
    exit()

def thread_malicious():

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if len(sys.argv) != 3:
        print("Correct usage: script, IP address, port number")
        exit()
    IP_address = str(sys.argv[1])
    Port = int(sys.argv[2])

    server.connect((IP_address, Port))

    client_logged_in = False

    time.sleep(1)

    login_account = test_client.process("Login", client_logged_in, username="yush")
    server.send(login_account)
    response, client_logged_in = test_client.receive(server)

    if response != "Username logged in elsewhere. Please try again.":
        print('TEST009 FAILED: double login')
    else:
        print('TEST009 PASSED')

    exit()


    

thread1 = threading.Thread(target=thread, daemon=True)
thread2 = threading.Thread(target=thread_malicious, daemon=True)

thread1.start()
thread2.start()


thread1.join()
thread2.join()

exit()