import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal
import time

# Concurrent sending tests

TEXT = "The Chiefs are the best team in the NFL."
REPLY = "I agree."

RECORDS = ["", ""]


# t_1 is the time before sending, t_2 is the time before waiting to check for a receipt while sleeping
# for two parties, t_1 and t_2 are reversed
# one t_1 is longer than t_2 to mimic call and response of a conversation
def thread(user, recep, text, t_1, t_2):

    received_message = ""

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if len(sys.argv) != 3:
        print("Correct usage: script, IP address, port number")
        exit()
    IP_address = str(sys.argv[1])
    Port = int(sys.argv[2])

    server.connect((IP_address, Port))

    client_logged_in = False

    create_account = test_client.process("Create Account", client_logged_in, username=user)
    server.send(create_account)
    response, client_logged_in = test_client.receive(server)

    time.sleep(t_1)

    send_message = test_client.process("Send", client_logged_in, username=user, recipient=recep, data=text)
    server.send(send_message)
    response, client_logged_in = test_client.receive(server)

    if t_1 == 2:
        received_message = response
        RECORDS[0] = received_message
        

    time.sleep(t_2)

    response, client_logged_in = test_client.receive(server)
    if t_1 == 1:
        received_message = response
        RECORDS[1] = received_message
        

    delete_account = test_client.process("Delete Account", client_logged_in)
    server.send(delete_account)
    response, client_logged_in = test_client.receive(server)
    exit()


    

thread1 = threading.Thread(target=thread, args=("yush", "mhla", TEXT, 1, 2), daemon=True)
thread2 = threading.Thread(target=thread, args=("mhla", "yush", REPLY, 2, 1), daemon=True)


thread1.start()
thread2.start()


thread1.join()
thread2.join()

if RECORDS[0] != "<yush>: " + TEXT or RECORDS[1] != "<mhla>: " + REPLY:
    print('TEST008 FAILED: send error')
else:
    print('TEST008 PASSED')


exit()



