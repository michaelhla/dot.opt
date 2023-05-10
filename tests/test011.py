import threading
import subprocess
import select
import sys
import socket
import test_client
import os
import signal
import time
import random
import string

# Load management on server

NUM_THREADS = 4

messages = ['Want to get a meal sometime?',
            'What are you studying?', 'Not much, just grinding', "Where are you from?",
            "What classes are you taking?", "What's the move?", "What's your concentration?", "What are your summer plans?"]


def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

def generate_random_string_array(n, min_length=5, max_length=10):
    return [generate_random_string(random.randint(min_length, max_length)) for _ in range(n)]


bot_array = generate_random_string_array(NUM_THREADS)

bot_lock = threading.Lock()

def thread(user, t):
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

    for i in range(0, 2000):
        time.sleep(t)
        send_message = test_client.process("Send", client_logged_in, username=user, recipient=bot_array[random.randint(0, NUM_THREADS-1)], data=messages[random.randint(0, len(messages)-1)])
        server.send(send_message)
        response, client_logged_in = test_client.receive(server)
        print(response)

thread_list = []

for i in range(0, NUM_THREADS):
    thread_list.append(threading.Thread(target=thread(bot_array[i], i/100000), daemon=True))

for thread in thread_list:
    thread.start()

for thread in thread_list:
    thread.join()

exit()






