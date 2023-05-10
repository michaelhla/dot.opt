import time
import socket
import traceback
import sys
from _thread import *
from threading import Lock
import re
import json
import os
import json
import select
import threading

user_state_dictionary = {}
client_dictionary = {}
message_queue = {}
msg_db = {}

dict_lock = Lock()

username = 'mike'

update = (0).to_bytes(1, "big") + (len(username)
                                   ).to_bytes(1, "big") + username.encode()

# normal handle message behavior


def handle_message(message, tag=None):
    tag = message[0]
    # Wire protocol for replica interaction is different; need to read out the username relevant to the state change first
    length_of_username = message[1]
    username = message[2:2+length_of_username].decode()

    if tag == 0 and username != '':
        # acquire lock for client_dictionary, with timeout in case of failure
        dict_lock.acquire(timeout=10)
        if username in user_state_dictionary.keys():
            pass
        else:
            # backup stores username as logged off, as we will automatically log off clients when the server crashes
            client_dictionary[username] = 0
            user_state_dictionary[username] = 0
            message_queue[username] = []
            msg_db[username] = []
            # write(0)
            # write(1)
            # write(2)
        dict_lock.release()

    if tag == 3:
        # deletes the username from backup server state
        dict_lock.acquire(timeout=10)
        client_dictionary.pop(username)
        user_state_dictionary.pop(username)
        if username in message_queue.keys():
            message_queue.pop(username)
        # write(0)
        # write(2)
        dict_lock.release()

    if tag == 4:
        # adding messages to queue

        length_of_recep = message[2+length_of_username]  # convert to int
        recep_username = message[3+length_of_username:3 +
                                 length_of_username+length_of_recep].decode()

        dict_lock.acquire(timeout=10)
        # Checks if recipeint is actually a possible recipient
        if recep_username not in user_state_dictionary.keys():
            pass
        else:
            queue_tag = message[3+length_of_username+length_of_recep]
            text_message = message[4+length_of_username +
                                   length_of_recep:].decode()
            # Checks if recipient logged out
            if queue_tag == 0:
                message_queue[recep_username].append(
                    [username, text_message])
                # write(2)

            # If logged in, look up connection in dictionary
            else:
                msg_db[recep_username].append(
                    [username, text_message])
                # write(1)
        dict_lock.release()

    if tag == 5:
        # notification that message queue has been read from
        dict_lock.acquire(timeout=10)
        # TO DO: for message in message_queue, persistent store the successfully sent message

        # current active message_queue is empty in backup state
        msg_db[username].append(message_queue[username])
        message_queue[username] = []
        # TO DO: overwrite the persistent store state of the new current message_queue
        # write(1)
        # write(2)
        dict_lock.release()


handle_message(update)

if "mike" in msg_db.keys() and 'mike' in user_state_dictionary.keys() and 'mike' in message_queue.keys():
    print('TEST012 PASSED')
else:
    print('TEST012 FAILED')
