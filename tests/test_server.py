# Python program to implement server side of chat room.
import socket
import select
import sys
'''Replace "thread" with "_thread" for python 3'''
from _thread import *
from threading import Lock
import re


"""The first argument AF_INET is the address domain of the
socket. This is used when we have an Internet Domain with
any two hosts The second argument is the type of socket.
SOCK_STREAM means that data or characters are read in
a continuous flow."""
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

if len(sys.argv) != 3:
    print("Correct usage: script, IP address, port number")
    exit()

# IP address is first argument
IP_address = str(sys.argv[1])


# Port number is second argument
port = int(sys.argv[2])

# Server initialized at input IP address and port
server.bind((IP_address, port))

server.listen()

# maintains a list of potential clients
list_of_clients = []


# client username dictionary, with login status: 0 if logged off, corresponding address if logged in
client_dictionary = {}


# lock for dictionary
dict_lock = Lock()

# message queues per username
message_queue = {}


def clientthread(conn, addr):

    client_state = True
    logged_in = False
    username = None
    while client_state:

        # maintain a state variable as logged in or logged off
        # while logged off, logged_in = False
        print("here again")
        # sends a message to the client whose user object is conn
        # message = "Welcome to Messenger! Please login or create an account:"
        # return_tag = (0).to_bytes(1, "big")
        # bmsg = return_tag + message.encode()
        # conn.sendall(bmsg)

        # client can only create an account or login while client state is False
        # To Do: big endian byte interpretation
        while logged_in == False:
            try:
                message = conn.recv(2048)
                # check if message is of type create account or login
                # wire protocol demands initial byte is either 0 (create) or 1 (login) here
                # tag = int.from_bytes(message[0], "big")
                tag = message[0]

                # account creation
                if tag == 0:
                    print('create')
                    username = message[1:]
                    username = username.decode()
                    # If the username is in existence, server asks to retry.

                    # acquire lock for client_dictionary, with timeout in case of failure
                    dict_lock.acquire(timeout=10)

                    if username in client_dictionary.keys():
                        message = "The account " + username + " already exists. Please try again."
                        return_tag = (0).to_bytes(1, "big")
                        bmsg = return_tag + message.encode()
                        conn.sendall(bmsg)
                    else:
                        client_dictionary[username] = conn
                        message_queue[username] = []
                        message = "Account created. Welcome " + username + "!"
                        return_tag = (1).to_bytes(1, "big")
                        bmsg = return_tag + message.encode()
                        conn.sendall(bmsg)
                        logged_in = True

                    dict_lock.release()

                # login
                if tag == 1:
                    username = message[1:]
                    username = username.decode()

                    # acquire lock for client_dictionary, with timeout in case of failure
                    dict_lock.acquire(timeout=10)

                    if username not in client_dictionary.keys():
                        message = "Username not found. Please try again."
                        return_tag = (0).to_bytes(1, "big")
                        bmsg = return_tag + message.encode()
                        conn.sendall(bmsg)
                    else:
                        # Check if username logged in elsewhere (i.e. dictionary returns 1)
                        if client_dictionary[username] != 0:
                            message = "Username logged in elsewhere. Please try again."
                            return_tag = (0).to_bytes(1, "big")
                            bmsg = return_tag + message.encode()
                            conn.sendall(bmsg)
                        else:
                            client_dictionary[username] = conn
                            message = "Welcome back " + username + "!"
                            return_tag = (1).to_bytes(1, "big")
                            bmsg = return_tag + message.encode()
                            conn.sendall(bmsg)
                            logged_in = True
                    print(client_dictionary)
                    dict_lock.release()
                # List Accounts if logged out
                if tag == 6:
                    query = message[1:].decode()
                    dict_lock.acquire(timeout=10)
                    users = match(query)
                    dict_lock.release()
                    if users == '':
                        res = 'No users found'
                    else:
                        res = "Users matching " + query + ':\n'
                        res += users
                    
                    return_tag = (2).to_bytes(1, "big")
                    bmsg = return_tag + res.encode()
                    conn.sendall(bmsg)
            except:
                continue

        # now suppose that the client is logged in
        # allowable actions are: list accounts, send message, log off, delete account, dump queue

        while logged_in == True:
            try:
                message = conn.recv(2048)
                if message:
                    tag = message[0]

                    # Logout
                    if tag == 2:
                        # Acquire dict lock, change logged in state to false and remove address info in client dictionary
                        logout(username)
                        message = username + " successfully logged out."
                        return_tag = (0).to_bytes(1, "big")
                        bmsg = return_tag + message.encode()
                        conn.sendall(bmsg)
                        logged_in = False

                    # Delete Account
                    if tag == 3:  # start of delete account
                        # Acquire dict lock, remove username from client dictionary and message_queue
                        dict_lock.acquire(timeout=10)
                        client_dictionary.pop(username)
                        message_queue.pop(username)
                        logged_in = False
                        dict_lock.release()
                        message = "Account " + username + " successfully deleted."
                        return_tag = (0).to_bytes(1, "big")
                        bmsg = return_tag + message.encode()
                        conn.sendall(bmsg)

                    # Send Message
                    if tag == 4:
                        # Wire Protocol: tag-length of username (< 256 char by demand) - recepient - message
                        length_of_recep = message[1]  # convert to int
                        recep_username = message[2:2+length_of_recep].decode()

                        dict_lock.acquire(timeout=10)
                        # To Do: Should we make this more granular with locking? Like variables for recep in client_dictionary
                        # Checks if recipeint is actually a possible recipient
                        if recep_username not in client_dictionary.keys():
                            message = "Sorry, message recipient not found. Please try again."
                            return_tag = (1).to_bytes(1, "big")
                            bmsg = return_tag + message.encode()
                            conn.sendall(bmsg)
                        else:
                            text_message = message[2+length_of_recep:].decode()
                            # Checks if recipient logged out
                            if client_dictionary[recep_username] == 0:
                                message_queue[recep_username].append(
                                    [username, text_message])
                                confirmation_message = "Message successfully sent."
                                return_tag = (1).to_bytes(1, "big")
                                bmsg = return_tag + confirmation_message.encode()
                                conn.sendall(bmsg)
                            # If logged in, look up connection in dictionary
                            else:
                                recep_conn = client_dictionary[recep_username]
                                new_message = "<"+username+">: " + text_message
                                try:
                                    return_tag = (1).to_bytes(1, "big")
                                    bmsg = return_tag + new_message.encode()
                                    recep_conn.sendall(bmsg)
                                    confirmation_message = "Message successfully sent."
                                    bmsg = return_tag + confirmation_message.encode()
                                    conn.sendall(bmsg)
                                # If sending fails, let the sender know; otherwise, send confirmation to sender
                                except:
                                    error_message = "Sorry, message could not be sent. Please try again."
                                    return_tag = (1).to_bytes(1, "big")
                                    bmsg = return_tag + error_message.encode()
                                    conn.sendall(bmsg)

                        # To Do: Ask about blocking and ask about timeouts? What happened if send fails?
                        dict_lock.release()

                    # Dump Message queue
                    if tag == 5:
                        # Message Queue preconfigured so it stores packet of username, message
                        for undelivered in message_queue[username]:
                            sender = undelivered[0]
                            undel_message = undelivered[1]
                            # To Do: username will have /n at the end
                            message = "<" + sender + ">: " + undel_message
                            return_tag = (1).to_bytes(1, "big")
                            bmsg = return_tag + message.encode()
                            conn.sendall(bmsg)

                    if tag == 6:
                        query = message[1:].decode()
                        dict_lock.acquire(timeout=10)
                        users = match(query)
                        dict_lock.release()
                        if users == '':
                            res = 'No users found'
                        else:
                            res = "Users matching " + query + ':\n'
                            res += users
                        
                        return_tag = (2).to_bytes(1, "big")
                        bmsg = return_tag + res.encode()
                        conn.sendall(bmsg)

                else:
                    """message may have no content if the connection
                    is broken, in this case we remove the connection"""
                    remove(conn, username)
                    logged_in = False
                    client_state = False

            # except Exception as e:
            except:
                continue


def logout(username):
    dict_lock.acquire(timeout=10)
    client_dictionary[username] = 0
    dict_lock.release()
    # To Do: What happens if these fail


def remove(connection, username):
    # If connection breaks, automatically logs username out
    logout(username)
    if connection in list_of_clients:
        print(f"{connection} has left")
        list_of_clients.remove(connection)

# not sure how to implement this efficiently/without locking the entire dictionary, maybe keep list of all users that have logged in separately


def match(query):
    message = ''
    query = query.replace('*', ".*")
    for key in client_dictionary.keys():
        match = re.search(query, key)
        if match is not None:
            message += key + " "
    # print(message)
    return message

    # def send_message(message):
    #     sender = [i for i in client_dictionary if client_dictionary[i]==addr]
    #     receiver = message[2:]
    #     proc_message = "<" + sender + "> " + message[3:]
    #     dict_lock.acquire(timeout=10)
    #     if client_dictionary[receiver] != 0:
    #         rec_conn = client_dictionary.find(receiver[1])
    #         rec_conn.sendall(proc_message.encode())
    #         conn.send("delivered".encode())
    #     else:
    #         message_queue[receiver] += message
    # """Using the below function, we broadcast the message to all
    # clients who's object is not the same as the one sending
    # the message """
    # def broadcast(message, connection):
    #     for clients in list_of_clients:
    #         if clients!=connection:
    #             try:
    #                 clients.send(message.encode())
    #             except:
    #                 clients.close()
    #                 # if the link is broken, we remove the client
    #                 remove(clients)
    # """The following function simply removes the object
    # from the list that was created at the beginning of
    # the program"""
while True:

    """Accepts a connection request and stores two parameters,
    conn which is a socket object for that user, and addr
    which contains the IP address of the client that just
    connected"""
    conn, addr = server.accept()

    """Maintains a list of clients for ease of broadcasting
    a message to all available people in the chatroom"""
    list_of_clients.append(conn)

    # prints the address of the user that just connected
    print(addr[0] + " connected")

    # creates and individual thread for every user
    # that connects
    start_new_thread(clientthread, (conn, addr))

conn.close()
server.close()
