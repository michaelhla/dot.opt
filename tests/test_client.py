# Python program to implement client side of chat room.
import socket
import select
import sys

MAX_MESSAGE_LENGTH = 280
MAX_RECIPIENT_LENGTH = 50

# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# if len(sys.argv) != 3:
#     print("Correct usage: script, IP address, port number")
#     exit()
# IP_address = str(sys.argv[1])
# Port = int(sys.argv[2])
# server.connect((IP_address, Port))


# # Keywords that client side parses and tags to send to the server
# MESSAGE_KEYS = ['Create Account', 'Login', 'Logout',
#                 'Delete Account', 'Send', 'List Accounts']



# # global client state of logged in or logged out
# global client_logged_in
# client_logged_in = False



# always returns encoded message
def process(message, client_logged_in, username=None, recipient=None, data=None, query=None):
    message = message.rstrip()
    # Messages are first entered as types, and are tagged based on those types
    if message.find('Create Account') == 0:
        if client_logged_in == True:
            print("Please logout to create an account.")
            return
        name = username
        # Ensures created accounts are no more than the max alotted length
        if len(name) <= MAX_RECIPIENT_LENGTH:
            pass
        else:
            print("All usernames must be at most " +
                str(MAX_RECIPIENT_LENGTH) + " characters. Please try again.")
            return
        message = name
        message = message.encode()
        tag = (0).to_bytes(1, "big")
    elif message.find('Login') == 0:
        if client_logged_in == True:
            print("Please logout to login.")
            return
        message = username
        message = message.encode()
        tag = (1).to_bytes(1, "big")
    elif message.find('Logout') == 0:
        if client_logged_in == False:
            print("Currently logged out. Please create an account or login.")
            return
        message = ""
        message = message.encode()
        tag = (2).to_bytes(1, "big")
    elif message.find('Delete Account') == 0:
        if client_logged_in == False:
            print("Currently logged out. Please create an account or login.")
            return
        message = ""
        message = message.encode()
        tag = (3).to_bytes(1, "big")
    elif message.find("Send") == 0:
        if client_logged_in == False:
            print("Currently logged out. Please create an account or login.")
            return
        # print("To: ")
        # Ensures recipient is in the length limit, for wire protocol tagging
        if len(recipient) <= MAX_RECIPIENT_LENGTH:
            pass
        else:
            print("All usernames are at most " +
                    str(MAX_RECIPIENT_LENGTH) + " characters. Please try again.")
            return
        len_r = len(recipient)
        recep_tag = (len_r).to_bytes(1, "big")
        recipient = recipient.encode()
        # print("Message: ")
        message = data
        # Ensures message is in the length limit
        if len(message) <= MAX_MESSAGE_LENGTH:
            pass
        else:
            print("Message must be at most " +
                    str(MAX_MESSAGE_LENGTH) + " characters. Please try again.")
            return
        message = message.encode()
        type_tag = (4).to_bytes(1, "big")
        # Wire protocol tags with type, then length of receiving username for metadata parsing, and the receiving username
        tag = type_tag + recep_tag + recipient
    # Dump messages on demand
    elif message.find("Open Undelivered Messages") == 0:
        if client_logged_in == False:
            print("Currently logged out. Please create an account or login.")
            return
        message = ""
        message = message.encode()
        tag = (5).to_bytes(1, "big")
    elif message.find("List Accounts") == 0:
        message = query
        message = message.encode()
        tag = (6).to_bytes(1, "big")
    else:
        print('Input not recognized. Please try again.')
        return

    bmsg = tag + message
    return bmsg



def receive(socks):
    message = socks.recv(2048)
    tag = message[0]
    val = False
    if tag == 0:
        val = False
    elif tag == 1:
        val = True
    else:
        pass
    dmessage = message[1:].decode()
    return dmessage, val






# while True:

#     # maintains a list of possible input streams
#     sockets_list = [1, server]

#     """ There are two possible input situations. Either the
#     user wants to give manual input to send to other people,
#     or the server is sending a message to be printed on the
#     screen. Select returns from sockets_list, the stream that
#     is reader for input. So for example, if the server wants
#     to send a message, then the if condition will hold true
#     below.If the user wants to send a message, the else
#     condition will evaluate as true"""
#     read_sockets, write_socket, error_socket = select.select(
#         sockets_list, [], [])
    

#     for socks in read_sockets:
#         if socks == server:
#             receive(socks, client_logged_in)
#         else:
#             bmsg = process("Create Account", client_logged_in, username="yush")
#             if bmsg:
#                 try:
#                     server.send(bmsg)
#                 except:
#                     print('Message could not send.')
# server.close()


# To Do: If we log in, and try to log in again, do we have to have a client side error, or can we just let that be a null operation (second option probably good too)
# To Do: Probably more comments about the wire protocol
