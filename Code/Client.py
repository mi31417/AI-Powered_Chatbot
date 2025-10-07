import socket


HEADER = 64
PORT = 5050
FORMAT= 'utf-8'
DISCONNECT_MSG = 'Dissconnected!'
SERVER ='10.200.154.196'
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode(FORMAT)
    msg_len = len(message)
    send_len = str(msg_len).encode(FORMAT)
    send_len += b' ' *(HEADER - len(send_len))
    client.send(send_len)
    client.send(message)

print(f'[CONNECTED] client is connected to server')
send('Hi! is it working ?')
send(DISCONNECT_MSG)

send('bye')
