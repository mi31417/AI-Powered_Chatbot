import socket
import threading

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())

ADDR = (SERVER, PORT)
FORMAT= 'utf-8'
DISCONNECT_MSG = 'Dissconnected!'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)



def handle_client(conn, addr):
    print(f'[NEW CONNECTION] {addr} connected.') 

    connected = True
    while connected:
        msg_len= conn.recv(HEADER).decode(FORMAT)
        if msg_len:

            msg_len = int(msg_len)
            msg = conn.recv(msg_len).decode(FORMAT)
            if msg == DISCONNECT_MSG:
                connected= False

            print(f"[{addr}] {msg}")

    conn.close()

    

def start():
    server.listen()
    print(f"[LISTENTING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client,args=(conn, addr))
        thread.start()


print("Server is starting ......")
start()
