import socket

HOST = "192.168.1.100"  # <-- Replace with ESP IP from Serial Monitor
PORT = 1234

# Create TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to ESP server
s.connect((HOST, PORT))

# =========================
# SEND command to ESP
# =========================
s.sendall(b"ping\n")   # newline is important!

while True:
    # =========================
    # RECEIVE from ESP
    # =========================
    data = s.recv(1024).decode().strip()

    if data:
        print("Received:", data)

        # =========================
        # RESPOND to ESP
        # =========================
        if data == "hello_from_esp":
            s.sendall(b"ack\n")


# Experiment
    # 1) No exo: Bend Index finger
    # 2) Exo: passiv bend index
    # 3) Exo: active bend index

# Record EMG
# Protocol (Depends on active state)
    # Rest                                      3s                              [REST]
    # Duration to bend index to position        3 + x                           [CONTRACT]
    # Hold-on duration of 2 seconds             3 + x + 2                       [CONTRACT]
    # Release to init position                  3 + x + 2 + y = total time      [RELEASE]