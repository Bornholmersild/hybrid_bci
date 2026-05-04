import time
import socket
import msvcrt

HOST = "10.126.128.129"  # <-- Replace with ESP IP from Serial Monitor
PORT = 1234

# One rotation : 1494
# 4 rotatins max : 5975 - 3.5 sec

# Create TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to ESP server
s.connect((HOST, PORT))

# =========================
# SEND command to ESP
# =========================
# Handshake
s.sendall(b"connect\n")

while True:
    
    data = s.recv(1024).decode().strip()            # recv : receive data of byte size

    if data:
        print("Received:", data)

        if data == "connection complete":
            print("Connected to ESP")
            break

while True:
    # Get command
    cmd = input("Command (contract/release + ticks): ").strip().lower()

    # Validate
    parts = cmd.split()
    if len(parts) != 2:
        print("Invalid format. Use: contract 500")
        continue

    action, ticks = parts
    if action not in ["contract", "release"] or not ticks.isdigit():
        print("Invalid input")
        continue

    # Send command
    s.sendall((cmd + "\n").encode())

    print("Waiting for motor to finish...")

    # BLOCK until response comes
    while True:
        data = s.recv(1024)   # ← blocking call (no timeout)

        if data:
            msg = data.decode().strip()
            print("ESP:", msg)

            # Break when motor is done
            if "STOPPED" in msg or "DONE" in msg:
                print("Motor finished\n")
                break
            else:
                break
    


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