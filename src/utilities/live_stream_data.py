from .pytrigno import TrignoEMG
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import butter, sosfilt, iirnotch
from math import ceil

# ---------------- CONFIG ----------------
CHANNEL_RANGE = (1, 2)       # 2 channels: 0 and 1
SAMPLES_PER_READ = 200        # ~10 ms latency at 2 kHz
FS = 2000
WINDOW_SEC = 1
BUFFER_LEN = FS * WINDOW_SEC
# ----------------------------------------

def design_filters(fs):
    # Band-pass 20–450 Hz (4th order)
    sos_bp = butter(4, [20, 450], btype='bandpass', fs=fs, output='sos')

    # Notch 50 Hz (Q ≈ 30)
    b_notch, a_notch = iirnotch(50, 30, fs=fs)

    # Envelope low-pass 10 Hz (2nd order)
    sos_env = butter(2, 10, btype='lowpass', fs=fs, output='sos')

    return sos_bp, (b_notch, a_notch), sos_env

def stream_data():
    emg = TrignoEMG(
        channel_range=CHANNEL_RANGE,
        samples_per_read=SAMPLES_PER_READ,
        units='mV'
    )

    num_channels = CHANNEL_RANGE[1] - CHANNEL_RANGE[0] + 1
    print("Streaming channels:", num_channels)

    # Ring buffer for plotting
    buffers = [deque(np.zeros(BUFFER_LEN), maxlen=BUFFER_LEN) for _ in range(num_channels)]

    # Setup plot
    plt.ion()
    n_rows = 1
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
    axes = axes.flatten()

    x = np.arange(BUFFER_LEN) / FS
    lines = []

    for ch in range(num_channels):
        ax = axes[ch]
        line, = ax.plot(x, np.zeros(BUFFER_LEN))
        ax.set_title(f"Sensor {ch}")
        ax.set_ylim(-0.5, 0.5)        # adjust as needed
        ax.set_xlim(0, WINDOW_SEC)
        lines.append(line)
    plt.show()

    emg.start()

    try:
        while True:
            # Read small batch new_chunk.shape = (channels, samples)
            new_chunk = emg.read()

            # Append new samples to ring buffer
            for ch in range(num_channels):
                buffers[ch].extend(new_chunk[ch])

            # Update plot lines
            for ch in range(num_channels):
                lines[ch].set_ydata(buffers[ch])

            plt.pause(0.001)  # allow GUI to update (very small delay)

    except KeyboardInterrupt:
        print("Stopped.")

    finally:
        print('FINISHED')
        emg.stop()
        plt.ioff()
        plt.show()

def stream_data_filt():
    emg = TrignoEMG(
        channel_range=CHANNEL_RANGE,
        samples_per_read=SAMPLES_PER_READ,
        units='mV'
    )

    num_channels = CHANNEL_RANGE[1] - CHANNEL_RANGE[0] + 1
    print(f"Streaming {num_channels} EMG channels")

    # Filters
    sos_bp, (b_notch, a_notch), sos_env = design_filters(FS)

    # States for IIR filters, per channel
    # sosfilt needs shape (n_sections, 2) for each channel
    n_sections_bp = sos_bp.shape[0]
    n_sections_env = sos_env.shape[0]

    zi_bp = np.zeros((num_channels, n_sections_bp, 2))
    zi_env = np.zeros((num_channels, n_sections_env, 2))

    # Ring buffers (envelope for plotting, but you could also use filtered EMG)
    buffers = [deque(np.zeros(BUFFER_LEN), maxlen=BUFFER_LEN) for _ in range(num_channels)]

    # ---------------- PLOT SETUP ----------------
    plt.ion()
    n_cols = 2
    n_rows = ceil(num_channels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
    axes = axes.flatten()

    x = np.arange(BUFFER_LEN) / FS
    lines = []

    for ch in range(num_channels):
        ax = axes[ch]
        line, = ax.plot(x, np.zeros(BUFFER_LEN), lw=1)
        ax.set_title(f"Sensor {ch}")
        #ax.set_ylim(-0.1, 0.1)           # adjust after seeing real envelope ranges
        ax.set_xlim(0, WINDOW_SEC)
        lines.append(line)

    # Hide unused axes
    for i in range(num_channels, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------

    emg.start()
    print("Streaming started... CTRL+C to stop.")

    try:
        while True:
            # new_chunk shape: (channels, samples)
            new_chunk = emg.read().astype(np.float32)

            # Process each channel independently
            for ch in range(num_channels):
                x_raw = new_chunk[ch]

                # --- 1) Band-pass 20–450 Hz (IIR) ---
                y_bp, zi_bp[ch] = sosfilt(sos_bp, x_raw, zi=zi_bp[ch])

                # --- 2) Optional: notch 50 Hz ---
                # If line noise is bad, you can uncomment:
                from scipy.signal import lfilter
                y_bp = lfilter(b_notch, a_notch, y_bp)

                # --- 3) Rectify ---
                y_rect = np.abs(y_bp)

                # --- 4) Envelope via 10 Hz low-pass ---
                #y_env, zi_env[ch] = sosfilt(sos_env, y_rect, zi=zi_env[ch])

                # Append to ring buffer (use envelope for plotting)
                buffers[ch].extend(y_rect)

            # Update plots
            for ch in range(num_channels):
                lines[ch].set_ydata(buffers[ch])

            plt.pause(0.001)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        emg.stop()
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    stream_data_filt()

    # import socket
    # s = socket.socket()
    # try:
    #     s.connect(("192.168.56.1", 50043))   # change IP to TCA IP
    #     print("Port OPEN")
    # except Exception as e:
    #     print("Port CLOSED:", e)

