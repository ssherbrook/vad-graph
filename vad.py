import queue

import numpy as np
import pyaudio
import pyqtgraph as pg
import webrtcvad
from PySide6 import QtCore, QtWidgets

# Initialize PyAudio and WebRTC VAD
audio = pyaudio.PyAudio()
vad = webrtcvad.Vad()

# Set the VAD mode (0-3), higher values are more aggressive in filtering out non-speech
vad.set_mode(1)

# Define audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 1000 * 30)  # 30 ms frames

# Enumerate audio input devices
info = audio.get_host_api_info_by_index(0)
numdevices = info.get("deviceCount")
devices = []
for i in range(0, numdevices):
    if audio.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels") > 0:
        devices.append(
            (i, audio.get_device_info_by_host_api_device_index(0, i).get("name"))
        )

print("Available audio input devices:")
for i, device in devices:
    print(f"{i}: {device}")

# Get user's choice of audio input device
device_index = int(input("Select the device index: "))

# Queue to store audio data for plotting
audio_queue = queue.Queue()


# Callback function to process audio data
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    try:
        is_speech = vad.is_speech(in_data, RATE)
    except webrtcvad.Error as e:
        print(f"VAD error: {e}")
        is_speech = False
    audio_queue.put((audio_data, is_speech))
    return (in_data, pyaudio.paContinue)


# Open audio stream
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK,
    stream_callback=callback,
)

# Start the stream
stream.start_stream()

# Set up pyqtgraph window and plots
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Audio Plot")
win.resize(1000, 600)

# Waveform plot
waveform_plot = win.addPlot(title="Audio Signal")
waveform_curve = waveform_plot.plot()
waveform_plot.setYRange(-32768, 32767)
waveform_plot.setLabel("left", "Amplitude")
waveform_plot.setLabel("bottom", "Time", units="s")

# Data buffer for the plot (increase size to show more time)
data_buffer = np.zeros(
    CHUNK * 100
)  # Buffer to hold 100 chunks of data (3 seconds at 16000 Hz)

# Set x-axis range to match the data buffer size
waveform_plot.setXRange(0, len(data_buffer))

# List to store vertical lines indicating voice activity
voice_activity_lines = []
is_speech_active = False


def update_plot():
    global data_buffer, voice_activity_lines, is_speech_active
    if not audio_queue.empty():
        audio_data, is_speech = audio_queue.get()

        # Update waveform
        data_buffer = np.roll(data_buffer, -CHUNK)
        data_buffer[-CHUNK:] = audio_data
        waveform_curve.setData(data_buffer)

        # Handle voice activity lines
        current_time = len(data_buffer) - CHUNK

        if is_speech and not is_speech_active:
            # Add a new line at the start of speech
            line = pg.InfiniteLine(pos=current_time, angle=90, pen=pg.mkPen("r"))
            waveform_plot.addItem(line)
            voice_activity_lines.append({"line": line, "type": "start"})
            is_speech_active = True
        elif not is_speech and is_speech_active:
            # Add a new line at the end of speech
            line = pg.InfiniteLine(pos=current_time, angle=90, pen=pg.mkPen("g"))
            waveform_plot.addItem(line)
            voice_activity_lines.append({"line": line, "type": "end"})
            is_speech_active = False

        waveform_plot.setTitle("Speech Detected" if is_speech else "No Speech Detected")

        # Remove lines that are no longer visible
        for activity in voice_activity_lines:
            activity["line"].setPos(activity["line"].value() - CHUNK)
        voice_activity_lines = [
            activity
            for activity in voice_activity_lines
            if activity["line"].value() > 0
        ]


# Set up a timer to update the plot periodically
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(30)

# Start Qt event loop
if __name__ == "__main__":
    app.exec()

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()
