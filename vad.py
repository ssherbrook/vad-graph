import time
import wave

import numpy as np
import pyaudio
import pyqtgraph as pg
import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from scipy.signal import butter, sosfilt, sosfilt_zi

# Set up the VAD model
torch.set_num_threads(1)
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)
get_speech_timestamps, _, _, _, _ = utils


# Helper function to convert int16 to float32
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


# Helper function to convert float32 to int16
def float2int(sound):
    sound = np.clip(sound, -1, 1)
    sound = sound * 32768
    return sound.astype(np.int16)


# Bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def bandpass_filter(data, sos, zi):
    y, zf = sosfilt(sos, data, zi=zi)
    return y, zf


# Audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 512


class AudioStream(QThread):
    update_plot = Signal(np.ndarray, float)

    def __init__(self):
        super().__init__()
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback,
        )
        self.model = model
        self.frames = []
        self.sos = butter_bandpass(200, 7000, SAMPLE_RATE)  # Initialize filter
        self.zi = sosfilt_zi(self.sos)
        self.last_above_threshold_time = time.time()
        self.initialized = False  # To ensure proper initialization

    def run(self):
        self.stream.start_stream()
        while self.stream.is_active():
            self.msleep(50)

    def callback(self, in_data, frame_count, time_info, status):
        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        audio_float32 = int2float(audio_chunk)

        # Initialize filter state with silence to avoid oscillation
        if not self.initialized:
            silence = np.zeros(CHUNK, dtype=np.float32)
            _, self.zi = sosfilt(self.sos, silence, zi=self.zi)
            self.initialized = True

        # Apply bandpass filter
        audio_filtered, self.zi = bandpass_filter(audio_float32, self.sos, self.zi)

        # Convert to torch.float32
        audio_tensor = torch.from_numpy(audio_filtered).float()

        confidence = self.model(audio_tensor, SAMPLE_RATE).item()

        current_time = time.time()
        if confidence > 0.5:
            self.last_above_threshold_time = current_time
            audio_chunk = float2int(audio_filtered)
        else:
            if current_time - self.last_above_threshold_time >= 0.5:
                audio_chunk = np.zeros_like(audio_chunk)
            else:
                audio_chunk = float2int(audio_filtered)

        self.frames.append(audio_chunk.tobytes())
        self.update_plot.emit(audio_filtered, confidence)
        return (audio_chunk.tobytes(), pyaudio.paContinue)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        # Save audio to file
        wf = wave.open("output_with_agc_and_filter.wav", "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(self.frames))
        wf.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Voice Activity Detection")
        self.setGeometry(100, 100, 800, 600)

        self.audio_stream = AudioStream()
        self.audio_stream.update_plot.connect(self.update_plot)

        self.data = np.zeros(SAMPLE_RATE * 5)  # 5 seconds of data
        self.confidences = np.zeros(
            int(SAMPLE_RATE / CHUNK) * 5
        )  # 5 seconds of confidences

        self.graph_layout = pg.GraphicsLayoutWidget()

        self.plot_audio = self.graph_layout.addPlot(row=0, col=0)
        self.plot_audio.setYRange(-1, 1)  # Fixed y-axis for normalized float32 range
        self.plot_audio.setXRange(0, 5, padding=0)  # Fixed x-axis for 5 seconds
        self.plot_data = self.plot_audio.plot(self.data, pen="b")

        self.graph_layout.nextRow()

        self.plot_vad = self.graph_layout.addPlot(row=1, col=0)
        self.plot_vad.setYRange(0, 1)  # Fixed y-axis range for VAD
        self.plot_vad.setXRange(0, 5, padding=0)  # Fixed x-axis for 5 seconds
        self.plot_confidences = self.plot_vad.plot(self.confidences, pen="r")

        layout = QVBoxLayout()
        layout.addWidget(self.graph_layout)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.audio_stream.start()

    def update_plot(self, audio_chunk, confidence):
        self.data = np.roll(self.data, -len(audio_chunk))
        self.data[-len(audio_chunk) :] = audio_chunk

        self.confidences = np.roll(self.confidences, -1)
        self.confidences[-1] = confidence

        self.plot_data.setData(np.linspace(0, 5, len(self.data)), self.data)
        self.plot_confidences.setData(
            np.linspace(0, 5, len(self.confidences)), self.confidences
        )

    def closeEvent(self, event):
        self.audio_stream.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
