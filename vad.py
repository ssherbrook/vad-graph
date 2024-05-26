import numpy as np
import pyaudio
import pyqtgraph as pg
import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

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

    def run(self):
        self.stream.start_stream()
        while self.stream.is_active():
            self.msleep(50)

    def callback(self, in_data, frame_count, time_info, status):
        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        audio_float32 = int2float(audio_chunk)
        confidence = self.model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
        self.update_plot.emit(audio_chunk, confidence)
        return (in_data, pyaudio.paContinue)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


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
        self.plot_audio.setYRange(-32768, 32767)  # Fixed y-axis for int16 range
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
