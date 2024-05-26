import numpy as np
import pyaudio
import pyqtgraph as pg
import webrtcvad
from PySide6 import QtCore, QtGui, QtWidgets

FS = 16000  # Sampling rate
CHUNKSZ = int(FS / 1000 * 30)  # 30 ms frames


class MicrophoneRecorder:
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=FS,
            input=True,
            frames_per_buffer=CHUNKSZ,
        )

    def read(self):
        data = self.stream.read(CHUNKSZ, exception_on_overflow=False)
        y = np.frombuffer(data, dtype=np.int16)
        self.signal.emit(y)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class SpectrogramWidget(pg.PlotItem):
    read_collected = QtCore.Signal(np.ndarray)

    def __init__(self):
        super(SpectrogramWidget, self).__init__()

        self.img = pg.ImageItem()
        self.addItem(self.img)

        # To show 5 seconds of data, we need FS * 5 / CHUNKSZ chunks of data
        self.img_array = np.zeros((int(FS * 5 / CHUNKSZ), int(CHUNKSZ / 2 + 1)))

        # bipolar colormap
        pos = np.array([0.0, 1.0, 0.5, 0.25, 0.75])
        color = np.array(
            [
                [0, 255, 255, 255],
                [255, 255, 0, 255],
                [0, 0, 0, 255],
                (0, 0, 255, 255),
                (255, 0, 0, 255),
            ],
            dtype=np.ubyte,
        )
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50, 40])

        # setup the correct scaling for y-axis
        freq = np.arange((CHUNKSZ / 2) + 1) / (float(CHUNKSZ) / FS)
        yscale = 1.0 / (self.img_array.shape[1] / freq[-1])
        transform = QtGui.QTransform()
        transform.scale((5.0 / self.img_array.shape[0]), yscale)
        self.img.setTransform(transform)

        self.setLabel("left", "Frequency", units="Hz")
        self.setLabel("bottom", "Time", units="s")

        # prepare window for later use
        self.win = np.hanning(CHUNKSZ)
        self.show()

    def update(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(chunk * self.win) / CHUNKSZ
        # get magnitude
        psd = np.abs(spec)
        # convert to dB scale
        psd = 20 * np.log10(psd)

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1, :] = psd

        self.img.setImage(self.img_array, autoLevels=False)


class AudioPlotWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super(AudioPlotWidget, self).__init__()
        self.resize(1000, 600)
        self.setWindowTitle("Real-Time Audio Plot and Spectrogram")

        self.audio_plot = self.addPlot(title="Audio Signal")
        self.audio_curve = self.audio_plot.plot()
        self.audio_plot.setYRange(-32768, 32767)
        self.audio_plot.setLabel("left", "Amplitude")
        self.audio_plot.setLabel("bottom", "Time", units="s")
        self.audio_plot.setXRange(0, 5)  # Set x-axis to display 5 seconds of data
        self.nextRow()

        self.spectrogram_plot = SpectrogramWidget()
        self.addItem(self.spectrogram_plot)

        self.data_buffer = np.zeros(
            CHUNKSZ * int(FS * 5 / CHUNKSZ)
        )  # Buffer to hold 5 seconds of data
        self.voice_activity_lines = []
        self.is_speech_active = False

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Set VAD mode (0-3), higher values are more aggressive

        self.spectrogram_plot.read_collected.connect(self.update)

    def update(self, audio_data):
        # Update waveform
        self.data_buffer = np.roll(self.data_buffer, -CHUNKSZ)
        self.data_buffer[-CHUNKSZ:] = audio_data
        time_axis = np.linspace(0, 5, len(self.data_buffer))
        self.audio_curve.setData(time_axis, self.data_buffer)

        # Handle voice activity lines
        current_time = len(self.data_buffer) / FS
        is_speech = self.vad.is_speech(audio_data.tobytes(), FS)

        if is_speech and not self.is_speech_active:
            # Add a new line at the start of speech
            line = pg.InfiniteLine(pos=current_time, angle=90, pen=pg.mkPen("r"))
            self.audio_plot.addItem(line)
            self.voice_activity_lines.append({"line": line, "type": "start"})
            self.is_speech_active = True
        elif not is_speech and self.is_speech_active:
            # Add a new line at the end of speech
            line = pg.InfiniteLine(pos=current_time, angle=90, pen=pg.mkPen("g"))
            self.audio_plot.addItem(line)
            self.voice_activity_lines.append({"line": line, "type": "end"})
            self.is_speech_active = False

        self.audio_plot.setTitle(
            "Speech Detected" if is_speech else "No Speech Detected"
        )

        # Remove lines that are no longer visible
        for activity in self.voice_activity_lines:
            activity["line"].setPos(activity["line"].value() - CHUNKSZ / FS)
        self.voice_activity_lines = [
            activity
            for activity in self.voice_activity_lines
            if activity["line"].value() > 0
        ]

        # Update spectrogram
        self.spectrogram_plot.update(audio_data)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = AudioPlotWidget()

    mic = MicrophoneRecorder(w.spectrogram_plot.read_collected)

    # time (seconds) between reads
    interval = FS / CHUNKSZ
    t = QtCore.QTimer()
    t.timeout.connect(mic.read)
    t.start(1000 / interval)  # QTimer takes ms

    w.show()
    app.exec()

    mic.close()
