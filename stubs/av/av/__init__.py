class _FakeLogging:
    ERROR = 16
    WARNING = 24
    INFO = 32
    DEBUG = 48
    VERBOSE = 40

    def set_level(self, level):
        pass

    def get_level(self):
        return self.ERROR


class _FakeVideoFrame:
    pict_type = "I"


class _FakeFrame:
    VideoFrame = _FakeVideoFrame


class _FakeVideo:
    frame = _FakeFrame()


class _FakeAudioFrame:
    pass


class _FakeAudioFrameModule:
    AudioFrame = _FakeAudioFrame


class _FakeAudio:
    frame = _FakeAudioFrameModule()


class _FakeContainer:
    streams = type("streams", (), {"video": [], "audio": []})()
    duration = 0
    bit_rate = 0

    def decode(self, *args, **kwargs):
        return iter([])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


logging = _FakeLogging()
video = _FakeVideo()
audio = _FakeAudio()


class AVError(Exception):
    pass


def open(*args, **kwargs):
    return _FakeContainer()
