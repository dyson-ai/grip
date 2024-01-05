import time


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.duration = 0

    def start(self, duration=0):
        self.t0 = time.time()
        self.duration = duration

    def elapsed(self):
        return time.time() - self.t0

    def finished(self):
        return self.elapsed() >= self.duration
