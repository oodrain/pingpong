"""Bounded playback preparation on the existing video decoder thread."""

import sys
from queue import Full, Queue
from threading import Event


class ReviewPlayback:
    def __init__(self, executor, reader, first, last, bounds, prepare):
        self.frames = Queue(maxsize=6)
        self.cancelled = Event()
        self.bounds = bounds
        self.last = last
        self.startup_frames = min(3, last - first + 1)
        self.timer = None
        self.task = executor.submit(self._fill, reader, first, prepare)

    def _fill(self, reader, first, prepare):
        for frame in range(first, self.last + 1):
            if self.cancelled.is_set():
                return
            image, pts = reader.get(frame)
            bounds = self.bounds
            preview = (bounds, prepare(image, bounds, fast=True)) if bounds is not None else None
            while not self.cancelled.is_set():
                try:
                    self.frames.put((frame, image, pts, preview), timeout=.015)
                    break
                except Full:
                    continue

    def start_clock(self):
        if sys.platform != "win32" or sys.getwindowsversion().build < 19041:
            return
        from ctypes import windll

        try:
            timer = windll.winmm
            if timer.timeBeginPeriod(1) == 0:
                self.timer = timer
        except (AttributeError, OSError):
            pass

    def cancel(self):
        self.cancelled.set()
        self.task.cancel()
        if self.timer is not None:
            self.timer.timeEndPeriod(1)
            self.timer = None
