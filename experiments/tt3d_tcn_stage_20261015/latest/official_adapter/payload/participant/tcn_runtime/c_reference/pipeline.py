"""Bounded ordered RGB queue; all worker completion stays in timed video calls."""
import queue
import threading


class GeometryWorker:
    def __init__(self,engine):
        self.engine=engine;self.queue=queue.Queue(maxsize=8);self.error=None;self.closed=False;self.maximum_queue=0
        self.thread=threading.Thread(target=self._run,name='table_geometry_worker',daemon=True);self.thread.start()

    def _run(self):
        try:
            while True:
                item=self.queue.get()
                if item is None:return
                frame,rgb=item;self.engine.process(frame,rgb)
        except BaseException as error:self.error=error

    def _put(self,item):
        while True:
            if self.error is not None:raise RuntimeError('Geometry worker failed') from self.error
            try:self.queue.put(item,timeout=.1);self.maximum_queue=max(self.maximum_queue,self.queue.qsize());return
            except queue.Full:continue

    def submit(self,frame,rgb):
        if self.closed:raise ValueError('Closed video pipeline')
        self._put((frame,rgb.copy()))

    def close(self):
        if not self.closed:
            self.closed=True;self._put(None);self.thread.join()
        if self.error is not None:raise RuntimeError('Geometry worker failed') from self.error
        result=self.engine.result();result['maximum_queue_frames']=self.maximum_queue
        return result

    def abort(self):
        if self.closed:return
        self.closed=True
        if self.error is None:
            while True:
                try:self.queue.get_nowait()
                except queue.Empty:break
            self._put(None)
        self.thread.join()
