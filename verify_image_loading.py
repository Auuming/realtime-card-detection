"""Tk integration check: slow images must not block names or newer clicks."""
import json
import tempfile
import time
import tkinter as tk
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app import App
from recognition import DetailStore


def main():
    ids = ['7286819f-6c57-4503-898c-528786ad86e9', 'd0def54b-9f0a-4ab1-9df9-25506a06350c']
    entered = [Event(), Event()]
    release = [Event(), Event()]
    names = ['First card', 'Second card']

    def metadata(url, **kwargs):
        index = ids.index(url.rsplit('/', 1)[-1])
        return {'name': names[index], 'oracle_text': 'Rules available before image',
                'image_uris': {'normal': 'https://example.test/'+str(index)}}

    def image_download(url, target, **kwargs):
        index = int(url.rsplit('/', 1)[-1])
        entered[index].set()
        if not release[index].wait(8):
            raise RuntimeError('Test image release timed out')
        Image.new('RGB', (252,352), ['red','blue'][index]).save(target)

    class LocalRecognizer:
        score_label = 'Embedding similarity'
        description = 'Delayed image integration fixture'

        def __init__(self):
            self.store = DetailStore()
            self.cards = self.store.cards

        def identify_selection(self, frame, points):
            index = int(points[:,0].mean() > 300)
            key = self.store.load({'id': ids[index], 'name': names[index]}, cache_only=True)
            return key, .9, .1, None

    with tempfile.TemporaryDirectory() as folder:
        root_path = Path(folder)
        with patch('recognition.DATA',root_path), patch('recognition.ROOT',root_path), patch('app.ROOT',root_path), patch('recognition.api',side_effect=metadata) as api, patch('recognition.download',side_effect=image_download) as download:
            root = tk.Tk()
            app = App(root, recognizer=LocalRecognizer())

            def until(predicate, seconds=4):
                passed = []
                deadline = time.monotonic()+seconds
                def poll():
                    if predicate():
                        passed.append(True)
                        root.quit()
                    elif time.monotonic() > deadline:
                        root.quit()
                    else:
                        root.after(10,poll)
                root.after(10,poll)
                root.mainloop()
                assert passed, 'Timed out waiting for GUI state'

            def click(index):
                x,y = app.boxes[index].mean(axis=0)
                app.click(SimpleNamespace(x=x*app.scale+app.offset[0],y=y*app.scale+app.offset[1]))

            try:
                frame = np.full((400,650,3),100,np.uint8)
                app.set_frame(frame)
                app.detection_frame = app.frame.copy()
                app.boxes = [np.float32([[30,30],[230,30],[230,310],[30,310]]),
                             np.float32([[350,30],[550,30],[550,310],[350,310]])]
                until(lambda: hasattr(app,'scale'))
                start=time.perf_counter()
                click(0)
                assert app.image_label.cget('image'), 'Selected crop must appear immediately'
                until(lambda: app.selected_key == ids[0]+'_0' and entered[0].is_set())
                first_ms=(time.perf_counter()-start)*1000
                assert 'First card' in app.text.get('1.0','end')
                assert 'Rules available before image' in app.text.get('1.0','end')
                assert not release[0].is_set()
                start=time.perf_counter()
                click(1)
                until(lambda: app.selected_key == ids[1]+'_0')
                second_ms=(time.perf_counter()-start)*1000
                assert 'Second card' in app.text.get('1.0','end'), 'New click must not wait for previous download'
                release[0].set()
                until(lambda: entered[1].is_set())
                assert 'Second card' in app.text.get('1.0','end'), 'Old image must not replace the newer selection'
                assert 'Rules available before image' in app.text.get('1.0','end')
                release[1].set()
                until(lambda: app.asset_pending is None and 'Reference image' in app.image_status.get())
                assert app.image_label.cget('image')
                assert download.call_count == 2
                click(1)
                until(lambda: app.selected_key == ids[1]+'_0' and app.asset_pending is None)
                assert download.call_count == 2, 'Cached image must not download twice'
                assert api.call_count == 2
                # Invalidate a request like source/seek does; no result may restore it.
                app.clear_selection()
                until(lambda: app.detail_pending is None and app.asset_pending is None)
                assert app.selected_key is None
                print(json.dumps({'status':'PASS','first_name_and_rules_before_image_ms':round(first_ms,1),
                    'new_name_while_previous_image_blocked_ms':round(second_ms,1),
                    'checks':['immediate video crop','name/rules before image','new click bypasses slow image',
                              'stale image ignored','reference image replaces crop','cached image reused']},indent=2))
            finally:
                for event in release:
                    event.set()
                app.close()


if __name__ == '__main__':
    main()
