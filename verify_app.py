"""Integration smoke test using the synthetic video, including card selection."""
import time
import tkinter as tk
from types import SimpleNamespace
from app import App
from catalog import DATA
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recognizer', choices=['orb','collectorvision'], default='collectorvision')
    args = parser.parse_args()
    root = tk.Tk()
    app = App(root, backend=args.recognizer)
    method = 'identify_selection' if hasattr(app.recognizer, 'identify_selection') else 'identify'
    original_identify = getattr(app.recognizer, method)
    calls = []
    def counted_identify(*inputs):
        calls.append(1)
        time.sleep(.3)  # Simulate slow recognition without stopping detection.
        return original_identify(*inputs)
    setattr(app.recognizer, method, counted_identify)
    try:
        app.start(str(DATA/'demo.avi'),True)
        deadline = time.monotonic()+20
        while time.monotonic()<deadline:
            root.update()
            if len(app.boxes)==3:
                break
            time.sleep(.02)
        assert len(app.boxes)==3,'Video detection did not produce three card outlines'
        assert not calls,'Recognition must not run before a click'
        pts = app.boxes[0]
        x,y = pts.mean(axis=0)
        app.click(SimpleNamespace(x=x*app.scale+app.offset[0],y=y*app.scale+app.offset[1]))
        assert 'Identifying selected card' in app.text.get('1.0','end')
        snapshot = app.detection_frame
        deadline = time.monotonic()+20
        while app.detail_pending and time.monotonic()<deadline:
            root.update()
            time.sleep(.02)
        assert app.detail_pending is None,'Click recognition did not finish'
        assert app.detection_frame is not snapshot,'Detection stopped during card identification'
        assert len(calls)==1,'Only the clicked card should be identified'
        details = app.text.get('1.0','end')
        assert app.recognizer.score_label+':' in details
        assert 'Rules text cached from Scryfall.' in details
        assert app.image_label.cget('image')
        print('PASS: outlines without recognition; click-only identification; detection continues; image and rules panel')
    finally:
        app.close()

if __name__ == '__main__':
    main()
