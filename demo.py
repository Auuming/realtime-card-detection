"""Create a deterministic synthetic multi-card video and run a recognition smoke test."""
import json
import time
import cv2
import numpy as np
from catalog import DATA, ROOT
from vision import Recognizer, detect, warp

def scene():
    records = json.loads((DATA/'cards.json').read_text('utf-8'))
    canvas = np.full((720,1100,3),(46,72,56),np.uint8)
    expected = []
    for (key,card),(x,y) in zip(records.items(),[(70,100),(420,160),(760,80)]):
        image = cv2.resize(cv2.imread(str(ROOT/card['image'])),(210,294))
        canvas[y:y+294,x:x+210] = image
        expected.append(key)
    return canvas,expected

def main():
    canvas,expected = scene()
    path = DATA/'demo.avi'
    video = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(1100,720))
    if not video.isOpened():
        raise RuntimeError('MJPG video writer unavailable')
    for _ in range(100):
        video.write(canvas)
    video.release()
    recognizer = Recognizer()
    t = time.perf_counter()
    found = []
    for pts in detect(canvas):
        key,score,margin = recognizer.identify(warp(canvas,pts))
        if key:
            found.append(key)
        cv2.polylines(canvas,[pts.astype(int)],True,(100,255,100),2)
    cv2.imwrite(str(DATA/'demo-result.jpg'),canvas)
    print(f'Synthetic test: expected {len(expected)}, recognized {len(found)}, {(time.perf_counter()-t)*1000:.0f} ms')
    assert set(found)==set(expected),(found,expected)
    print('PASS. Open data/demo.avi in app.py. Synthetic accuracy does not establish real gameplay accuracy.')

if __name__ == '__main__':
    main()
