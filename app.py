"""TableLens desktop scanner. All camera/video processing stays local."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from catalog import ROOT
from vision import detect, warp
from recognition import create_recognizer

class App:
    def __init__(self, root, model=None, backend='collectorvision', recognizer=None):
        self.root = root
        root.title('TableLens | Magic: The Gathering')
        root.geometry('1280x850')
        self.recognizer = recognizer or create_recognizer(backend)
        self.model = None
        if model:
            from ultralytics import YOLO
            self.model = YOLO(model)
        self.capture = None
        self.is_file = False
        self.paused = False
        self.generation = 0
        self.pending = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.detail_pool = ThreadPoolExecutor(max_workers=1)
        self.detail_pending = None
        self.detection_frame = None
        self.frame = None
        self.boxes = []
        self.last_submit = 0
        self.last_read = 0
        self.interval = 1/30
        bar = ttk.Frame(root, padding=10)
        bar.pack(fill='x')
        ttk.Label(bar, text='TABLELENS', font=('Segoe UI', 18, 'bold')).pack(side='left', padx=10)
        self.source = tk.StringVar(value='0')
        ttk.Entry(bar, textvariable=self.source, width=26).pack(side='left')
        ttk.Button(bar, text='Camera / stream', command=self.open_source).pack(side='left', padx=4)
        ttk.Button(bar, text='Open video', command=self.open_file).pack(side='left', padx=4)
        ttk.Button(bar, text='Pause / resume', command=self.toggle).pack(side='left', padx=4)
        body = ttk.Frame(root)
        body.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(body, bg='#101820', highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<Button-1>', self.click)
        side = ttk.Frame(body, width=310, padding=15)
        side.pack(side='right', fill='y')
        side.pack_propagate(False)
        ttk.Label(side, text='CARD DETAILS', font=('Segoe UI', 16, 'bold')).pack(anchor='w')
        self.image_label = ttk.Label(side)
        self.image_label.pack(pady=10)
        self.text = tk.Text(side, wrap='word', font=('Segoe UI', 11), height=20, relief='flat')
        self.text.pack(fill='both', expand=True)
        self.set_text('Open a webcam, video file, or direct RTSP/HTTP video stream.\n\nClick a detected card to inspect its reference image and rules.\n\n'+self.recognizer.description+'\n\nFirst lookup of a card needs internet for its image and rules; later lookups use the local cache.')
        self.status = tk.StringVar(value='Ready | '+ ('YOLO detector' if model else 'Contour baseline'))
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(20, self.tick)

    def set_text(self, text):
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', text)
        self.text.configure(state='disabled')

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('All', '*.*')])
        if path:
            self.start(path, True)

    def open_source(self):
        source = self.source.get().strip()
        self.start(int(source) if source.isdigit() else source, False)

    def start(self, source, is_file):
        if self.capture:
            self.capture.release()
        self.generation += 1
        self.boxes = []
        self.frame = None
        self.detection_frame = None
        if self.detail_pending:
            self.detail_pending.cancel()
            self.detail_pending = None
        self.image_label.configure(image='')
        self.set_text('Click a card border to identify it and load its details.')
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror('Source unavailable', 'Could not open source. Check path, camera permissions, or direct stream URL.')
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.interval = 1/fps if is_file and 1 < fps < 240 else 1/30
        self.is_file, self.paused = is_file, False
        self.status.set('Running | click a card for details')

    def toggle(self):
        self.paused = not self.paused

    def analyze(self, frame, generation):
        started = time.perf_counter()
        if self.model:
            prediction = self.model.predict(frame, verbose=False, conf=.35)[0]
            points = [np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]) for x1,y1,x2,y2 in prediction.boxes.xyxy.cpu().numpy()]
            # Refine each axis-aligned YOLO region with quadrilateral geometry.
            refined = []
            for pts in points:
                x1,y1 = np.maximum(pts[0].astype(int)-8, 0)
                x2,y2 = np.minimum(pts[2].astype(int)+8, [frame.shape[1],frame.shape[0]])
                local = detect(frame[y1:y2,x1:x2], min_area=600)
                refined.append(local[0]+[x1,y1] if local else pts)
            points = refined
        else:
            points = detect(frame)
        # Detection never performs catalog recognition. Keep this fast even
        # while a separate worker is identifying a clicked card.
        return generation, points, frame, (time.perf_counter()-started)*1000

    def tick(self):
        if self.pending and self.pending.done():
            try:
                generation, boxes, snapshot, ms = self.pending.result()
                if generation == self.generation:
                    self.boxes = boxes
                    self.detection_frame = snapshot
                    self.status.set(f'{len(boxes)} card regions | detection {ms:.0f} ms | click a border for details')
            except Exception as e:
                self.status.set('Analysis error: '+str(e))
            self.pending = None
        if self.detail_pending and self.detail_pending.done():
            try:
                key, count, margin = self.detail_pending.result()
                self.show_detail(key, count)
            except Exception as e:
                self.set_text('Could not identify the selected card: '+str(e))
            self.detail_pending = None
        now = time.perf_counter()
        if self.capture and not self.paused and now-self.last_read >= self.interval:
            ok, frame = self.capture.read()
            self.last_read = now
            if ok:
                scale = min(1., 1100/frame.shape[1])
                self.frame = cv2.resize(frame, None, fx=scale, fy=scale)
                if self.pending is None and now-self.last_submit >= .05:
                    self.pending = self.pool.submit(self.analyze, self.frame.copy(), self.generation)
                    self.last_submit = now
            else:
                self.capture.release()
                self.capture = None
                self.status.set('Video ended' if self.is_file else 'Source disconnected')
        if self.frame is not None:
            frame = self.frame.copy()
            for pts in self.boxes:
                cv2.polylines(frame, [pts.astype(int)], True, (210,100,240), 2)
            w,h = max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
            self.scale = min(w/frame.shape[1],h/frame.shape[0])
            im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            im = im.resize((max(1,int(frame.shape[1]*self.scale)),max(1,int(frame.shape[0]*self.scale))))
            self.offset = ((w-im.width)//2,(h-im.height)//2)
            self.photo = ImageTk.PhotoImage(im)
            self.canvas.delete('all')
            self.canvas.create_image(*self.offset, image=self.photo, anchor='nw')
        self.root.after(20,self.tick)

    def click(self,event):
        if self.detection_frame is None:
            return
        point = ((event.x-self.offset[0])/self.scale,(event.y-self.offset[1])/self.scale)
        for pts in self.boxes:
            if cv2.pointPolygonTest(pts.astype(np.float32),point,False) < 0:
                continue
            # Use the exact frame that produced this polygon, so movement
            # during recognition cannot change the requested card crop.
            if self.detail_pending:
                self.detail_pending.cancel()
            self.image_label.configure(image='')
            self.set_text('Identifying selected card…\n\nVideo and card borders continue updating.')
            if hasattr(self.recognizer, 'identify_selection'):
                self.detail_pending = self.detail_pool.submit(self.recognizer.identify_selection, self.detection_frame.copy(), pts.copy())
            else:
                self.detail_pending = self.detail_pool.submit(self.recognizer.identify, warp(self.detection_frame, pts))
            break

    def show_detail(self, key, count):
        if not key:
            self.image_label.configure(image='')
            self.set_text('Unknown card\n\nMove closer, reduce glare, and keep the full face visible. This artwork may be missing from the recognition catalog.\n\nA weak match is not treated as an identity.')
            return
        card = self.recognizer.cards[key]
        if card.get('image'):
            with Image.open(ROOT/card['image']) as source:
                im = source.copy()
            im.thumbnail((260,350))
            self.detail_photo = ImageTk.PhotoImage(im)
            self.image_label.configure(image=self.detail_photo)
        else:
            self.image_label.configure(image='')
        score = f'{count:.3f}' if isinstance(count, float) else str(count)
        warning = card.get('warning', '')
        self.set_text(f"{card['name']}\n{card['mana_cost']}\n{card['type_line']}\n\n{card['oracle_text']}\n\nSet: {card['set']} #{card['collector_number']}\n{self.recognizer.score_label}: {score}\n\nRules text cached from Scryfall.\n{warning}")

    def close(self):
        if self.capture:
            self.capture.release()
        # Finish the worker before Tk objects are destroyed. A bound analysis
        # task retains this App; allowing it to outlive Tk can finalize GUI
        # variables on the worker thread and crash Tcl during shutdown.
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.detail_pool.shutdown(wait=True,cancel_futures=True)
        self.pending = None
        self.detail_pending = None
        self.root.destroy()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', help='Optional trained YOLO detection weights')
    p.add_argument('--recognizer', choices=['collectorvision','orb'], default='collectorvision')
    args = p.parse_args()
    root = tk.Tk()
    try:
        App(root,args.model,args.recognizer)
    except Exception as exc:
        messagebox.showerror('Scanner setup', str(exc)+'\n\nRun .venv\\Scripts\\python.exe recognition.py --prepare first. Use --recognizer orb for the starter baseline.')
        root.destroy()
        raise
    root.mainloop()
