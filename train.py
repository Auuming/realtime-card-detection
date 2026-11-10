"""Generate synthetic card detection data or train a GPU YOLO detector."""
import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
from catalog import ROOT, DATA

def generate(output, count, backgrounds=None, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    records = json.loads((DATA/'cards.json').read_text('utf-8'))
    groups = {}
    for c in records.values():
        image = cv2.imread(str(ROOT/c['image']))
        if image is not None:
            groups.setdefault(c.get('oracle_id') or c['id'], []).append(image)
    if len(groups) < 2:
        raise ValueError('Download at least two distinct Oracle card identities first.')
    # Hold out identities to measure detector generalization to unseen artwork.
    identities = list(groups)
    random.shuffle(identities)
    split = max(1,int(len(identities)*.8))
    train_images = [im for key in identities[:split] for im in groups[key]]
    val_images = [im for key in identities[split:] for im in groups[key]]
    background_paths = list(backgrounds.glob('*')) if backgrounds else []
    for subset, pool, amount in [('train',train_images,count),('val',val_images,max(10,count//5))]:
        for kind in ['images','labels']:
            (output/kind/subset).mkdir(parents=True,exist_ok=True)
        for index in range(amount):
            canvas = np.full((720,960,3),rng.integers(35,180,size=3),np.uint8)
            if background_paths:
                bg = cv2.imread(str(random.choice(background_paths)))
                if bg is not None:
                    canvas = cv2.resize(bg,(960,720))
            labels = []
            # Non-overlapping grid cells keep synthetic labels unambiguous.
            cells = random.sample([(x,y) for x in [0,320,640] for y in [0,360]],random.randint(0,6))
            for x,y in cells:
                image = random.choice(pool)
                w = int(rng.integers(80,160)); h = int(w*1.4)
                theta = float(rng.uniform(-np.pi,np.pi))
                base = np.float32([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
                rotation = np.float32([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                quad = base@rotation.T + [x+160,y+180]
                quad += rng.uniform(-8,8,(4,2))
                source = np.float32([[0,0],[image.shape[1]-1,0],[image.shape[1]-1,image.shape[0]-1],[0,image.shape[0]-1]])
                matrix = cv2.getPerspectiveTransform(source,quad.astype(np.float32))
                warped = cv2.warpPerspective(image,matrix,(960,720))
                mask = cv2.warpPerspective(np.full(image.shape[:2],255,np.uint8),matrix,(960,720))
                canvas[mask>0] = warped[mask>0]
                lo,hi = quad.min(axis=0),quad.max(axis=0)
                cx,cy = (lo+hi)/2/[960,720]
                bw,bh = (hi-lo)/[960,720]
                labels.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            canvas = np.clip(canvas.astype(np.float32)*rng.uniform(.65,1.25)+rng.normal(0,2,canvas.shape),0,255).astype(np.uint8)
            if rng.random()<.3:
                canvas = cv2.GaussianBlur(canvas,(5,5),0)
            cv2.imwrite(str(output/'images'/subset/f'{index:06}.jpg'),canvas)
            (output/'labels'/subset/f'{index:06}.txt').write_text('\n'.join(labels),'utf-8')
    (output/'dataset.yaml').write_text('path: '+output.resolve().as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: card\n','utf-8')
    print('Generated',output/'dataset.yaml')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--output',type=Path,default=DATA/'synthetic')
    gen.add_argument('--count',type=int,default=2000)
    gen.add_argument('--backgrounds',type=Path,help='Folder of your own tabletop/playmat photos')
    train = sub.add_parser('train')
    train.add_argument('--data',default=str(DATA/'synthetic/dataset.yaml'))
    train.add_argument('--device',default='0')
    train.add_argument('--epochs',type=int,default=60)
    train.add_argument('--batch',type=int,default=4)
    args = p.parse_args()
    if args.command == 'generate':
        generate(args.output,args.count,args.backgrounds)
    else:
        from ultralytics import YOLO
        YOLO('yolo11n.pt').train(data=args.data,epochs=args.epochs,imgsz=960,batch=args.batch,device=args.device,project=str(ROOT/'runs'),name='card_detector')
