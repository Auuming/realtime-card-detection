"""Exercise real pretrained MTG retrieval, rotation, rejection, and detail caching."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from catalog import DATA, ROOT, api
from recognition import CollectorRecognizer

def main():
    recognizer = CollectorRecognizer()
    results = []
    references = json.loads((DATA/'cards.json').read_text('utf-8'))
    for key, card in references.items():
        image = cv2.imread(str(ROOT/card['image']))
        for rotation in [0,180]:
            query = image if not rotation else cv2.rotate(image,cv2.ROTATE_180)
            started = time.perf_counter()
            found, score, margin = recognizer.identify(query)
            identified = recognizer.cards[found] if found else {}
            correct = identified.get('oracle_id') == card['oracle_id']
            results.append({'case': card['name'], 'rotation':rotation, 'correct_oracle':correct,
                            'score':score, 'margin':margin, 'ms':round((time.perf_counter()-started)*1000),
                            'identified': identified.get('name'), 'warning':identified.get('warning')})
            print(results[-1],flush=True)
    # Upstream's example photographs are external smoke fixtures, not our held-out test set.
    samples = DATA/'vendor'/'CollectorVision-8b295ab80a997766afc00c640a616c1f2716df21'/'examples'/'images'
    for path in samples.glob('*.jpg'):
        image = cv2.imread(str(path))
        detection = recognizer.corners.detect(image)
        expected_id = path.name[:36]
        expected = api('https://api.scryfall.com/cards/'+expected_id)
        found,score,margin = recognizer._identify_pil(detection.dewarp(image)) if detection.card_present else (None,0.,0.)
        identified = recognizer.cards[found] if found else {}
        results.append({'case':path.name,'correct_oracle':identified.get('oracle_id')==expected.get('oracle_id'),
                        'identified':identified.get('name'),'score':score,'margin':margin})
        print(results[-1],flush=True)
    for name,image in [('blank',np.zeros((352,252,3),np.uint8)),
                       ('noise',np.random.default_rng(42).integers(0,256,(352,252,3),dtype=np.uint8))]:
        found,score,margin = recognizer.identify(image)
        results.append({'case':name,'correct_rejection':found is None,'score':score,'margin':margin})
        print(results[-1],flush=True)
    report = {'catalog_faces':len(recognizer.keys),'catalog_version':recognizer.catalog.version,
              'embedding_model':recognizer.catalog.embedding_model,'results':results,
              'note':'Smoke tests only. References and upstream examples are not independent gameplay evaluation.'}
    (DATA/'collector-verification.json').write_text(json.dumps(report,indent=2),'utf-8')
    assert all(r.get('correct_oracle',r.get('correct_rejection',False)) for r in results),'Some smoke cases failed; inspect report'
    print('PASS: CollectorVision smoke checks',flush=True)

if __name__ == '__main__':
    main()
