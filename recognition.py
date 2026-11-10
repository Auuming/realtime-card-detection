"""CollectorVision MTG retrieval and on-demand, cached Scryfall details."""
import json
import os
from pathlib import Path
from uuid import UUID
import cv2
import numpy as np
from PIL import Image
from catalog import ROOT, DATA, api, download

# Keep all downloaded models/catalogs in this project, including hub caches.
os.environ.setdefault('COLLECTORVISION_CACHE', str(DATA/'collectorvision'))
os.environ.setdefault('HF_HOME', str(DATA/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
CACHE = DATA/'collectorvision'/'catalogs'

class DetailStore:
    def __init__(self):
        self.cards = {}

    def load(self, match):
        identifier = str(UUID(match['id']))  # Explicitly using the Scryfall catalog.
        face_index = int(match.get('face_index', 0))
        if face_index < 0:
            raise ValueError('Invalid card face index')
        key = f'{identifier}_{face_index}'
        if key in self.cards and not self.cards[key].get('warning'):
            return key
        folder = DATA/'details'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/(identifier+'.json')
        warning = ''
        try:
            if path.exists():
                raw = json.loads(path.read_text('utf-8'))
            else:
                raw = api('https://api.scryfall.com/cards/'+identifier)
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(raw, ensure_ascii=False), 'utf-8')
                temp.replace(path)
        except (OSError, ValueError, RuntimeError) as exc:
            raw = match.get('metadata') or {}
            warning = 'Recognized, but full details are not cached and could not be downloaded: '+str(exc)
        faces = raw.get('card_faces', [])
        face = faces[face_index] if face_index < len(faces) else raw
        image_uris = face.get('image_uris') or raw.get('image_uris', {})
        image = folder/(key+'.jpg')
        if not image.exists() and image_uris.get('normal'):
            try:
                download(image_uris['normal'], image)
            except (OSError, ValueError, RuntimeError) as exc:
                warning += '\nReference image could not be downloaded: '+str(exc)
        self.cards[key] = {
            'id': identifier, 'name': face.get('name', match['name']),
            'oracle_id': raw.get('oracle_id') or match.get('identifiers', {}).get('scryfall_oracle'),
            'oracle_text': face.get('oracle_text', raw.get('oracle_text', 'Rules text unavailable in local cache.')),
            'type_line': face.get('type_line', raw.get('type_line', '')),
            'mana_cost': face.get('mana_cost', raw.get('mana_cost', '')),
            'set': raw.get('set', ''), 'collector_number': raw.get('collector_number', ''),
            'image': str(image.relative_to(ROOT)) if image.exists() else None,
            'warning': warning.strip(), 'scryfall_uri': raw.get('scryfall_uri', '')}
        return key

class CollectorRecognizer:
    score_label = 'Embedding similarity'

    def __init__(self, offline=True, min_score=.78, min_margin=.025):
        import collector_vision as cvg
        self.cvg = cvg
        self.min_score, self.min_margin = min_score, min_margin
        self.catalog = cvg.Catalog.load('mtg', source='scryfall', cache_dir=CACHE, offline=offline)
        self.embedder = self.catalog.embedder
        self.corners = cvg.NeuralCornerDetector(provider='cpu')
        self.store = DetailStore()
        self.cards = self.store.cards
        self.keys = self.catalog.card_ids
        self.description = f'CollectorVision | {len(self.keys):,} catalog faces'

    def identify_selection(self, frame, points):
        from vision import order
        # Preserve original pixels until the model's native 448x448 warp.
        points = order(points)
        lo = np.maximum(np.floor(points.min(axis=0)).astype(int)-12, 0)
        hi = np.minimum(np.ceil(points.max(axis=0)).astype(int)+12, [frame.shape[1],frame.shape[0]])
        region = frame[lo[1]:hi[1],lo[0]:hi[0]]
        detection = self.corners.detect(region)
        if detection.card_present and detection.corners is not None and np.isfinite(detection.corners).all():
            crop = detection.dewarp(region)
        else:
            normalized = points/np.float32([frame.shape[1], frame.shape[0]])
            crop = self.cvg.DetectionResult(normalized.astype(np.float32)).dewarp(frame)
        return self._identify_pil(crop)

    def identify(self, crop):
        return self._identify_pil(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    def _identify_pil(self, crop):
        if np.asarray(crop.convert('L')).std() < 8:
            return None, 0., 0.
        # Upstream notes that embeddings are sensitive to upside-down cards.
        variants = [crop, crop.transpose(Image.Transpose.ROTATE_180)]
        groups = {}
        for embedding in self.embedder.embed(variants):
            for match in self.catalog.search_records(embedding, top_k=20):
                identity = match['identifiers'].get('scryfall_oracle') or match['id']
                if identity not in groups or match['score'] > groups[identity]['score']:
                    groups[identity] = match
        ranked = sorted(groups.values(), key=lambda m: -m['score'])
        if not ranked:
            return None, 0., 0.
        best = ranked[0]
        score = float(best['score'])
        margin = score-float(ranked[1]['score']) if len(ranked)>1 else score
        if score < self.min_score or margin < self.min_margin:
            return None, score, margin
        return self.store.load(best), score, margin

def create_recognizer(backend='collectorvision', **kwargs):
    if backend == 'orb':
        from vision import Recognizer
        result = Recognizer()
        result.score_label = 'Geometric inliers'
        result.description = f'ORB baseline | {len(result.keys)} local reference faces'
        return result
    return CollectorRecognizer(**kwargs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download/update the pretrained MTG catalog. No YOLO training.')
    parser.add_argument('--prepare', action='store_true', required=True)
    args = parser.parse_args()
    recognizer = CollectorRecognizer(offline=False)
    print('Ready:', recognizer.description, flush=True)
    print('Embedding model:', recognizer.catalog.embedding_model)
