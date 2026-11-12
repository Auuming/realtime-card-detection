"""CollectorVision MTG retrieval and on-demand, cached Scryfall details."""
import json
import os
import time
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
    # Interactive downloads have shorter timeouts than bulk/setup downloads.
    timeout = (3, 6)
    retry_delay = 60

    def __init__(self):
        self.cards = {}
        self.matches = {}
        self.retry_after = {}

    def _image(self, path):
        if not path.exists():
            return None
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError):
            path.unlink(missing_ok=True)
            return None
        return str(path.relative_to(ROOT))

    def load(self, match, cache_only=False, force=False):
        """Publish local results immediately; optionally fetch metadata/image.

        Each published card is a fresh dict, so the UI can detect updates while
        the image request is still running. The app's asset worker performs network I/O.
        """
        identifier = str(UUID(match['id']))
        face_index = int(match.get('face_index', 0))
        if face_index < 0:
            raise ValueError('Invalid card face index')
        key = f'{identifier}_{face_index}'
        self.matches[key] = match
        previous = self.cards.get(key)
        if previous and not force:
            if cache_only or self.retry_after.get(key, 0) > time.monotonic():
                return key
            if previous['details_state'] == 'ready' and previous['image_state'] in ('ready', 'absent'):
                return key
        folder = DATA/'details'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/(identifier+'.json')
        image = folder/(key+'.jpg')
        warning = ''
        raw = match.get('metadata') or {}
        metadata_ready = False
        if path.exists():
            try:
                raw = json.loads(path.read_text('utf-8'))
                if not isinstance(raw, dict):
                    raise ValueError('Invalid metadata cache')
                metadata_ready = True
            except (OSError, ValueError):
                # Replace a corrupt cache on the next successful online fetch.
                raw = match.get('metadata') or {}
        if not cache_only and not metadata_ready:
            try:
                raw = api('https://api.scryfall.com/cards/'+identifier, timeout=self.timeout)
                if not isinstance(raw, dict):
                    raise ValueError('Invalid card metadata')
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(raw, ensure_ascii=False), 'utf-8')
                temp.replace(path)
                metadata_ready = True
            except (OSError, ValueError, RuntimeError) as exc:
                raw = match.get('metadata') or {}
                warning = 'Recognized, but full details are not cached and could not be downloaded: '+str(exc)
        faces = raw.get('card_faces', [])
        face = faces[face_index] if face_index < len(faces) else raw
        image_uris = face.get('image_uris') or raw.get('image_uris', {})
        cached_image = self._image(image)
        card = {
            'id': identifier, 'name': face.get('name', match['name']),
            'oracle_id': raw.get('oracle_id') or match.get('identifiers', {}).get('scryfall_oracle'),
            'oracle_text': face.get('oracle_text', raw.get('oracle_text', 'Rules text unavailable in local cache.')),
            'type_line': face.get('type_line', raw.get('type_line', '')),
            'mana_cost': face.get('mana_cost', raw.get('mana_cost', '')),
            'set': raw.get('set', ''), 'collector_number': raw.get('collector_number', ''),
            'image': cached_image, 'warning': warning,
            'details_state': 'ready' if metadata_ready else ('pending' if cache_only else 'error'),
            'image_state': 'ready' if cached_image else ('pending' if image_uris.get('normal') or not metadata_ready else 'absent'),
            'scryfall_uri': raw.get('scryfall_uri', '')}
        # Rules/name become visible before waiting for an uncached image.
        self.cards[key] = card
        if not cache_only:
            if not cached_image and image_uris.get('normal'):
                try:
                    download(image_uris['normal'], image, timeout=self.timeout)
                    cached_image = self._image(image)
                    if not cached_image:
                        raise ValueError('Downloaded reference image is invalid')
                except (OSError, ValueError, RuntimeError) as exc:
                    warning += '\nReference image could not be downloaded: '+str(exc)
                    image.with_suffix(image.suffix+'.part').unlink(missing_ok=True)
            elif not cached_image and not metadata_ready:
                warning += '\nReference image unavailable until card details can be downloaded.'
            card = dict(card, image=cached_image, warning=warning.strip(),
                        image_state='ready' if cached_image else ('error' if warning else 'absent'))
            self.cards[key] = card
            if warning:
                self.retry_after[key] = time.monotonic()+self.retry_delay
            else:
                self.retry_after.pop(key, None)
        return key

    def refresh(self, key, force=False):
        return self.load(self.matches[key], force=force)

class CollectorRecognizer:
    score_label = 'Embedding similarity'

    def __init__(self, offline=True, min_score=.78, min_margin=.025, defer_details=False):
        import collector_vision as cvg
        self.cvg = cvg
        self.min_score, self.min_margin = min_score, min_margin
        self.defer_details = defer_details
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
        return self._match_crop(crop)

    def identify(self, crop):
        return self._identify_pil(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    def _identify_pil(self, crop):
        return self._match_crop(crop)[:3]

    def _match_crop(self, crop):
        if np.asarray(crop.convert('L')).std() < 8:
            return None, 0., 0., 'The straightened crop had too little contrast for a catalog search.'
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
            return None, 0., 0., 'The catalog search returned no candidates.'
        best = ranked[0]
        score = float(best['score'])
        margin = score-float(ranked[1]['score']) if len(ranked)>1 else score
        # Temporarily accept the highest-scoring candidate for inspection.
        # if score < self.min_score:
        #     return None, score, margin, 'The best match scored below the similarity cutoff.'
        # if margin < self.min_margin:
        #     reason = f'The score gap was {margin:.3f} and the required gap is {self.min_margin:.3f}.'
        #     return None, score, margin, reason
        key = self.store.load(best, cache_only=True) if self.defer_details else self.store.load(best)
        return key, score, margin, None

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
