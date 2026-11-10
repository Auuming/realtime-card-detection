"""Scryfall metadata and reference images, downloaded once for offline recognition."""
import argparse
import gzip
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
HEADERS = {'User-Agent': 'TableLens-StudentCV/0.1', 'Accept': 'application/json;q=0.9,*/*;q=0.8'}

def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.part')
    with requests.get(url, headers=HEADERS, stream=True, timeout=(10, 90)) as r:
        r.raise_for_status()
        with temp.open('wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    temp.replace(target)

def api(url):
    time.sleep(0.15)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()  # Stop on rate limits; never hammer the service.
    return r.json()

def ingest(cards, limit=0):
    DATA.mkdir(exist_ok=True)
    path = DATA / 'cards.json'
    records = json.loads(path.read_text('utf-8')) if path.exists() else {}
    count = 0
    try:
        for card in cards:
            faces = card.get('card_faces', [])
            image_faces = faces if not card.get('image_uris') else [card]
            for n, face in enumerate(image_faces):
                uri = face.get('image_uris', {}).get('normal')
                if not uri:
                    continue
                key = card['id'] + (f'_{n}' if len(image_faces) > 1 else '')
                image = DATA / 'images' / (key + '.jpg')
                if not image.exists():
                    download(uri, image)
                records[key] = {'id': card['id'], 'name': face.get('name', card['name']),
                    'oracle_id': card.get('oracle_id'), 'oracle_text': face.get('oracle_text', card.get('oracle_text', '')),
                    'type_line': face.get('type_line', card.get('type_line', '')),
                    'mana_cost': face.get('mana_cost', card.get('mana_cost', '')),
                    'set': card.get('set'), 'collector_number': card.get('collector_number'),
                    'image': str(image.relative_to(ROOT)), 'scryfall_uri': card.get('scryfall_uri')}
            count += 1
            if count % 25 == 0:
                print(f'Processed {count} cards; {len(records)} reference faces', flush=True)
            if limit and count >= limit:
                break
    finally:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), 'utf-8')
        temporary.replace(path)
    print(f'Saved {len(records)} reference faces to {path}')

def read_bulk(path):
    with path.open('rb') as probe:
        compressed = probe.read(2) == b'\x1f\x8b'
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as f:
        first = f.read(1)
        while first.isspace():
            first = f.read(1)
        f.seek(0)
        if first == b'[':
            import ijson
            yield from ijson.items(f, 'item')
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bulk', type=Path, help='Existing Scryfall JSON array or JSONL, optionally gzip')
    p.add_argument('--download-bulk', action='store_true', help='Download default_cards metadata (large)')
    p.add_argument('--limit', type=int, default=0, help='0 = all; no banned-card filtering')
    args = p.parse_args()
    if args.download_bulk:
        info = api('https://api.scryfall.com/bulk-data/default_cards')
        uri = info.get('jsonl_download_uri') or info.get('download_uri')
        if not uri:
            raise RuntimeError('Bulk response has no supported download URI')
        args.bulk = DATA / 'bulk.download'
        download(uri, args.bulk)
    if args.bulk:
        ingest(read_bulk(args.bulk), args.limit)
    else:
        names = ['Lightning Bolt', 'Llanowar Elves', 'Counterspell', 'Sol Ring', 'Island', 'Forest']
        from urllib.parse import quote
        ingest((api('https://api.scryfall.com/cards/named?exact=' + quote(n)) for n in names), args.limit)

if __name__ == '__main__':
    main()
