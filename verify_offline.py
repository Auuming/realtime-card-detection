"""Prove cached recognition/details do not require an active network connection."""
import json
from unittest.mock import patch
import cv2
import httpx
import requests
from catalog import DATA, ROOT
from recognition import CollectorRecognizer

def main():
    def blocked(*args, **kwargs):
        raise AssertionError('Network access attempted during cached offline lookup')
    with patch('urllib.request.urlopen',blocked),patch.object(requests.sessions.Session,'request',blocked),patch.object(httpx.Client,'send',blocked):
        recognizer = CollectorRecognizer(offline=True)
        card = next(iter(json.loads((DATA/'cards.json').read_text('utf-8')).values()))
        key,score,margin = recognizer.identify(cv2.imread(str(ROOT/card['image'])))
        assert key is not None
        result = recognizer.cards[key]
        assert result['oracle_id']==card['oracle_id']
        assert result['oracle_text']
        assert result['image'] and (ROOT/result['image']).exists()
        assert not result['warning']
        print('PASS: cached catalog/model, identity, image, and rules load with network calls blocked')

if __name__ == '__main__':
    main()
