import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from PIL import Image
import requests
from recognition import DetailStore

CARD_ID = '7286819f-6c57-4503-898c-528786ad86e9'

class DetailTests(unittest.TestCase):
    def test_cache_only_returns_name_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('recognition.DATA', root), patch('recognition.ROOT', root), patch('recognition.api') as api, patch('recognition.download') as download:
                store = DetailStore()
                key = store.load({'id': CARD_ID, 'name': 'Instant name'}, cache_only=True)
                self.assertEqual(store.cards[key]['name'], 'Instant name')
                self.assertEqual(store.cards[key]['details_state'], 'pending')
                api.assert_not_called()
                download.assert_not_called()

    def test_metadata_is_published_while_image_is_still_loading(self):
        entered, release = Event(), Event()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def slow_image(url, path, **kwargs):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test image timed out')
                Image.new('RGB', (20, 28), 'red').save(path)
            raw = {'name': 'Ready name', 'oracle_text': 'Ready rules', 'image_uris': {'normal': 'https://example.test/card.jpg'}}
            with patch('recognition.DATA', root), patch('recognition.ROOT', root), patch('recognition.api', return_value=raw), patch('recognition.download', side_effect=slow_image):
                store = DetailStore()
                key = store.load({'id': CARD_ID, 'name': 'Ready name'}, cache_only=True)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(store.refresh, key)
                    try:
                        self.assertTrue(entered.wait(2))
                        self.assertFalse(future.done())
                        self.assertEqual(store.cards[key]['oracle_text'], 'Ready rules')
                        self.assertEqual(store.cards[key]['image_state'], 'pending')
                    finally:
                        release.set()
                    self.assertEqual(future.result(timeout=3), key)
                self.assertEqual(store.cards[key]['image_state'], 'ready')

    def test_failed_image_uses_cooldown_but_manual_retry_can_succeed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = {'name': 'Known', 'image_uris': {'normal': 'https://example.test/card.jpg'}}
            with patch('recognition.DATA', root), patch('recognition.ROOT', root), patch('recognition.api', return_value=raw) as api, patch('recognition.download', side_effect=requests.ConnectionError('offline')) as download:
                store = DetailStore()
                key = store.load({'id': CARD_ID, 'name': 'Known'})
                self.assertEqual(store.cards[key]['image_state'], 'error')
                store.refresh(key)
                self.assertEqual(download.call_count, 1)
                def successful_image(url, path, **kwargs):
                    Image.new('RGB', (20, 28), 'blue').save(path)
                download.side_effect = successful_image
                store.refresh(key, force=True)
                self.assertEqual(download.call_count, 2)
                self.assertEqual(store.cards[key]['image_state'], 'ready')
                store.refresh(key)
                self.assertEqual(download.call_count, 2)
                self.assertEqual(api.call_count, 1)

    def test_cached_image_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'details').mkdir()
            (root/'details'/(CARD_ID+'.json')).write_text(json.dumps({'name': 'Cached', 'image_uris': {'normal': 'https://example.test/card.jpg'}}))
            Image.new('RGB', (20,28)).save(root/'details'/(CARD_ID+'_0.jpg'))
            with patch('recognition.DATA',root), patch('recognition.ROOT',root), patch('recognition.api') as api, patch('recognition.download') as download:
                store = DetailStore()
                key = store.load({'id': CARD_ID, 'name': 'Cached'}, cache_only=True)
                store.refresh(key)
                self.assertEqual(store.cards[key]['image_state'], 'ready')
                api.assert_not_called()
                download.assert_not_called()

    def test_corrupt_image_cache_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'details').mkdir()
            (root/'details'/(CARD_ID+'_0.jpg')).write_bytes(b'not an image')
            def valid_image(url, path, **kwargs):
                Image.new('RGB', (20,28)).save(path)
            with patch('recognition.DATA',root), patch('recognition.ROOT',root), patch('recognition.api',return_value={'name':'Card','image_uris':{'normal':'https://example.test/image'}}), patch('recognition.download',side_effect=valid_image):
                store = DetailStore()
                key = store.load({'id':CARD_ID,'name':'Card'})
                self.assertEqual(store.cards[key]['image_state'], 'ready')

    def test_cached_double_face_uses_matched_face_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'details').mkdir()
            raw = {'id':CARD_ID,'oracle_id':'oracle-example','set':'abc', 'card_faces':[
                {'name':'Front','oracle_text':'Front rules'}, {'name':'Back','oracle_text':'Back rules'}]}
            (root/'details'/(CARD_ID+'.json')).write_text(json.dumps(raw),'utf-8')
            with patch('recognition.DATA',root),patch('recognition.ROOT',root),patch('recognition.api') as request:
                store = DetailStore()
                key = store.load({'id':CARD_ID,'face_index':1,'name':'Front // Back'})
                self.assertEqual(store.cards[key]['name'],'Back')
                self.assertEqual(store.cards[key]['oracle_text'],'Back rules')
                request.assert_not_called()

    def test_network_failure_keeps_recognized_name_and_reports_missing_details(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('recognition.DATA',root),patch('recognition.ROOT',root),patch('recognition.api',side_effect=requests.ConnectionError('offline')):
                store = DetailStore()
                key = store.load({'id':CARD_ID,'name':'Recognized card','identifiers':{}})
                self.assertEqual(store.cards[key]['name'],'Recognized card')
                self.assertIn('could not be downloaded',store.cards[key]['warning'])
                self.assertIsNone(store.cards[key]['image'])

if __name__ == '__main__':
    unittest.main()
