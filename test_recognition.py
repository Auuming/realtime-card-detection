import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import requests
from recognition import DetailStore

CARD_ID = '7286819f-6c57-4503-898c-528786ad86e9'

class DetailTests(unittest.TestCase):
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
