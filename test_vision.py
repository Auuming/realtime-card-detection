import unittest
import cv2
import numpy as np
from vision import detect, warp, order, Recognizer

class GeometryTests(unittest.TestCase):
    def test_two_cards_and_background_rejection(self):
        im = np.zeros((700,1000,3),np.uint8)
        cv2.rectangle(im,(40,80),(240,360),(255,255,255),-1)
        cv2.rectangle(im,(500,150),(700,430),(255,255,255),-1)
        cv2.circle(im,(850,500),45,(255,255,255),-1)
        self.assertEqual(len(detect(im)),2)

    def test_warp_portrait_and_tapped(self):
        im = np.full((500,500,3),150,np.uint8)
        for pts in [[[20,20],[220,20],[220,300],[20,300]],[[20,20],[300,20],[300,220],[20,220]]]:
            result = warp(im,np.float32(pts))
            self.assertEqual(result.shape,(352,252,3))

    def test_blank_is_unknown(self):
        r = Recognizer()
        self.assertIsNone(r.identify(np.zeros((352,252,3),np.uint8))[0])

if __name__ == '__main__':
    unittest.main()
