import unittest
from src.features.dimensions import parse_text_dimensions

class TestDimensionFeatures(unittest.TestCase):
    def test_inch_parsing(self):
        res = parse_text_dimensions("Standard screen size 15.6 inch display")
        self.assertAlmostEqual(res['dim_max'], 15.6 * 100.0)

    def test_cm_parsing(self):
        res = parse_text_dimensions("Beautiful custom-made canvas height 30 cm")
        self.assertAlmostEqual(res['dim_max'], (30.0 / 2.54) * 100.0)

    def test_false_positive_prevention(self):
        res = parse_text_dimensions("High capacity 5000 mAh battery pack with 20W power adapter")
        self.assertEqual(res['dim_count'], 0)
        self.assertAlmostEqual(res['dim_max'], 0.0)

if __name__ == '__main__':
    unittest.main()
