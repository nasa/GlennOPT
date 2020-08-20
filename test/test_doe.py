import unittest
import sys, copy
sys.path.insert(0,'../../')
from glennopt.doe import * 

class TestDOE(unittest.TestCase):
    def test_generate_reference_points(self):
        temp = generate_reference_points(4,4)
        print(temp)
        pass


if __name__ == '__main__':
    unittest.main()