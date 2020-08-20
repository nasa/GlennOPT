import unittest
import sys, copy
sys.path.insert(0,'../../')
import os
from glennopt.doe import * 
from glennopt.nsga3 import nsga3

class TestNSGA3(unittest.TestCase):
    
    def test_nsga3_rosenbrock(self):
        # temp = generate_reference_points(4,4)
        pass

    def test_restart_opt(self):
        ns = NSGA3()
        ns.start_optimization()
    def test_nsga3_KUR(self):
        # temp = generate_reference_points(4,4)
        print(os.getcwd())
        kur_function = '../test_functions/'
        pass
if __name__ == '__main__':
    unittest.main()