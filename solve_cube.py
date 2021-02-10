from rubikscube import RubiksCube
import itertools 
import random
import sys

distance, seed = 8, 30
cube = RubiksCube(distance=distance, seed=seed)
cube.solve()

