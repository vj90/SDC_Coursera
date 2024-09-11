from path_optimizer import *
import numpy as np
import matplotlib.pyplot as plt

circle_offsets = [-1.0, 1.0, 3.0]
circle_locations = np.zeros((len(circle_offsets), 2))
print(circle_locations)

circle_locations[:, 0] = [0.4 + 0.1*k for k in circle_offsets]
print(circle_locations)