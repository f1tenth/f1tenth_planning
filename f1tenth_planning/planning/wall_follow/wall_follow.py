from pyclothoids import Clothoid
import cProfile
import numpy as np
import matplotlib.pyplot as plt

def sample_grid():
    x = np.linspace(0.2, 4, 10)
    y = np.linspace(-2, 2, 11)
    all_x = []
    all_y = []
    for x1 in x:
        for y1 in y:
            clothoid0 = Clothoid.G1Hermite(0, 0, 0, x1, y1, 0)
            curr_x, curr_y = clothoid0.SampleXY(100)
            all_x.extend(curr_x)
            all_y.extend(curr_y)
            # plt.scatter(curr_x, curr_y)
    # plt.show()
def test():
    for i in range(100):
        sample_grid()
if __name__ == '__main__':
    cProfile.run('test()')
    # test()