from pyclothoids import Clothoid
from f1tenth_planning.utils.utils import sample_traj
# import cProfile
import numpy as np
import matplotlib.pyplot as plt

def sample_grid():
    x = np.linspace(0.2, 4, 10)
    y = np.linspace(-2, 2, 11)
    # all_x = []
    # all_y = []
    for x1 in x:
        for y1 in y:
            clothoid0 = Clothoid.G1Hermite(0, 0, 0, x1, y1, 0)
            traj = sample_traj(clothoid0, 100)
            # traj = clothoid0.SampleXY(100)
            # all_x.extend(curr_x)
            # all_y.extend(curr_y)
            # plt.scatter(curr_x, curr_y)
    # plt.show()

def test_cont():
    clothoid = Clothoid.G1Hermite(0, 0, 0, 1, 1, 0)
    traj = sample_traj(clothoid, 100)
    x, y = clothoid.SampleXY(100)
    thetas = np.linspace(clothoid.ThetaStart, clothoid.ThetaEnd, 100)
    plt.plot(traj[:, 2])
    plt.plot(thetas)
    plt.show()
    plt.plot(traj[:, 0], traj[:, 1])
    plt.plot(x, y)
    plt.show()

def test():
    for i in range(100):
        sample_grid()
if __name__ == '__main__':
    # cProfile.run('test_cont()')
    # test()
    clothoid = Clothoid.G1Hermite(0, 0, 0, 1, 1, 0)
    c = Clothoid.G1Hermite(0, 0, 0, 1, 1, 0)
    traj = sample_traj(clothoid, 100)
    x, y = clothoid.SampleXY(100)
    thetas = np.linspace(clothoid.ThetaStart, clothoid.ThetaEnd, 100)
    plt.plot(traj[:, 2])
    plt.plot(thetas)
    plt.show()
    plt.scatter(traj[:, 0], traj[:, 1], marker='x')
    plt.scatter(x, y, marker='*')
    plt.show()