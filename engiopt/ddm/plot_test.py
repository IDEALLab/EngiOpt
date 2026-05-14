import matplotlib.pyplot as plt
import matplotlib
import numpy as np



if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, subplot_kw={'projection': '3d'})

    print(matplotlib.__version__)
    N = 200
    theta = np.linspace(0, 2*np.pi, N)

    x1 = np.cos(theta)
    y1 = np.sin(theta)
    z1 = 0.1 * np.sin(6 * theta)
    # convert to float 32
    x1 = x1.astype(np.float32)

    x2 = 0.6 * np.cos(theta)
    y2 = 0.6 * np.sin(theta)
    z2 = 2  # Note that scalar values work in addition to length N arrays
    x3 = x1 + 0.5
    y3 = y1 + 0.5
    z3 = 3

    # fig = plt.figure()
    # ax = fig.add_subplots(projection='3d')
    ax[0].fill_between(x1, y1, z1, x2, y2, z2, facecolor='grey')
    ax[1].fill_between(x2, y2, z2, x3, y3, z3, )

    plt.show()