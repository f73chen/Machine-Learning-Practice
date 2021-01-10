from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

def get_q_colour(value, vals):
    if value == max(vals): return "green", 1.0
    else: return "red", 0.3
    
fig = plt.figure(figsize = (12, 9))

for i in range(0, 20000, 100):
    print(i)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    qTable = np.load(f"qtables\{i}-qtable.npy")

    for x, xVals in enumerate(qTable):
        for y, yVals in enumerate(xVals):
            ax1.scatter(x, y, c = get_q_colour(yVals[0], yVals)[0], marker = "o", alpha = get_q_colour(yVals[0], yVals)[1])
            ax2.scatter(x, y, c = get_q_colour(yVals[1], yVals)[0], marker = "o", alpha = get_q_colour(yVals[1], yVals)[1])
            ax3.scatter(x, y, c = get_q_colour(yVals[2], yVals)[0], marker = "o", alpha = get_q_colour(yVals[2], yVals)[1])
            
            ax1.set_ylabel("Action 0")
            ax2.set_ylabel("Action 1")
            ax3.set_ylabel("Action 2")
        
    #plt.show()
    plt.savefig(f"qtable_charts\{i}.png")
    plt.clf()