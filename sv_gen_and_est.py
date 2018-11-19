import numpy as np
import matplotlib.pyplot as plt

# ============================================= #
# ===== Functions ============================= #
# ============================================= #

def plotting(x, y1, y2, y1_lab, y2_lab, x_lab, y_lab, title = ""):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = y1_lab)
    ax.plot(x, y2, label = y2_lab, marker = 'o', alpha = 0.5, markerfacecolor="None")
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    fig.tight_layout()
    return plt.show()

# ============================================= #
# ===== Models ================================ #
# ============================================= #

# --------------------------------------------- #
" Markov switching SV model "
mat = 250

" Theta parameters"
s_h = 4.0 
s_l = 0.1
p11 = 0.95
p22 = 0.90

z = np.random.normal(0,1, size = mat)
u = np.random.uniform(0, 1, size = mat)

state_ms = np.repeat(0, mat)
for t in range(1,mat):
    if state_ms[t-1] == 0:
        state_ms[t] = (u[t] < p11) * 0 + (u[t] > p11) * 1
    else:
        state_ms[t] = (u[t] < p22) * 1 + (u[t] > p11) * 0

returns_ms = (state_ms == 1) * s_h * z + (state_ms == 0) * s_l * z

plot_time = np.arange(0,mat)
plotting(plot_time, returns_ms, state_ms, 'Returns', 'State',
         'Time', 'Returns', 'Markov Switching SV Model')

# We have generated a series of returns in returns_ms
# which we now want to estimate. The idea is, that the
# algorithm works correctly, if it returns the same
# parameters we used for the simulation.

# ============================================= #
# ===== Mixture model only ==================== #
# ============================================= #

" We need the density function "
def md(vol, returns):
    1 / (2 * np.pi * vol ** 2) * np.exp(-0.5 * returns ** 2 / vol ** 2)

" We need smoothed probabilities "
def smoothed_probs(theta, returns):
    end = len(returns)
    p1   = np.zeros(end)
    for t in np.arange(0,end):
        md1 = md(theta[1], returns[t])
        md2 = md(theta[2], returns[t])
        p11 = theta[2]
        p22 = theta[3]
        p1[t] = p11 * md1 / (p11 * md1 + p22 * md2)
    return p1

y = returns_ms
theta = (2, 1, 0.5, 0.5) # Initial guess of: s_h, s_l, p11, p22



# ============================================= #
# ===== Forward - Backward Algorithm ========== #
# ============================================= #

def a_j(vol, returns):
