from matplotlib import pyplot as plt   # Default plotting
plt.style.use('ggplot')                # Use grid from likes of R

def plotUno(x, y, yLab = 'Return process', xLab = 'Time', title = ''):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y, label = yLab)
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def plotDuo(x, y1, y2, yLab1, yLab2, xLab, yLab, title = ""):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = yLab1)
    ax.plot(x, y2, label = yLab2)
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def plotSVModel(x, y1, y2, yLab1, yLab2, yLab, xLab, title = ''):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = yLab1)
    ax.plot(x, y2, label = yLab2, marker = 'o', alpha = 0.5, markerfacecolor="None")
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def scatterUno(x, y, yLab = 'Return process', xLab = 'Lagged return process', title = ''):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.scatter(x, y, label = yLab, facecolors = 'none', edgecolors = 'r', alpha = 0.5)
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def scatterDuo(x1, x2, y1, y2, yLab1, yLab2, yLab = 'Variance', xLab = 'Lagged returns', title = ""):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.scatter(x1, y1, s=10, c='b', marker="s", label=yLab1)
    ax.scatter(x2, y2, s=10, c='r', marker="o", label=yLab2)
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

