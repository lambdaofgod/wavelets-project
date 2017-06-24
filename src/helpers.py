from numpy import sqrt
import pywt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Constants:
  fontsize = 25

gray_plotter = lambda img: plt.imshow(img, cmap='gray', vmax = 255, vmin=0)

def plot_linear_layout(imgs, titles = [], plotter = gray_plotter):
  N = len(imgs)
  gs = GridSpec(1, N)
  gs.update(wspace=0, hspace=0)
  plot_with_gs(imgs, titles, gs, plotter, N)

def plot_rectangular_layout(imgs, titles = [], plotter = gray_plotter):
  N = len(imgs)
  n = int(sqrt(len(imgs)))
  gs = GridSpec(n, n)
  gs.update(wspace=0, hspace=0)
  plot_with_gs(imgs, titles, gs, plotter, N)

def plot_with_gs(imgs, titles, gridspec, plotter, N):
  def plottitle(i):
    if len(titles) == N:
      plt.title(titles[i], fontsize = Constants.fontsize)

  for i in range(N):
    img = imgs[i]
    plt.subplot(gridspec[i])
    plotter(img)
    plottitle(i)
    plt.axis('off')
  plt.show()

