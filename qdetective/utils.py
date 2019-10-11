import matplotlib.pyplot as plt
import numpy as np

def plot_s21(s21, axes = None):
    if axes is None:
        fig, axes = plt.subplots(ncols = 3)

    axes[0].set_title('$|S_{21}|$')
    axes[1].set_title('$Arg (S_{21})$')
    axes[2].set_title('$S_{21}$ - complex')

    axes[0].plot(s21.index, 10*np.log10(np.abs(s21)) + 30)
    axes[1].plot(s21.index, np.rad2deg(np.angle(s21)))
    axes[2].scatter(np.real(s21), np.imag(s21))

    axes[2].scatter(1, 0, marker = 'x', color = 'r', s = 3)

    marginx = 0.5*np.ptp(np.real(s21))
    marginy = 0.5*np.ptp(np.imag(s21))
    axes[2].set_xlim(np.real(s21).min() - marginx, np.real(s21).max() + marginx)
    axes[2].set_ylim(np.imag(s21).min() - marginy, np.imag(s21).max() + marginy)

    return axes
