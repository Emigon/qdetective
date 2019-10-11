import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from qdetective import *
from qdetective.utils import *

def fit_notch(s21, N = 500, debug = False):
    params = pd.Series(index = ['phi', 'fr', 'Qi', '|Qc|', 'Ql', 'a', 'alpha', 'tau'],\
                       dtype = np.float64)

    # sample the resonance over N points to reduce fitting power off resonance
    np.random.seed(42)
    sampled = draw_samples(s21, N = N)
    f = sampled.index.values

    # remove line delay
    opt, s21_circular, circle_A = fit_line_delay(sampled)
    params['tau'] = opt.x[0]

    # normalise the resonance about 1 + 0j
    z, s21_norm, circle_B = normalise(s21_circular, circle_A)
    params['a'], params['alpha'] = np.abs(z), np.rad2deg(np.angle(z))

    # estimate the tilt angle
    phi = -np.angle(1 - circle_B.z)
    params['phi'] = np.rad2deg(phi)

    # estimate the resonance frequency to be diametrically opposite 1 + 0j
    sfr = -(1 + 0j - circle_B.z) + circle_B.z
    params['fr'] = np.abs(s21_norm - sfr).idxmin()

    # polish Ql with shgo
    def costf(q):
        fit = 1 - 2*circle_B.r*np.exp(-1j*phi) / (1 + 2j*q*(f/params['fr'] - 1))
        return np.sum(np.abs(fit - s21_norm))

    Qrough = params['fr']/fwhm(s21)
    opt = shgo(costf, bounds = [(.8*Qrough, 1e3*Qrough)], iters = 5)

    # calculate individual Q factors
    params['Ql'] = opt.x
    params['|Qc|'] = params['Ql']/(2*circle_B.r)
    params['Qi'] = 1/(1/params['Ql'] - 1/(params['|Qc|']*np.cos(phi)))

    params['sigma_x'] = 2*np.std(np.abs(s21_norm - circle_B.z))
    params['sigma_Qi'] = _sigma_Qi(2*circle_B.r, params['Ql'], params['sigma_x'])
    params['sigma_Qc'] = _sigma_Qc(2*circle_B.r, params['Ql'], params['sigma_x'])

    # combine all the parameters into one model to return
    notch = 1 - params['Ql']/np.abs(params['|Qc|'])*np.exp(-1j*phi) / \
                (1 + 2j*params['Ql']*(f/params['fr'] - 1))

    fit = params['a'] * np.exp(1j*np.deg2rad(params['alpha'])) * \
            np.exp(-2*np.pi*1j*params['tau']*f) * notch

    # plot the intermediate results if in debug mode
    if debug:
        # line delay removed and f_inf point estimated
        axes = plot_s21(s21_circular)
        circle_A.add_to(axes[2])
        fm, fp = s21_circular.iloc[0], s21_circular.iloc[-1]
        axes[2].scatter(np.real(z), np.imag(z), color = 'k', marker = 's')
        axes[2].scatter(np.real(fm), np.imag(fm), color = 'k', marker = 'v')
        axes[2].scatter(np.real(fp), np.imag(fp), color = 'k', marker = '^')
        axes[0].set_ylabel('line delay removed')
        plt.tight_layout()
        plt.show()

        # resonance after normalisation
        axes = plot_s21(s21_norm)
        circle_B.add_to(axes[2])
        axes[0].set_ylabel('normalised')
        plt.tight_layout()
        plt.show()

        # resonance after Ql polishing
        axes = plot_s21(s21_norm)
        plot_s21(pd.Series(notch, index = f), axes = axes)
        circle_B.add_to(axes[2])
        axes[0].set_ylabel('polished')
        axes[2].scatter(np.real(sfr), np.imag(sfr), color = 'r')
        plt.tight_layout()
        plt.show()

    return params, pd.Series(fit, index = f)

# for uncertainty calculations
def _sigma_Qc(x, Ql, sigma_x):
    r = 1/(1/x - 1)
    if r < 0:
        return np.nan

    if np.log10(r) < -2 or np.log10(r) > 2:
        warnings.warn('Likely underestimated std on Qc for r = %.2f' % r)

    return Ql * sigma_x / x**2

def _mu_Qc(x, Ql):
    return Ql/x

def _mu_Qi(x, Ql):
    return 1/(1/Ql - 1/_mu_Qc(x, Ql))

def _sigma_Qi(x, Ql, sigma_x):
    r = 1/(1/x - 1)
    if r < 0:
        return np.nan

    if np.log10(r) < -2 or np.log10(r) > 2:
        warnings.warn('Likely underestimated std on Qi for r = %.2f' % r)

    return _mu_Qi(x, Ql)**2 / _mu_Qc(x, Ql)**2 * _sigma_Qc(x, Ql, sigma_x)
