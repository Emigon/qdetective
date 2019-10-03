import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scipy.optimize import shgo

from qdetective import *
from qdetective.utils import *

def fit_notch(s21, debug = False):
    params = pd.Series(index = ['phi', 'fr', 'Qi', '|Qc|', 'Ql', 'a', 'alpha', 'tau'],\
                       dtype = np.float64)

    # remove line delay
    opt, s21_circular = fit_line_delay(s21)
    params['tau'] = opt.x[0]

    if debug:
        xc, yc, r, _ = circle_fit(s21_circular)
        axes = plot_s21(s21_circular)
        axes[2].add_patch(Circle((xc, yc), radius = r, fill = False))
        plt.tight_layout()
        plt.show()

    # normalise the resonance about 1 + 0j
    z, s21_norm = normalise(s21_circular)
    params['a'], params['alpha'] = np.abs(z), np.rad2deg(np.angle(z))

    if debug:
        xc, yc, r, _ = circle_fit(s21_norm)
        axes = plot_s21(s21_norm)
        axes[2].add_patch(Circle((xc, yc), radius = r, fill = False))
        plt.tight_layout()
        plt.show()

    # compute the circle tilt and resonance freqency
    params['phi'], s21_sym = symmeterise(s21_norm)
    phi = np.deg2rad(params['phi'])
    params['fr'] = s21_sym.idxmin()

    if debug:
        xc, yc, r, _ = circle_fit(s21_sym)
        axes = plot_s21(s21_sym)
        axes[2].add_patch(Circle((xc, yc), radius = r, fill = False))

    # fit qualitiy factors
    opt, s21_final = polish_Ql(s21_sym)
    xc, yc, r, _ = circle_fit(s21_sym)
    params['Ql'] = opt.x
    params['|Qc|'] = params['Ql']/(2*r)
    params['Qi'] = 1/(1/params['Ql'] - 1/(params['|Qc|']*np.cos(phi)))

    params['sigma_x'] = 2*np.std(np.abs(s21_sym - xc - 1j*yc))
    params['sigma_Qi'] = _sigma_Qi(2*r, params['Ql'], params['sigma_x'])
    params['sigma_Qc'] = _sigma_Qc(2*r, params['Ql'], params['sigma_x'])

    if debug:
        plot_s21(s21_final, axes = axes)
        axes[2].add_patch(Circle((xc, yc), radius = r, fill = False))
        plt.tight_layout()
        plt.show()

    f = s21.index.values
    notch = 1 - params['Ql']/np.abs(params['|Qc|'])*np.exp(-1j*phi) / \
                (1 + 2j*params['Ql']*(f/params['fr'] - 1))

    fit = params['a'] * np.exp(1j*np.deg2rad(params['alpha'])) * \
            np.exp(-2*np.pi*1j*params['tau']*f) * notch

    return params, pd.Series(fit, index = f)

def polish_Ql(s21):
    fr = np.abs(s21).idxmin()
    _, _, r, _ = circle_fit(s21)

    def costf(q):
        fit = 1 - 2*r/(1 + 2j*q*(s21.index.values/fr - 1))
        return np.linalg.norm(fit - s21)

    Ql_rough = np.abs(s21).idxmin()/fwhm(s21)
    result = shgo(costf, bounds = [(.1*Ql_rough, 10*Ql_rough)], iters = 5)

    fit = 1 - 2*r/(1 + 2j*result.x*(s21.index.values/fr - 1))
    return result, pd.Series(fit, index = s21.index)

# for uncertainty calculations
def _sigma_Qc(x, Ql, sigma_x):
    r = 1/(1/x - 1)
    if np.log10(r) < -2 or np.log10(r) > 2:
        warnings.warn('Likely underestimated std on Qc for r = %.2f' % r)
    return Ql * sigma_x / x**2

def _mu_Qc(x, Ql):
    return Ql/x

def _mu_Qi(x, Ql):
    return 1/(1/Ql - 1/_mu_Qc(x, Ql))

def _sigma_Qi(x, Ql, sigma_x):
    r = 1/(1/x - 1)
    if np.log10(r) < -2 or np.log10(r) > 2:
        warnings.warn('Likely underestimated std on Qi for r = %.2f' % r)
    return _mu_Qi(x, Ql)**2 / _mu_Qc(x, Ql)**2 * _sigma_Qc(x, Ql, sigma_x)
