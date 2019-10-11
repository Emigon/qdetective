import warnings

import numpy as np
import pandas as pd

from scipy.linalg import eig
from scipy.optimize import minimize, shgo

def circle_fit(s21_complex):
    # increases the abs values of the complex data so that the moments don't
    # look small compared to n = x.size
    norm = np.ptp(np.abs(s21_complex.values))
    x, y = np.real(s21_complex)/norm, np.imag(s21_complex)/norm
    z = x**2 + y**2

    Mx, My, Mz = x.sum(), y.sum(), z.sum()
    M = np.array(
        [
            [z@z, x@z, y@z, Mz    ],
            [z@x, x@x, y@x, Mx    ],
            [z@y, x@y, y@y, My    ],
            [Mz,  Mx,  My,  x.size]
        ])

    P = np.array(
        [
            [4*Mz,   2*Mx,   2*My,  0],
            [2*Mx, x.size,      0,  0],
            [2*My,      0, x.size,  0],
            [   0,      0,      0,  0]
        ])

    # find eigenvector associated with smallest non-negative eigenvalue
    vals, vects = eig(M, b = P)
    A, B, C, D = vects[:,np.abs(vals).argmin()] # works better than > 0 min ???
    xc, yc, r = -B/(2*A), -C/(2*A), np.sqrt(B**2 + C**2 - 4*A*D)/(2*np.abs(A))

    xc, yc = -norm * B/(2*A), -norm * C/(2*A)
    r = norm * np.sqrt(B**2 + C**2 - 4*A*D)/(2*np.abs(A))
    err = np.sum(np.abs(np.abs(s21_complex - (xc + 1j*yc)) - r))

    # normalisation of eigenvector is handled by sqrt. other factors are cancelled
    return xc, yc, r, err

def fwhm(s21):
    mod = np.abs(s21)
    flipped = mod.max() - mod
    return np.ptp(mod[flipped > .5*flipped.max()].index)

def symmeterise(s21):
    x0, y0, r, _ = circle_fit(s21)
    phi = np.sin(y0/r)
    return np.rad2deg(phi), 1 - r + np.exp(1j*phi)*(s21 - (x0 + 1j*y0))

def fit_line_delay(s21, max_line_length = 25):
    def costf(tau):
        return circle_fit(np.exp(1j*tau*2*np.pi*s21.index.values)*s21)[-1]
    optresult = shgo(costf, bounds = [(0,max_line_length/3e8)], n = 200, iters = 5,
                     sampling_method = 'sobol')
    return optresult, np.exp(1j*optresult.x*2*np.pi*s21.index.values)*s21

def normalise(s21):
    if np.ptp(s21.index) < 5*fwhm(s21):
        warnings.warn('Frequency span < 10*fwhm. Normalisation likely to be erroneous!')

    xc, yc, r, err = circle_fit(s21)
    w = xc + 1j*yc
    theta = (np.angle(s21.iloc[-1] - w) - np.angle(s21.iloc[0] - w))/2
    z = w + (s21.iloc[0] - w)*np.exp(1j*theta)
    return z, s21 / z

def draw_samples(s21, N):
    """ Samples s21 N times without replacement using gaussian with std = fwhm """
    probs = np.exp(-.5*(s21.index - np.abs(s21).idxmin())**2/(fwhm(s21)**2))
    probs /= np.sum(probs)

    samples = np.random.choice(s21.index, size = N, p = probs, replace = False)
    return s21[samples].sort_index()

