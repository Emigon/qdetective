import warnings

import numpy as np
import pandas as pd

from scipy.linalg import eig
from scipy.optimize import shgo

import matplotlib.patches as patches

class Circle(object):
    def __init__(self, z, r):
        self.z = z
        self.r = r

    @property
    def x(self):
        return np.real(self.z)

    @property
    def y(self):
        return np.imag(self.z)

    def rotate(self, theta):
        self.z *= np.exp(1j*theta)
        return self

    def scale(self, scaling_factor):
        self.z *= scaling_factor
        self.r *= np.abs(scaling_factor)
        return self

    def add_to(self, axes):
        axes.add_patch(patches.Circle((self.x, self.y), radius = self.r, fill = False))
        axes.scatter(self.x, self.y, marker = '.', color = 'k')

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
    idxs, = np.where(vals > 0)
    A, B, C, D = vects[:,idxs][:,vals[idxs].argmin()]

    xc, yc = -norm * B/(2*A), -norm * C/(2*A)
    r = norm * np.sqrt(B**2 + C**2 - 4*A*D)/(2*np.abs(A))
    err = np.sum(np.abs(np.abs(s21_complex - (xc + 1j*yc)) - r))

    # normalisation of eigenvector is handled by sqrt. other factors are cancelled
    return Circle(xc + 1j*yc, r), err

def fwhm(s21):
    mod = np.abs(s21)
    flipped = mod.max() - mod
    return np.ptp(mod[flipped > .5*flipped.max()].index)

def fit_line_delay(s21, max_line_length = 25):
    def costf(tau):
        return circle_fit(np.exp(1j*tau*2*np.pi*s21.index.values)*s21)[-1]
    optresult = shgo(costf, bounds = [(0,max_line_length/3e8)], n = 200, iters = 5,
                     sampling_method = 'sobol')
    circular_s21 = np.exp(1j*optresult.x*2*np.pi*s21.index.values)*s21
    return optresult, circular_s21, circle_fit(circular_s21)[0]

def normalise(s21, circle):
    if np.ptp(s21.index) < 5*fwhm(s21):
        warnings.warn('Frequency span < 10*fwhm. Normalisation likely to be erroneous!')

    theta = ((np.angle(s21.iloc[-1] - circle.z) - \
                np.angle(s21.iloc[0] - circle.z))/2 % np.pi)

    z = circle.z + (s21.iloc[0] - circle.z)*np.exp(1j*theta)
    # adjust the point so that it sits on the fitted circle
    z = circle.z + circle.r*(z - circle.z)/np.abs(z - circle.z)
    return z, s21 / z, circle.scale(1/z)

def draw_samples(s21, N):
    """ Samples s21 N times without replacement using gaussian with std = fwhm """
    probs = np.exp(-.5*(s21.index - np.abs(s21).idxmin())**2/(fwhm(s21)**2))
    probs /= np.sum(probs)

    samples = np.random.choice(s21.index, size = N, p = probs, replace = False)
    return s21[samples].sort_index()
