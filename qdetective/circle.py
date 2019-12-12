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
    x, y = s21_complex.real().values/norm, s21_complex.imag().values/norm
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
    err = np.sum(np.abs(np.abs(s21_complex.values - (xc + 1j*yc)) - r))

    # normalisation of eigenvector is handled by sqrt. other factors are cancelled
    return Circle(xc + 1j*yc, r), err
