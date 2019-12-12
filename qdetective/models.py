""" models.py

author: daniel parker

a collection of parametric models for modelling resonators and testing fits
"""

import numpy as np

from sympy import *
from dsptypes import Parametric1D

def line_delay(b_tau = (0, 50, 100)):
    """ returns a Parametric1D model for the line delay

    Params:
        tau:    The line delay

    Args:   Parameter bounds as required by Parametric1D
    """
    tau, f = symbols('tau f')
    return Parametric1D(exp(-2j*np.pi*tau*f), {'tau': b_tau})

def ideal_notch(b_Qi = (2, 4, 6),
                b_Qc = (2, 4, 6),
                b_phi = (-45, 0, 45),
                b_fr = (1e9, 5e9, 11e9)):
    """ returns a Parametric1D model for an ideal notch resonator

    Params:
        Qi:     The log10 of the internal quality factor
        Qc:     The log10 of the modulus of the complex coupling quality factor
        phi:    The argument of the complex coupling quality factor
        fr:     The resonance frequency

    Args:   Parameter bounds as required by Parametric1D
    """
    Ql, Qi, Qc, phi, fr, f = symbols('Ql Qi Qc phi fr f')
    s21 = 1 - Ql/(10**Qc) * exp(1j * (np.pi/180) * phi)/(1 + 2j*Ql*(f/fr - 1))
    expr = s21.subs(Ql, 1/(1/(10**Qi) + 1/((10**Qc)*cos((np.pi/180) * phi))))

    params = {'Qi': b_Qi, 'Qc': b_Qc, 'phi': b_phi, 'fr': b_fr}

    return Parametric1D(expr, params)
