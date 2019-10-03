import pandas as pd
import numpy as np

def environment(f, a = 1, alpha = 0, tau = 0):
    return pd.Series(a*np.exp(1j*alpha)*np.exp(-2*np.pi*1j*f*tau), index = f)

def measurement_noise(f, power_dBm = -np.inf):
    voltage = np.sqrt(50 * 10**(power_dBm - 30)) # assumes a 50 ohm load
    complex_noise = np.random.normal(scale = 6*voltage, size = f.size) + \
                    1j*np.random.normal(scale = 6*voltage, size = f.size)
    return pd.Series(complex_noise, index = f)

def ideal_notch(f, fr, Qi = 1000, Qc = 1000, phi = 0):
    Ql = 1/(1/Qi + 1/(Qc*np.cos(np.deg2rad(phi))))
    s21 = 1 - Ql/Qc * np.exp(1j*np.deg2rad(phi))/(1 + 2j*Ql*(f/fr - 1))
    return pd.Series(s21, index = f)
