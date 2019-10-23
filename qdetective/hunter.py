import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, find_peaks

def estimate_baseline(sparam, Qmin):
    window_size = int(2*Qmin) # the number of points to fit the spline over
                              # make sure its bigger than the minimum Q
    if not(window_size % 2):
        window_size += 1

    filtered = savgol_filter(np.abs(sparam), window_size, 3)
    baseline = lambda f: np.interp(f, sparam.index.values, filtered)
    return baseline

def high_res_sampler(sampler, fmin, fmax, N, Qmax = 1e6, k = 10):
    # the smallest interval required to get k samples over a resonance
    delta_f = N*fmin/(k*Qmax)
    N_total = (fmax - fmin)/delta_f
    N_intervals = round(N_total/N + 0.5)

    high_res = pd.Series()

    f_start = fmin
    for i in range(N_intervals):
        f_stop = np.min([f_start + delta_f*N, fmax])
        samples = sampler(f_start, f_stop)
        high_res = pd.concat([high_res, samples], sort = True)

        if f_stop == fmax:
            break

        f_start = f_stop + delta_f
    return high_res

def locate_resonances(high_res, fmin, fmax, N, Qmin = 1e2, Qmax = 1e6, k = 10):
    delta_f = N*fmin/(k*Qmax)
    widest = int(fmax/Qmin / delta_f)
    narrowest = int(fmin/Qmax / delta_f)

    peaks, properties = find_peaks(-high_res,\
                                   prominence = 6*np.std(high_res),\
                                   distance = widest,\
                                   width = (narrowest, widest))

    peak_info = pd.DataFrame(properties, index = high_res.iloc[peaks].index)
    peak_info['widths'] *= delta_f
    return peaks, peak_info

def hunt(sampler, fmin, fmax, N, Qmin = 1e2, Qmax = 1e6, debug = False):
    baseline_est = estimate_baseline(sampler(fmin, fmax, N = N), Qmin)

    def cleaned_sampler(f_start, f_stop, N = 1000):
        pts = np.linspace(f_start, f_stop, N)
        return np.abs(sampler(f_start, f_stop, N = N)) - baseline_est(pts)

    tr = high_res_sampler(cleaned_sampler, fmin, fmax, N, Qmax = Qmax)

    peaks, peak_info = \
        locate_resonances(tr, 6.0e9, 9.0e9, 1000, Qmin = Qmin, Qmax = Qmax)

    if debug:
        fig, (ax1, ax2) = plt.subplots(nrows = 2)
        ax1.plot(np.abs(sampler(fmin, fmax, N = N)))
        ax2.plot(tr)
        ax2.plot(tr.iloc[peaks], 'x')
        fig.tight_layout()
        plt.show()

    resonances = []
    for fc in peak_info.index:
        # use the find_peaks fwhm estimate to zoom in on each resonance
        fwhm = peak_info.loc[fc,'widths']
        test = cleaned_sampler(fc - 2.5*fwhm, fc + 2.5*fwhm, N = N)

        # refine the fwhm estimate with these new high resolution traces
        halfmax = .5*(np.max(test) - tr.loc[fc])
        region, = np.where(test < tr.loc[fc] + halfmax)
        fwhm = np.ptp(test.iloc[region].index)

        # set to 5*fwhm for circle fitting
        resonances.append(sampler(fc - 2.5*fwhm, fc + 2.5*fwhm, N = N))

    return resonances
