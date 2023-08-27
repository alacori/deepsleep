
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.sparse import spdiags, dia_matrix


__all__ = ['polynomial_detrend', 'smoothness_priors', 'sg_detrend']


def polynomial_detrend(rri, degree=1):
    if isinstance(rri, RRi):
        time = rri.time
        rri = rri.values
    else:
        time = _create_time_array(rri)

    coef = np.polyfit(time, rri, deg=degree)
    polynomial = np.polyval(coef, time)
    detrended_rri = rri - polynomial
    return RRiDetrended(detrended_rri, time=time)


def smoothness_priors(rri, l=500, fs=4.0):
    if isinstance(rri, RRi):
        time = rri.time
        rri = rri.values
    else:
        time = _create_time_array(rri)

    # TODO: only interp if not interpolated yet
    cubic_spline = CubicSpline(time, rri)
    time_interp = np.arange(time[0], time[-1], 1.0 / fs)
    rri_interp = cubic_spline(time_interp)
    N = len(rri_interp)
    identity = np.eye(N)
    B = np.dot(np.ones((N - 2, 1)), np.array([[1, -2, 1]]))
    D_2 = dia_matrix((B.T, [0, 1, 2]), shape=(N - 2, N))
    inv = np.linalg.inv(identity + l ** 2 * D_2.T @ D_2)
    z_stat = ((identity - np.linalg.inv(identity + l ** 2 * D_2.T @ D_2))) @ rri_interp

    rri_interp_detrend = np.squeeze(np.asarray(rri_interp - z_stat))
    return RRiDetrended(
        rri_interp - rri_interp_detrend,
        time=time_interp,
        detrended=True,
        interpolated=True,
    )


def sg_detrend(rri, window_length=51, polyorder=3, *args, **kwargs):
    if isinstance(rri, RRi):
        time = rri.time
        rri = rri.values
    else:
        time = _create_time_array(rri)

    trend = savgol_filter(
        rri, window_length=window_length, polyorder=polyorder, *args, **kwargs
    )
    return RRiDetrended(rri - trend, time=time, detrended=True)

"""### utils"""

#----------------------------------utils-------------------------------------#

from functools import wraps
from numbers import Number

import numpy as np
from scipy import interpolate

# TODO: Remove unused functions


# def _identify_rri_file_type(file_content):
#     is_hrm_file = file_content.find('[HRData]')
#     if is_hrm_file >= 0:
#         file_type = 'hrm'
#     else:
#         rri_lines = file_content.split('\n')
#         for line in rri_lines:
#             current_line_number = re.findall(r'\d+', line)
#             if current_line_number:
#                 if not current_line_number[0] == line.strip():
#                     raise FileNotSupportedError('Text file not supported')
#         file_type = 'text'
#     return file_type


# TODO: Refactor validation decorator
def validate_rri(func):
    @wraps(func)
    def _validate(rri, *args, **kwargs):
        _validate_positive_numbers(rri)
        rri = _transform_rri(rri)
        return func(rri, *args, **kwargs)

    def _validate_positive_numbers(rri):
        if not all(map(lambda value: isinstance(value, Number) and value > 0, rri)):
            raise ValueError(
                "rri must be a list or numpy.ndarray of positive"
                " and non-zero numbers"
            )

    return _validate


def _transform_rri(rri):
    return _transform_rri_to_miliseconds(np.array(rri))


# TODO: Refactor validation decorator
def validate_frequency_domain_arguments(func):
    def _check_frequency_domain_arguments(
        rri, fs=4.0, method="welch", interp_method="cubic", **kwargs
    ):
        _validate_available_methods(method)
        return func(rri, fs, method, interp_method, **kwargs)

    def _validate_available_methods(method):
        available_methods = ("welch", "ar")
        if method not in available_methods:
            raise ValueError(
                "Method not supported! Choose among: {}".format(
                    ", ".join(available_methods)
                )
            )

    return _check_frequency_domain_arguments


def _create_time_info(rri):
    rri_time = np.cumsum(rri) / 1000.0  # make it seconds
    return rri_time - rri_time[0]  # force it to start at zero


def _transform_rri_to_miliseconds(rri):
    if np.median(rri) < 1:
        rri *= 1000
    return rri


def _interpolate_rri(rri, time, fs=4, interp_method="cubic"):
    if interp_method == "cubic":
        return _interp_cubic_spline(rri, time, fs)
    elif interp_method == "linear":
        return _interp_linear(rri, time, fs)


def _interp_cubic_spline(rri, time, fs):
    time_rri_interp = _create_interp_time(time, fs)
    tck = interpolate.splrep(time, rri, s=0)
    rri_interp = interpolate.splev(time_rri_interp, tck, der=0)
    return rri_interp


def _interp_linear(rri, time, fs):
    time_rri_interp = _create_interp_time(time, fs)
    rri_interp = np.interp(time_rri_interp, time, rri)
    return rri_interp


def _create_interp_time(time, fs):
    time_resolution = 1 / float(fs)
    return np.arange(0, time[-1] + time_resolution, time_resolution)


def _ellipsedraw(ax, a, b, x0, y0, phi, *args, **kwargs):
    theta = np.arange(-0.03, 2 * np.pi, 0.01)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    X = np.cos(phi) * x - np.sin(phi) * y
    Y = np.sin(phi) * x + np.cos(phi) * y

    X = X + x0
    Y = Y + y0

    ax.plot(X, Y, *args, **kwargs)
    return ax

#--------------------------------------RRI------------------------------------------#

import sys
from collections.abc import MutableMapping
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np



__all__ = ['RRi', 'RRiDetrended']


class RRi:
    def __init__(self, rri, time=None, *args, **kwargs):
        if not isinstance(self, RRiDetrended):
            self.__rri = _validate_rri(rri)
            self.__detrended = False
            self.__interpolated = False
        else:
            self.__rri = np.array(rri)
            self.__detrended = kwargs.pop("detrended")
            self.__interpolated = kwargs.pop("interpolated", False)

        if time is None:
            self.__time = _create_time_array(self.rri)
        else:
            self.__time = _validate_time(self.__rri, time)

    def __len__(self):
        return len(self.__rri)

    def __getitem__(self, position):
        if isinstance(position, (slice, np.ndarray)):
            return RRi(self.__rri[position], self.time[position])
        else:
            return self.__rri[position]

    @property
    def values(self):
        """Return a numpy array containing the RRi values"""
        return self.__rri

    @property
    def rri(self):
        """Return a numpy array containing the RRi values"""
        return self.__rri

    @property
    def time(self):
        """Return a numpy array containing the time information"""
        return self.__time

    @property
    def detrended(self):
        """Return if the RRi series is detrended"""
        return self.__detrended

    @property
    def interpolated(self):
        """Return if the RRi series is interpolated"""
        return self.__interpolated

    def describe(self):
        table = _prepare_table(RRi(self.rri))
        rri_descr = RRiDescription(table)
        for row in table[1:]:
            rri_descr[row[0]]["rri"] = row[1]
            rri_descr[row[0]]["hr"] = row[2]

        return rri_descr

    def info(self):

        def _mem_usage(nbytes):
            mem_val = nbytes / 1024
            if mem_val < 1000:
                mem_str = "{:.2f}Kb".format(mem_val)
            else:
                mem_str = "{:.2f}Mb".format(mem_val / 1024)

            return mem_str

        n_points = self.__rri.size
        duration = self.__time[-1] - self.__time[0]
        # Hard coded interp and detrended. Future versions will have proper
        # attributes
        mem_usage = _mem_usage(self.__rri.nbytes)

        msg_template = (
            "N Points: {n_points}\nDuration: {duration:.2f}s\n"
            "Interpolated: {interp}\nDetrended: {detrended}\n"
            "Memory Usage: {mem_usage}"
        )
        sys.stdout.write(
            msg_template.format(
                n_points=n_points,
                duration=duration,
                interp=self.interpolated,
                detrended=self.detrended,
                mem_usage=mem_usage,
            )
        )

    def to_hr(self):
        return 60 / (self.rri / 1000.0)

    def time_range(self, start, end):
        interval = np.logical_and(self.time >= start, self.time <= end)
        return RRi(self.rri[interval], time=self.time[interval])

    def reset_time(self, inplace=False):
        if inplace:
            self.__time -= self.time[0]
        else:
            return RRi(self.rri, time=self.time - self.time[0])

    def plot(self, ax=None, *args, **kwargs):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.time, self.rri, *args, **kwargs)
        ax.set(xlabel="Time (s)", ylabel="RRi (ms)")
        plt.show(block=False)

        return fig, ax

    def hist(self, hr=False, *args, **kwargs):
        fig, ax = plt.subplots(1, 1)
        if hr:
            ax.hist(self.to_hr(), *args, **kwargs)
            ax.set(xlabel="HR (bpm)", ylabel="Frequency")
        else:
            ax.hist(self.rri, *args, **kwargs)
            ax.set(xlabel="RRi (ms)", ylabel="Frequency")

        plt.show(block=False)

        return fig, ax

    def poincare_plot(self):
        """
        Poincaré plot of the RRi series
        """
        fig, ax = plt.subplots(1, 1)
        rri_n, rri_n_1 = self.rri[:-1], self.rri[1:]
        ax.plot(rri_n, rri_n_1, ".k")

        ax.set(xlabel="$RRi_n$ (ms)", ylabel="$RRi_{n+1}$ (ms)", title="Poincaré Plot")

        plt.show(block=False)

        dx = abs(max(rri_n) - min(rri_n)) * 0.05
        dy = abs(max(rri_n_1) - min(rri_n_1)) * 0.05
        xlim = [min(rri_n) - dx, max(rri_n) + dx]
        ylim = [min(rri_n_1) - dy, max(rri_n_1) + dy]

        from hrv.classical import non_linear

        nl = non_linear(self.rri)
        a = rri_n / np.cos(np.pi / 4.0)
        ca = np.mean(a)

        cx, cy, _ = ca * np.cos(np.pi / 4.0), ca * np.sin(np.pi / 4.0), 0

        width = nl["sd2"]  # to seconds
        height = nl["sd1"]  # to seconds

        # plot fx(x) = x
        sd2_l = ax.plot(
            [xlim[0], xlim[1]], [ylim[0], ylim[1]], "--", color=[0.5, 0.5, 0.5]
        )
        fx = lambda val: -val + 2 * cx

        sd1_l = ax.plot([xlim[0], xlim[1]], [fx(xlim[0]), fx(xlim[1])], "k--")
        ax = _ellipsedraw(
            ax, width, height, cx, cy, np.pi / 4.0, color="r", linewidth=3
        )
        ax.legend(
            (sd1_l[0], sd2_l[0]), ("SD1: %.2fms" % height, "SD2: %.2fms" % width),
        )

        return fig, ax

    # TODO: Create methods for time domain to be calculted in the instance

    def mean(self, *args, **kwargs):
        """Return the average of the RRi series"""
        return np.mean(self.rri, *args, **kwargs)

    def var(self, *args, **kwargs):
        """Return the variance of the RRi series"""
        return np.var(self.rri, *args, **kwargs)

    def std(self, *args, **kwargs):
        """Return the standard deviation of the RRi series"""
        return np.std(self.rri, *args, **kwargs)

    def median(self, *args, **kwargs):
        """Return the median of the RRi series"""
        return np.median(self.rri, *args, **kwargs)

    def max(self, *args, **kwargs):
        """Return the max value of the RRi series"""
        return np.max(self.rri, *args, **kwargs)

    def min(self, *args, **kwargs):
        """Return the min value of the RRi series"""
        return np.min(self.rri, *args, **kwargs)

    def amplitude(self):
        """Return the amplitude (max - min) of the RRi series"""
        return self.max() - self.min()

    def rms(self):
        """Return the root mean squared of the RRi series"""
        return np.sqrt(np.mean(self.rri ** 2))

    def time_split(self, seg_size, overlap=0, keep_last=False):
        rri_duration = self.time[-1]
        if overlap > seg_size:
            raise Exception("`overlap` can not be bigger than `seg_size`")
        elif seg_size > rri_duration:
            raise Exception("`seg_size` is longer than RRi duration.")

        begin = 0
        end = seg_size
        step = seg_size - overlap
        n_splits = int((rri_duration - seg_size) / step) + 1
        segments = []
        for i in range(n_splits):
            OP = np.less if i + 1 != n_splits else np.less_equal
            mask = np.logical_and(self.time >= begin, OP(self.time, end))
            segments.append(RRi(self.rri[mask], time=self.time[mask]))
            begin += step
            end += step

        last = segments[-1]
        if keep_last and last.time[-1] < rri_duration:
            mask = self.time > begin
            segments.append(RRi(self.rri[mask], time=self.time[mask]))

        return segments

    def __repr__(self):
        return "RRi %s" % np.array_repr(self.rri)

    def __mul__(self, val):
        return RRi(self.rri * val, self.time)

    def __add__(self, val):
        return RRi(self.rri + val, self.time)

    def __sub__(self, val):
        return RRi(self.rri - val, self.time)

    def __truediv__(self, val):
        return RRi(self.rri / val, self.time)

    def __pow__(self, val):
        return RRi(self.rri ** val, self.time)

    def __abs__(self):
        return RRi(np.abs(self.rri), self.time)

    def __eq__(self, val):
        return self.rri == val

    def __ne__(self, val):
        return self.rri != val

    def __gt__(self, val):
        return self.rri > val

    def __ge__(self, val):
        return self.rri >= val

    def __lt__(self, val):
        return self.rri < val

    def __le__(self, val):
        return self.rri <= val


class RRiDetrended(RRi):
    # TODO: add trend as attribute of the instance
    def __init__(self, rri, time, *args, **kwargs):
        detrended = True
        interpolated = kwargs.pop("interpolated", False)
        super().__init__(rri, time, interpolated=interpolated, detrended=detrended)


class RRiDescription(MutableMapping):
    def __init__(self, table, *args, **kwargs):
        self.store = defaultdict(dict)
        self.update(dict(*args, **kwargs))
        self.table = table

    def keys(self):
        return self.store.keys()

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        descr = ""
        dash = "-" * 40 + "\n"
        for i in range(len(self.table)):
            if i == 0:
                descr += dash
                descr += "{:<10s}{:>12s}{:>12s}\n".format(
                    self.table[i][0], self.table[i][1], self.table[i][2]
                )
                descr += dash
            else:
                descr += "{:<10s}{:>12.2f}{:>12.2f}\n".format(
                    self.table[i][0], self.table[i][1], self.table[i][2]
                )

        return descr


def _prepare_table(rri):
    def _amplitude(values):
        return values.max() - values.min()

    header = ["", "rri", "hr"]
    fields = ["min", "max", "mean", "var", "std"]
    hr = rri.to_hr()

    table = []
    for field in fields:
        rri_var = rri.__getattribute__(field)()
        hr_var = hr.__getattribute__(field)()
        table.append([field, rri_var, hr_var])

    table.append(["median", rri.median(), np.median(hr)])
    table.append(["amplitude", rri.amplitude(), _amplitude(hr)])

    return [header] + table


def _validate_rri(rri):
    # TODO: let the RRi be in seconds if the user wants to
    rri = np.array(rri, dtype=np.float64)

    if any(rri <= 0):
        raise ValueError("rri series can only have positive values")

    # Use RRi series median value to check if it is in seconds or miliseconds
    if np.median(rri) < 10:
        rri *= 1000.0

    return rri


def _validate_time(rri, time):
    time = np.array(time, dtype=np.float64)
    if len(rri) != len(time):
        raise ValueError("rri and time series must have the same length")

    if any(time[1:] == 0):
        raise ValueError("time series cannot have 0 values after first position")

    if not all(np.diff(time) > 0):
        raise ValueError("time series must be monotonically increasing")

    if any(time < 0):
        raise ValueError("time series cannot have negative values")

    return time


def _create_time_array(rri):
    time = np.cumsum(rri) / 1000.0
    return time - time[0]

"""### classical"""

#-----------------------------classical-----------------------------#

import numpy as np

from scipy.signal import welch
from spectrum import pburg

from collections.abc import MutableMapping
from collections import defaultdict


__all__ = ['time_domain', 'frequency_domain', 'non_linear']


# validate_rri
def time_domain(rri):
    # TODO: let user choose interval for pnn50 and nn50.
    diff_rri = np.diff(rri)
    rmssd = np.sqrt(np.mean(diff_rri ** 2))
    sdnn = np.std(rri, ddof=1)  # make it calculates N-1
    sdsd = np.std(diff_rri, ddof=1)
    nn50 = _nn50(rri)
    pnn50 = _pnn50(rri)
    mrri = np.mean(rri)
    mhr = np.mean(60 / (rri / 1000.0))

    return dict(
        zip(
            ["rmssd", "sdnn", "sdsd", "nn50", "pnn50", "mrri", "mhr"],
            [rmssd, sdnn, sdsd, nn50, pnn50, mrri, mhr],
        )
    )


def _nn50(rri):
    return sum(abs(np.diff(rri)) > 50)


def _pnn50(rri):
    return _nn50(rri) / len(rri) * 100


# TODO: create nperseg, noverlap, order, nfft, and detrend arguments
def frequency_domain(
    rri,
    time=None,
    fs=4.0,
    method="welch",
    interp_method="cubic",
    detrend="constant",
    vlf_band=(0, 0.04),
    lf_band=(0.04, 0.15),
    hf_band=(0.15, 0.4),
    **kwargs
):
    if isinstance(rri, RRi):
        time = rri.time if time is None else time
        detrend = detrend if not rri.detrended else False
        interp_method = interp_method if not rri.interpolated else None

    if interp_method is not None:
        rri = _interpolate_rri(rri, time, fs, interp_method)

    if method == "welch":
        fxx, pxx = welch(x=rri, fs=fs, detrend=detrend, **kwargs)
    elif method == "ar":
        if detrend:
            rri = polynomial_detrend(rri, degree=1)
        fxx, pxx = _calc_pburg_psd(rri=rri, fs=fs, **kwargs)

    return _auc(fxx, pxx, vlf_band, lf_band, hf_band)


def _auc(fxx, pxx, vlf_band, lf_band, hf_band):
    vlf_indexes = np.logical_and(fxx >= vlf_band[0], fxx < vlf_band[1])
    lf_indexes = np.logical_and(fxx >= lf_band[0], fxx < lf_band[1])
    hf_indexes = np.logical_and(fxx >= hf_band[0], fxx < hf_band[1])

    vlf = np.trapz(y=pxx[vlf_indexes], x=fxx[vlf_indexes])
    lf = np.trapz(y=pxx[lf_indexes], x=fxx[lf_indexes])
    hf = np.trapz(y=pxx[hf_indexes], x=fxx[hf_indexes])
    total_power = vlf + lf + hf
    lf_hf = lf / hf
    lfnu = (lf / (total_power - vlf)) * 100
    hfnu = (hf / (total_power - vlf)) * 100

    return dict(
        zip(
            ["total_power", "vlf", "lf", "hf", "lf_hf", "lfnu", "hfnu"],
            [total_power, vlf, lf, hf, lf_hf, lfnu, hfnu],
        )
    )


def _calc_pburg_psd(rri, fs, order=16, nfft=None):
    burg = pburg(data=rri, order=order, NFFT=nfft, sampling=fs)
    burg.scale_by_freq = False
    burg()
    return np.array(burg.frequencies()), burg.psd


# validate_rri
def non_linear(rri):
    sd1, sd2 = _poincare(rri)
    return dict(zip(["sd1", "sd2"], [sd1, sd2]))


def _poincare(rri):
    diff_rri = np.diff(rri)
    sd1 = np.sqrt(np.std(diff_rri, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(rri, ddof=1) ** 2 - 0.5 * np.std(diff_rri, ddof=1) ** 2)
    return sd1, sd2