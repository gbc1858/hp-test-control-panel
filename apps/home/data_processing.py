# from hardware_settings import *
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import scipy.signal as signal
import os

lowcut = 11e9
highcut = 12.4e9
# lowcut = 10e6
# highcut = 20e6

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y


def cal_max_power(wfm, cable_attn, add_on_attn, dc_coupling=-60):
    try:
        max_v_raw = max(wfm)
        min_v_raw = min(wfm)
    #     v_pp = (max_v_raw - min_v_raw)/2
        v_pp = max_v_raw
        max_v = v_pp * 10 ** (-1 * (dc_coupling + cable_attn + add_on_attn) / 20) * 0.95
        max_p = max_v ** 2 /50/2/1e6     # MW
    except:
        max_p = 0
#     print(max_p)
    return max_p


def cal_envolope(wfm, Xincr, NR_Pt):
    time = [Xincr * i for i in list(range(int(NR_Pt)))]
    time_new = np.linspace(np.min(time), np.max(time), len(time) * 1)
    V_ref_interp_f = interp1d(time, wfm, kind='cubic', bounds_error=False, fill_value=0)
    V_ref_interp = V_ref_interp_f(time_new)
    V_ref_interp_filter = butter_bandpass_filter(V_ref_interp, lowcut, highcut, 1/ (time_new[1]-time_new[0]))
    env = np.abs(signal.hilbert(V_ref_interp_filter))
    return np.array(time_new) * 1e9, env


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def cal_convolution(wfm, Xincr, NR_Pt):
    gun_s11_file_path = os.getcwd() + '/S11_data/20201109_TWG.s2p'
    gun_s11 = pd.read_fwf(gun_s11_file_path, header=None, skiprows=list(np.arange(0, 9)))
    gun_s11_f = [float(i) for i in gun_s11[0]]
    s_data = gun_s11[1].str.split(' ', expand=True)
    gun_s11_raw = [float(i) for i in s_data[0]]
    gun_s11_phi_raw = [float(i) for i in s_data[1]]

    df0 = 5e5
    fmax = 25e9

    length = np.arange(0, fmax, df0).size
    n = next_power_of_2(int(2*length))
    freq = np.linspace(0, 2*fmax, n)
    freq = freq[0:int(n/2)]

    dt = 1/2/fmax
    t = np.arange(0, n/2) * dt

    gun_s11_raw_linear = [10**(i/20) for i in gun_s11_raw]
    s11_real = np.array(gun_s11_raw_linear) * np.cos(np.deg2rad(gun_s11_phi_raw))
    s11_real_interp_f = interp1d(gun_s11_f, s11_real, kind='cubic', bounds_error=False, fill_value=0)
    s11_real_interp = s11_real_interp_f(freq)
    s11_real_interp = np.concatenate((s11_real_interp, np.flipud(s11_real_interp)), axis=None)

    s11_img = np.array(gun_s11_raw_linear) * np.sin(np.deg2rad(gun_s11_phi_raw))
    s11_img_interp_f = interp1d(gun_s11_f, s11_img, kind='cubic', bounds_error=False, fill_value=0)
    s11_img_interp = s11_img_interp_f(freq)
    s11_img_interp = np.concatenate((s11_img_interp, np.flipud(s11_img_interp)*(-1)), axis=None)

    s11 = s11_real_interp + 1j * s11_img_interp
    gun_S11_linear = np.fft.ifft(s11)

    time = [Xincr * i for i in list(range(int(NR_Pt)))]
    V_for_interp_f = interp1d(time, wfm, kind='cubic', bounds_error=False, fill_value=0)
    V_for_interp = V_for_interp_f(t)

    Y1_con = np.convolve(V_for_interp, gun_S11_linear.real, mode='full')
    Y1_con = Y1_con[0:int(n/2)]
    env = np.abs(signal.hilbert(Y1_con))

    print('Convolution done.')
    return t * 1e9, env
