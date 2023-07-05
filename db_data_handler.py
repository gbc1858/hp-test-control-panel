import sqlite3
from sqlite3 import Error
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d, CubicSpline
import pandas as pd

import numpy as np
from scipy.integrate import simps
import scipy.signal as signal
plt.rcParams['savefig.dpi'] = 500


def connect_db(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        print('connected to db.')
    except Error as e:
        print(e)

    return conn


def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM home_record")
    rows = cur.fetchall()

    for row in rows:
        print(row)


def select_fs_task_id(conn, task_id, for_ch, ref_ch, fc_ch):
    cur = conn.cursor()
    cmd = f"SELECT {for_ch}, {ref_ch}, {fc_ch}, created_at, Xincr, NR_Pt FROM home_record WHERE type=? AND task_id=?"
    cur.execute(cmd, ('fs', task_id,))
    rows = cur.fetchall()

    for_data, ref_data, fc_data, time = [], [], [], []
    for row in rows:
        for_data_temp = [float(i) for i in row[0][1:-1].split(', ')]
        for_data.append(for_data_temp)
        ref_data_temp = [float(i) for i in row[1][1:-1].split(', ')]
        ref_data.append(ref_data_temp)
        fc_data_temp = [float(i) for i in row[2][1:-1].split(', ')]
        fc_data.append(fc_data_temp)
        time.append(row[3])

        Xincr = float(row[4])
        NR_Pt = int(row[5])
    return for_data, ref_data, fc_data, time, Xincr, NR_Pt


def select_ss_task_id(conn, task_id, ict3_ch, ict4_ch, diode_ch):
    cur = conn.cursor()
    cmd = f"SELECT {ict3_ch}, {ict4_ch}, created_at, Xincr, NR_Pt, {diode_ch} FROM home_record WHERE type=? AND task_id=?"
    cur.execute(cmd, ('ss', task_id,))
    rows = cur.fetchall()

    ict3_data, ict4_data, diode_data, time = [], [], [], []
    for row in rows:
        ict3_data_temp = [float(i) for i in row[0][1:-1].split(', ')]
        ict3_data.append(ict3_data_temp)
        ict4_data_temp = [float(i) for i in row[1][1:-1].split(', ')]
        ict4_data.append(ict4_data_temp)
        time.append(row[2])

        Xincr = float(row[3])
        NR_Pt = int(row[4])

        diode_data_temp = [float(i) for i in row[5][1:-1].split(', ')]
        diode_data.append(diode_data_temp)
    return ict3_data, ict4_data, time, Xincr, NR_Pt, diode_data

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

    Y1_con_filter = butter_bandpass_filter(Y1_con, lowcut, highcut, 1/ (t[1]-t[0]))

    env = np.abs(signal.hilbert(Y1_con_filter))

    print('Convolution done.')
    return t * 1e9, env, Y1_con_filter

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


ICT_CALIBRATION_FACTOR = 1.25
ICT_CUT_LOW_BOUNDARY = 0
ICT_CUT_HIGH_BOUNDARY = 0

class CalICTCharge:
    def __init__(self, file, time, signal_volt):
        self.file = file
        self.time = time
        self.signal_volt = signal_volt
        self.signal_volt_denoise = []
        self.offset_all_shots = []
        self.dt = None
        self.charge_all_shots = []

        self.denoise_data()
        self.get_ict_charge()

    def denoise_data(self):
        for i in range(len(self.signal_volt)):
            N = 5       # Filter order
            Wn = 0.1    # Cutoff frequency
            sos = signal.butter(N, Wn, output='sos')
            self.signal_volt_denoise.append(signal.sosfiltfilt(sos, self.signal_volt[i]))

    def get_signal_offset(self):
        for i in range(len(self.signal_volt)):
            offset_single_shot = np.mean(self.signal_volt[i][:20])
            self.offset_all_shots.append(offset_single_shot)

    def integration_step(self):
        self.dt = abs(self.time[1] - self.time[0])

    def get_ict_charge(self):
        self.get_signal_offset()
        self.integration_step()
        volt_w_offset = [np.array(self.signal_volt[i]) - self.offset_all_shots[i] for i in range(len(self.offset_all_shots))]

        min = [np.min(volt_w_offset[i]) for i in range(len(volt_w_offset))]
        min_index = [list(volt_w_offset[i]).index(min[i]) for i in range(len(volt_w_offset))]

#         volt_w_offset = [volt_w_offset[i][min_index[i]-1500:min_index[i]+1500] for i in range(len(volt_w_offset))]
        volt_w_offset = [volt_w_offset[i] for i in range(len(volt_w_offset))]

        for i in range(len(volt_w_offset)):
            charge = simps(y=np.array(volt_w_offset[i]), dx=float(self.dt))
            self.charge_all_shots.append(abs(charge * 1e9 / ICT_CALIBRATION_FACTOR))  # charge in nC

        # remove highest N and lowest M results.
        desired_charges = sorted(self.charge_all_shots)[
                          ICT_CUT_LOW_BOUNDARY: (len(self.charge_all_shots) - ICT_CUT_HIGH_BOUNDARY)]
        print(f'min: {np.min(desired_charges): .2f} nC | max: {np.max(desired_charges): .2f} nC | '
              f'std: {np.std(desired_charges): .2f} nC | ave charge: {abs(np.mean(desired_charges)): .2f} nC')

    def plot_ict_data(self):
        for i in range(len(self.signal_volt)):
            plt.plot(self.time, self.signal_volt[i], c='b')
            plt.axhline(self.offset_all_shots[i], c='darkorange', ls=':')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.show()


def main():
    db_path = os.getcwd()
    db = '/2022_10_04.sqlite3'
    conn = connect_db(db_path + db)

    id = 21
    with conn:
        for_data, ref_data, fc_data, time, xincr, NR_Pt = select_fs_task_id(conn, task_id=id, for_ch='Ch2', ref_ch='Ch3', fc_ch='Ch4')
        ict3_data, ict4_data, time_ss, xincr_ss, NR_Pt_ss = select_ss_task_id(conn, task_id=id, ict3_ch='Ch1', ict4_ch='Ch2')

    for i in range(len(ict3_data)):
        ict3_data[i] = np.array(ict3_data[i]) * 10 ** (10 / 20)
        ict4_data[i] = np.array(ict4_data[i]) * 10 ** (10 / 20)

    x_fs = [i * xincr for i in list(range(NR_Pt))]
    x_ss = [i * xincr_ss for i in list(range(NR_Pt_ss))]

    ICT_ch3 = CalICTCharge(time_ss, x_ss, ict3_data)
    ICT_ch4 = CalICTCharge(time_ss, x_ss, ict4_data)

    plt.scatter(list(range(len(ICT_ch3.charge_all_shots))), ICT_ch3.charge_all_shots, c=cmap3[0], label='ICT3')
    plt.scatter(list(range(len(ICT_ch3.charge_all_shots))), ICT_ch4.charge_all_shots, c=cmap3[2], label='ICT4')
    plt.axhline(abs(np.mean(ICT_ch3.charge_all_shots)), ls=':', c=cmap3[0])
    plt.axhline(abs(np.mean(ICT_ch4.charge_all_shots)), ls=':', c=cmap3[2])
    plt.legend(loc='lower right', fancybox=False, framealpha=0.7, edgecolor='k')
    plt.xlabel('Shot#', fontsize=13)
    plt.ylabel('Charge (nC)', fontsize=13)
    plt.grid(ls=":", alpha=0.5)
    plt.title('Data acquried from '+ time_ss[0].split(' ')[1] + ' to ' + time_ss[-1])
    plt.show()
#     print(time[0])


#     for i in range(len(for_data)):
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#         axes[0].plot(x_fs, for_data[i], label='forward', c=cmap3[0])
#         axes[1].plot(x_fs, ref_data[i], label='reflect', c=cmap3[2])
#         axes[2].plot(x_fs, fc_data[i], label='faraday cup', c=cmap3[4])
#
#         plt.suptitle(f'Shot#{i+1}/{len(for_data)}\nRF data acquired @ ' + time[i])
#         axes[0].set_xlabel('Time (ns)', fontsize=13)
#         axes[1].set_xlabel('Time (ns)', fontsize=13)
#         axes[2].set_xlabel('Time (ns)', fontsize=13)
#         axes[0].set_ylabel('Signal (V)', fontsize=13)
#         axes[0].legend(loc='lower right', fancybox=False, framealpha=0.7, edgecolor='k')
#         axes[1].legend(loc='lower right', fancybox=False, framealpha=0.7, edgecolor='k')
#         axes[2].legend(loc='lower right', fancybox=False, framealpha=0.7, edgecolor='k')
#         plt.tight_layout()
#         plt.show()

def process_diode_data(data):
    results = []
    for i in range(len(data)):
        results.append(max(data[i])-min(data[i]))
    return results


def plot_data():
    db_path = os.getcwd()
    db = '/2022_10_05.sqlite3'
    conn = connect_db(db_path + db)

    id = 38
    with conn:
        for_data, ref_data, fc_data, time, xincr, NR_Pt = select_fs_task_id(conn, task_id=id, for_ch='Ch2', ref_ch='Ch3', fc_ch='Ch4')
        ict3_data, ict4_data, time_ss, xincr_ss, NR_Pt_ss, diode_data = select_ss_task_id(conn, task_id=id, ict3_ch='Ch1', ict4_ch='Ch2', diode_ch='Ch3')

    x_fs = [i * xincr*1e9 for i in list(range(NR_Pt))]
    x_ss = [i * xincr_ss for i in list(range(NR_Pt_ss))]

    for i in range(len(ict3_data)):
        ict3_data[i] = np.array(ict3_data[i]) * 10 ** (20 / 20)
        ict4_data[i] = np.array(ict4_data[i]) * 10 ** (20 / 20)

    ICT_ch3 = CalICTCharge(time_ss, x_ss, ict3_data)
    ICT_ch4 = CalICTCharge(time_ss, x_ss, ict4_data)

    shot_num = 44
    plt.figure(1)
    t_env_ref, env_ref = cal_envolope(ref_data[shot_num], xincr, NR_Pt)
    t_pre, pre, Y1_con = cal_convolution(for_data[shot_num], xincr, NR_Pt)

    plt.plot(x_fs, ref_data[shot_num], c='navy', alpha=0.3)
    plt.plot(t_env_ref, env_ref, c='navy', label='Measured reflection', zorder=20)

    t_env_for, env_for = cal_envolope(for_data[shot_num], xincr, NR_Pt)
    plt.plot(x_fs, for_data[shot_num], c=cmap3[2], alpha=0.4)
    plt.plot(t_env_for, env_for, c=cmap3[3], label='forward raw')

    plt.plot(np.array(t_pre)+5, Y1_con, c=cmap3[5], alpha=0.3)
    plt.plot(np.array(t_pre)+5, pre, c=cmap3[5], label='Predicted reflection')

    plt.title(time[shot_num] + '\n' + "ICT3 Charge@" + str(ICT_ch3.charge_all_shots[shot_num]) + " nC")
    plt.legend(loc='lower right', fancybox=False, framealpha=0.7, edgecolor='k')
    plt.xlim(125, 180)
    plt.xlim(120, 175)
#     plt.xlim(142, 168)
    plt.ylabel("RF signal (V) ", fontsize=15)
    plt.xlabel("Time (ns) ", fontsize=15)
    plt.grid(ls=':', alpha=0.5)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.annotate(f'ratio={np.max(env_ref)/np.max(Y1_con): .3f}', (160, 0.5), ha='center', fontsize='13', color="black",
                    annotation_clip=False)
    plt.tick_params(labelsize=11)
    plt.tight_layout()

#     plt.figure(2)
#     plt.title("FC signal " + time_ss[shot_num] + '\n' + "ICT3 Charge@" + str(ICT_ch3.charge_all_shots[shot_num]) + " nC")
#     plt.plot(np.array(x_ss)*1e9, diode_data[shot_num])
#     plt.xlabel("Time (ns) ", fontsize=15)
#     plt.ylim(-3, 3)
#     plt.grid(ls=':', alpha=0.5)
#     plt.figure(2)
#     plt.scatter(list(range(len(ICT_ch3.charge_all_shots))), ICT_ch3.charge_all_shots, c=cmap3[0], label='ICT3')
#     plt.scatter(list(range(len(ICT_ch3.charge_all_shots))), ICT_ch4.charge_all_shots, c=cmap3[2], label='ICT4')

    plt.show()

def plot_diode_data():
    db_path = os.getcwd()
    db = '/2022_10_05.sqlite3'
    conn = connect_db(db_path + db)

    id_ls = [11,12,16,15,17,18,19,20]
    c = 0
    with conn:
        for id in id_ls:
            for_data, ref_data, fc_data, time, xincr, NR_Pt = select_fs_task_id(conn, task_id=id, for_ch='Ch2', ref_ch='Ch3', fc_ch='Ch4')
            ict3_data, ict4_data, time_ss, xincr_ss, NR_Pt_ss, diode_data = select_ss_task_id(conn, task_id=id, ict3_ch='Ch1', ict4_ch='Ch2', diode_ch='Ch4')

            x_fs = [i * xincr*1e9 for i in list(range(NR_Pt))]
            x_ss = [i * xincr_ss for i in list(range(NR_Pt_ss))]

            for i in range(len(ict3_data)):
                ict3_data[i] = np.array(ict3_data[i]) * 10 ** (10 / 20)
                ict4_data[i] = np.array(ict4_data[i]) * 10 ** (10 / 20)

            ICT_ch3 = CalICTCharge(time_ss, x_ss, ict3_data)

            diode = process_diode_data(diode_data)

            plt.scatter(ICT_ch3.charge_all_shots, diode, c=cmap3[c], label="Charge:" + str(format(np.mean(ICT_ch3.charge_all_shots), '.1f')) + ' nC')
            plt.legend(loc='lower right', fancybox=False, framealpha=0.7, edgecolor='k')
            plt.xlabel("ICT3 Charge (nC) ", fontsize=13)
            plt.ylabel("Diode signal max-min (a.u.) ", fontsize=13)
            c+=1
    plt.show()

if __name__ == '__main__':
    cmap3 = sns.color_palette("Paired", 8)
#     main()
    plot_data()
#     plot_diode_data()