import pyvisa as visa
from datetime import datetime
import numpy as np
import struct
import time


EOL = '\n'
def Tektronix(port_address):
    """
    Scope configuration and setup
    :param port_address:
    :return:
    """
    try:
        rm = visa.ResourceManager()
        scope = rm.open_resource(port_address)
#         print(scope.query("*IDN?"))
        scope.timeout = 20000              # scope timeout [ms]
        return scope

    except visa.errors.VisaIOError:
        print(f'Scope ({port_address}) connection error!')
        return 0


def get_WFM_preamble(port_address, channel_list):
    """
    Get all channel waveform preambles
    :return:
    """
    scope = Tektronix(port_address)
    preamble_ch_all = {key: None for key in channel_list}

    # get time parameters (x parameters)
    try:
        XINcr = float(scope.query('WFMPRE:XINCR?'))         # Returns the horizontal sampling interval
        XZEro = float(scope.query('WFMPre:XZEro?'))         # Returns the time of first points in a waveform

        # get number of points
#         NR_Pt = int(scope.query('WFMPre:NR_Pt?'))             # Query the number of pts in the wfm (for TDS-like cheap scope)
        NR_Pt = int(scope.query('HORizontal:acqlength?'))   # Query the number of pts in the wfm
        for i in channel_list:
            channel_pream_key = ['YMUlt', 'YZEro', 'YOFf', 'XINcr', 'XZEro', 'NR_Pt', 'CH_SCAle']
            ch_pream = {key: None for key in channel_pream_key}

            scope.write("DATA:SOURCE " + i)

        #   get voltage parameters (y related parameter)
            YMUlt = float(scope.query('WFMPRE:YMULT?'))     # tektronix MSO Returns the vertical scale factor
            YZEro = float(scope.query('WFMPRE:YZERO?'))     # Returns the offset voltage
            YOFf = float(scope.query('WFMPRE:YOFF?'))       # Returns the vertical position
    #         YUNit = scope.query('WFMPre:YUNit?')            # Returns the vertical units

            # get y scale
            CH_SCAle = str(scope.query(f'{i}:SCAle?'))      # Returns the vertical scale

            temp_pream = [YMUlt, YZEro, YOFf, XINcr, XZEro, NR_Pt, CH_SCAle]
            ch_pream_temp = dict(zip(ch_pream, temp_pream))
            preamble_ch_all[i] = ch_pream_temp
        print(preamble_ch_all)
        return preamble_ch_all
    except:
        print(f'Scope ({port_address}) connection error!')


def cal_WFM(data_voltage: list, WFM_param: dict):
    YZEro = WFM_param['YZEro']
    YMUlt = WFM_param['YMUlt']
    YOFf = WFM_param['YOFf']
    WFM_voltage = [YZEro + (YMUlt * (i - YOFf)) for i in data_voltage]   # unit [V]

    return WFM_voltage


def get_WFM(port_address, channel_list, ch_param):
    print(f'     Event      |       Time     ')
    print(f'--------------- | ---------------')
    print(f'scope start     | {datetime.now().time()}')
    scope = Tektronix(port_address)
    wfm_all_ch = {key: [] for key in channel_list}
    try:
        pream_keys = list(ch_param.keys())
        xincr = ch_param['CH1']['XINcr']
        hori_len = ch_param['CH1']['NR_Pt']
        ch_scale = {}
        for i in pream_keys:
            ch_scale[i] = ch_param[i]['CH_SCAle']

        for channel in channel_list:
            scope.write('DATA:WIDTH 1')
            scope.write('DATA:ENCDG RIBin')
            scope.write('DATA:START 1')         # Set start index of measurement buffer data
            scope.write(f"DATA:STOP {hori_len}")
            scope.write("DATA:SOURCE " + channel)

#             scope.write('DATA:ENCDG Ascii')
    #         data = str(scope.query('CURVe?'))     # ascii wfm
    #         data = data.split(',')
    #         data[-1] = data[-1].split('\n')[0]
    #         data = [float(i) for i in data]

            data_bin = scope.query_binary_values('CURVe?', datatype='b', container=np.array)
            data = np.array(data_bin, dtype='double') # data decoding, type conversion

            wfm = cal_WFM(data, ch_param[channel])
            if 'CH1' in channel_list and channel == 'CH1':
                wfm_all_ch['CH1'].append(wfm)
            elif 'CH2' in channel_list and channel == 'CH2':
                wfm_all_ch['CH2'].append(wfm)
            elif 'CH3' in channel_list and channel == 'CH3':
                wfm_all_ch['CH3'].append(wfm)
            elif 'CH4' in channel_list and channel == 'CH4':
                wfm_all_ch['CH4'].append(wfm)
        print(f'Scope finished  | {datetime.now().time()}')
        print(f'--------------- | ---------------')
        return wfm_all_ch, xincr, hori_len, ch_scale
    except:
        print('Scope communication error!')
        wfm_all_ch_fake = {'CH1': [[0, 0]], 'CH2': [[0, 0]], 'CH3': [[0, 0]], 'CH4': [[0, 0]]}
        return wfm_all_ch_fake, 0, 0, {'CH1': 0, 'CH2': 0, 'CH3': 0, 'CH4': 0}


def preset_scope(fs_address, ss_address, fs_preambles, ss_preambles, fs_ch_ls, ss_ch_ls):

    print(f'     Event      |       Time     ')
    print(f'--------------- | ---------------')
    print(f'scope start     | {datetime.now().time()}')
    fs = Tektronix(fs_address)
    ss = Tektronix(ss_address)
    fs_channel_ls_join = ','.join(fs_ch_ls)
    ss_channel_ls_join = ','.join(ss_ch_ls)

    if fs != 0:
        fs.write("DATA:SOURCE " + fs_channel_ls_join)
    if ss != 0:
        ss.write("DATA:SOURCE " + ss_channel_ls_join)

    if fs != 0:
        fs_hori_len = fs_preambles['CH1']['NR_Pt']
        fs.write('DATA:WIDTH 1')
        fs.write('DATA:ENCDG RIBin')
        fs.write('DATA:START 1')         # Set start index of measurement buffer data
        fs.write(f"DATA:STOP {fs_hori_len}")

    if ss != 0:
        ss_hori_len = ss_preambles['CH1']['NR_Pt']
        ss.write('DATA:WIDTH 1')
        ss.write('DATA:ENCDG RIbinary')
        ss.write('DATA:START 1')         # Set start index of measurement buffer data
        ss.write(f"DATA:STOP {ss_hori_len}")

    print('fs and ss preset done.')
    if fs != 0 and ss != 0:
        return fs, ss, fs_hori_len, ss_hori_len
    elif fs != 0 and ss == 0:
        return fs, ss, fs_hori_len, 0
    elif fs == 0 and ss != 0:
        return fs, ss, 0, ss_hori_len
    else:
        print('No scope available.')


def pull_wfm_all_ch(fs_address, ss_address, fs_channel_ls, ss_channel_ls, fs_hori_len, ss_hori_len):
    fs_flag = 1
    ss_flag = 1
    if fs_hori_len != 0:
        fs_address.write('CURVEnext?')
    else:
        fs_flag = 0
        print('Pulling data from FS error!')

    if ss_hori_len != 0:
        # ss_address.write('CURVE?')
        ss_address.write('CURVEnext?')      # use CURVEnext for the second scope to avoid mis-matching issue between two scopes.
    else:
        ss_flag = 0
        print('Pulling data from SS error!')
        
    if fs_flag == 1:
        fs_bin = fs_address.read_raw()
    if ss_flag == 1:
        ss_bin = ss_address.read_raw()        

    fs_wfm_raw = {}
    ss_wfm_raw = {}

    try:
        n = int(len(fs_channel_ls))
        m = int((len(fs_bin)-1)/n)
        fs_bin_recons = []
        for i in range(n):
            fs_bin_recons.append(fs_bin[i*m:(i+1)*m])

        fullfmt_fs = "%s%d%s" % ('<', fs_hori_len, 'b')
        start_fs = m - fs_hori_len
        for i in range(n):
            fs_wfm_raw[fs_channel_ls[i]] = struct.unpack_from(fullfmt_fs, fs_bin_recons[i][start_fs:], 0)
    except:
        print('Check fs bin sorting')

    try:
        n = int(len(ss_channel_ls))
        m = int((len(ss_bin)-1)/n)
        ss_bin_recons = []
        for i in range(n):
            ss_bin_recons.append(ss_bin[i*m:(i+1)*m])

        fullfmt_ss = "%s%d%s" % ('<', ss_hori_len, 'b')
        start_ss = m - ss_hori_len
        for i in range(n):
            ss_wfm_raw[ss_channel_ls[i]] = struct.unpack_from(fullfmt_ss, ss_bin_recons[i][start_ss:], 0)
    except:
        print('Check ss bin sorting')

    return fs_wfm_raw, ss_wfm_raw


def get_WFM_advance(raw_wfm, channel_list, ch_param):
    wfm_all_ch = {key: [] for key in channel_list}
    try:
        pream_keys = list(ch_param.keys())
        xincr = ch_param['CH1']['XINcr']
        hori_len = ch_param['CH1']['NR_Pt']
        ch_scale = {}
        for i in pream_keys:
            ch_scale[i] = ch_param[i]['CH_SCAle']

        for i in range(len(channel_list)):
            channel = 'CH' + str(i+1)
            wfm = cal_WFM(raw_wfm[channel], ch_param[channel])
            wfm_all_ch[channel].append(wfm)
        return wfm_all_ch, xincr, hori_len, ch_scale
    except:
        # print('Scope communication error!')
        wfm_all_ch_fake = {'CH1': [[0, 0]], 'CH2': [[0, 0]], 'CH3': [[0, 0]], 'CH4': [[0, 0]]}
        return wfm_all_ch_fake, 0, 0, {'CH1': 0, 'CH2': 0, 'CH3': 0, 'CH4': 0}

