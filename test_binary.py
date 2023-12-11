import pyvisa as visa
from datetime import datetime
import numpy as np
import struct
import matplotlib.pyplot as plt



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
        print(scope.query("*IDN?"))
        scope.timeout = 100000              # scope timeout [ms]
        return scope

    except visa.errors.VisaIOError:
        print(f'Scope ({port_address}) connection error!')


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
        NR_Pt = int(scope.query('WFMPre:NR_Pt?'))   # Query the number of pts in the curve transfer from the scope
#         NR_Pt = int(scope.query('HORizontal:acqlength?'))   # Query the number of pts in the curve transfer from the scope
        for i in channel_list:
            channel_pream_key = ['YMUlt', 'YZEro', 'YOFf', 'XINcr', 'XZEro', 'NR_Pt', 'CH_SCAle']
            ch_pream = {key: None for key in channel_pream_key}

            scope.write("DATA:SOURCE " + i)

        #   get voltage parameters (y related parameter)
            YMUlt = float(scope.query('WFMPRE:YMULT?'))     # tektronix MSO Returns the vertical scale factor
            YZEro = float(scope.query('WFMPRE:YZERO?'))     # Returns the offset voltage
            YOFf = float(scope.query('WFMPRE:YOFF?'))       # Returns the vertical position

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


def decode_binary_block(block, dtype):
    """ Convert a binary block (as defined by IEEE 488.2), which is commonly returned by lab
    instruments, to a numpy array.
    Args:
        block : binary block bytestring
        dtype : data encoding format e.g. 'float32'
    """
    # The fixed length block is defined by IEEE 488.2 and consists of `#'' (ASCII), one numeric (ASCII) indicating the number of bytes that specifies the length after #, then the length designation (ASCII), and finally the actual binary data of a specified length.
    # First, locate start of block
    start_idx = block.find(b'#')
    # Read header that indicates the data length
    num_bytes_that_specify_length = int(block[start_idx + 1: start_idx + 2])
    data_length = int(block[start_idx + 2: start_idx + 2 + num_bytes_that_specify_length])
    data_start_idx = start_idx + 2 + num_bytes_that_specify_length
    # Finally, slice the relevant data from the block and convert to an array based on the data type
    data = block[data_start_idx:data_start_idx + data_length]
    data = np.fromstring(data, dtype=dtype)
    return data


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
        print(f'in try')
        for i in pream_keys:
            ch_scale[i] = ch_param[i]['CH_SCAle']
            print(f'in for 1-')
        for channel in channel_list:
            scope.write("DATA:WIDTH 1")
            scope.write("DATa:ENCdg RIbinary")
            scope.write('DATA:START 1')         # Set start index of measurement buffer data
            scope.write(f"DATA:STOP {hori_len}")
            scope.write("DATA:SOURCE " + channel)
            print(str(scope.query('DATA?')))
#             print(f'in for 2')
#             scope.write('DATA:ENCDG Ascii')
#             data = str(scope.query('CURVe?'))     # ascii wfm
#             data = data.split(',')
#             data[-1] = data[-1].split('\n')[0]
#             data = [float(i) for i in data]
#
#             print(data)

            scope.write('CURVE?')
            data_bin = scope.read_raw()
            start = len(data_bin) - 10000 - 1
#             data_bin = np.frombuffer(data_bin[start::], dtype='float')

            formatChars = {'1':'b','2':'h'}
            formatChar = formatChars[str('1')]
            endianess = '<'
            array_length = 10000
            datatype = 'b'
            fullfmt = "%s%d%s" % (endianess, array_length, datatype)
            print(fullfmt)
            data_bin = struct.unpack_from(fullfmt, data_bin[start:], 0)
            print(len(data_bin))
#             quit()
#             data_bin = scope.query_binary_values('curve?', datatype='b', container=np.array)

#             print(data_bin)
#             scope.write("CURVE?")
#             data = scope.read_raw()
#             print(np.frombuffer(data[9::], dtype='int16'))

#             headerlen = 2 + int(data[1])
#             header = data[:headerlen]
#             ADC_wave = data[headerlen:-1]
#             ADC_wave = np.fromstring(ADC_wave, dtype=np.int16)
#             data = np.array(data_bin, dtype='double') # data decoding, type conversion
#             header_len = 2 + int(data[1])
#             waveform = np.fromstring(data[header_len:], dtype='>i1')
#             print(ADC_wave)
#             quit()
#             preamble, data = preamble_and_data.split(b';:CURV ')

# data, dtype='>i1')
#             data_bin_fs = scope.read_binary_values(datatype='b', container=np.array)
#             plt.plot(data_bin_fs)
#             plt.show()
#             quit()
#             print(data_bin_fs)
#             data = np.array(data_bin, dtype='double') # data decoding, type conversion

            wfm = cal_WFM(data_bin, ch_param[channel])
            plt.plot(wfm)
            plt.show()
#             quit()
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


if __name__ == '__main__':
    test_scope = Tektronix('TCPIP0::192.168.0.47::inst0::INSTR')
    ls = ['CH1']
    preamble_ch_all = get_WFM_preamble('TCPIP0::192.168.0.47::inst0::INSTR', ls)
    wfm_all_ch, xincr, hori_len, ch_scale = get_WFM('TCPIP0::192.168.0.47::inst0::INSTR', ls, preamble_ch_all)
#     wfm = cal_WFM(wfm_all_ch['CH1'][0], preamble_ch_all['CH1'])
#     wfm1 = cal_WFM(wfm_all_ch['CH2'][0], preamble_ch_all['CH1'])
#     plt.plot(wfm)
#     plt.plot(wfm1)
#     plt.show()