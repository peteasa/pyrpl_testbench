#!/usr/bin/env python3

import numpy as np
import pyrpl
from pyrpl.hardware_modules.iir.iir_theory import bodeplot
import time

def prepare_to_show_plot(p):
    import matplotlib
    for i in range(1):
        try:
            matplotlib.use('TkAgg')
            break
        except:
            #print('{} qt is still running!'.format(10-i))
            pass

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'basic.netanalyser'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False, reloadfpga = True) # False)

    # network analyser is a software module so is accessed directly from pyrpl
    na = p.networkanalyzer

    # reserve the iq module
    na.iq_name = 'iq2'

    # setup network analyzer
    # start_freq / stop_freq: frequency range over which the transfer function is recorded
    # points: number of points in the frequency range
    # rbw: cutoff frequency used in the iq low-pass filter (bandwidth)
    # rbw value is related to avg_per_point and time_per_point

    na.setup(start_freq=1e3, stop_freq=62.5e6, points=1001, rbw=1000,
             average_per_point=1,
             amplitude=0.2, input='iq2', output_direct='off', acbandwidth=0)

    # fetch the curve iq2 -> iq2
    iq2 = na.single()

    # now reconfigure and fetch the curve iq2-> out2 -> connect to -> adc
    external = True
    outconn = 'off'
    inconn = 'networkanalyzer'
    if external:
        outconn = 'out1'
        inconn = 'in1'

    na.setup(input=inconn, output_direct=outconn)
    in1 = na.single()

    prepare_to_show_plot(p)

    # get x-axis for plotting
    f = na.frequencies
    print('acbandwidth: {} rbw: {} time_per_point: {} measured_time_per_point: {}'.format(na.acbandwidth, na.rbw, na.time_per_point, na.measured_time_per_point))
    print('frequencys: {}'.format(f))
    print('in1 chart points: {}'.format(in1))
    num_points = len(f)
    sample_0 = 0
    sample_5 = int(num_points/2)
    sample_25 = int(sample_5/2)
    sample_75 = sample_5 + sample_25
    print('frequency points: 0: {} @100: {} .25: {} .5: {} .75: {}'.format(f[sample_0], f[100], f[sample_25], f[sample_5], f[sample_75]))
    print('in1 chart raw points: 0: {} @100: {} .25: {} .5: {} .75: {}'.format(in1[sample_0], in1[100], in1[sample_25], in1[sample_5], in1[sample_75]))
    print('in1 abs raw values: 0: {} @100: {} .25: {} .5: {} .75: {}'.format(np.abs(in1[sample_0]), np.abs(in1[100]), np.abs(in1[sample_25]), np.abs(in1[sample_5]), np.abs(in1[sample_75])))

    title = 'internal'
    label1 = '{}->{}'.format(na.iq_name, na.iq_name)
    label2 = '{}->{}->{}'.format(na.iq_name, na.input, na.iq_name)
    attenuation = 1.0
    if 'in' in na.input:
        attenuation = 20.0
        title = 'external'
        label2 = '{}->{}->{}->{}'.format(na.iq_name, na.output_direct, na.input, na.iq_name)

    bodeplot([(f, iq2, label1), (f, in1*attenuation, label2)], xlog=True)
