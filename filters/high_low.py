#!/usr/bin/env python3

import numpy as np
import pyrpl

def loop_sample_freq_list(start, end, count, logdiff, loops, samples = None):
    if samples is None:
        samples = round(125e6 / (loops * start))

    freqs = np.array([[],[],[]], dtype = np.float64)
    for i in range(count):
        fc = 125e6 / (loops * samples)
        if end < fc:
            break

        freqs = np.append(freqs, [[loops], [samples], [fc]], axis = 1)
        osamples = samples
        samples = round(125e6 / (loops * 10 ** (np.log10(fc) + logdiff)))
        if samples < 2:
            break

        if samples == osamples:
            samples = samples - 1

    return freqs

def characterise_transfer_function(data, frequencies, condition):
    idxs = np.where(condition)
    fminidx_hi = idxs[0].min()
    fminidx_lo = fminidx_hi - 1 if 0 < fminidx_hi else fminidx_hi

    return fminidx_lo, fminidx_hi

def test_loop_sample_freq_list():
    count = 40
    logdiff = 0.5
    loops = 8
    start = 3e4
    end = 5e7
    freqs = loop_sample_freq_list(start, end, count, logdiff, loops)
    print(freqs.shape, freqs[0,0], freqs[1,0], freqs[2,0], freqs[0,-1], freqs[1,-1], freqs[2,-1])

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'filters.high_low'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False)
    na = p.networkanalyzer
    iir = p.rp.iir

    start_freq = 1e3
    stop_freq = 2e7

    highpass = True
    lowpass = False

    gain_adjust_lp = 3.2509426667e+07
    gain_adjust_hp = 9.840e+05
    gain_adjust_twin = 4.982265e-01

    gain = 1.0

    loops = 4

    fc = 1.5e5

    samples = round(125e6 / (loops * fc))
    fc = 125e6 / (loops * samples)

    # reduce the number of loops to increase the sampling rate
    loops = round(125e6 / (samples * fc))
    print('loops: {:0.3f} samples: {:0.0f}'.format(loops, samples))
    if 1023 < loops:
        loops = 1023
        print('frequency / time constant not possible - adjusted')
    if loops < 3:
        loops = 3
        print('frequency / time constant not possible - adjusted')

    fs = 125e6 / loops
    fc = 125e6 / (loops * samples)
    TC = 1 / (2 * np.pi * fc)
    omegac = 2 * np.pi * fc

    print('filter cutoff frequency: {:0.3f} omega_c: {:0.3f} time constant: {:0.3f} us'.format(fc, 2 * np.pi * fc, TC * 1e6))

    zeros = np.array([ ], dtype = np.complex128)
    if highpass:
        zeros = np.array([ fc * complex( -0.000001, 0 ) ], dtype = np.complex128)

    if highpass and lowpass:
        lmulti = 1.1 ; hmulti = 0.9
        poles = np.array([ fc * lmulti * complex( -1.0, 0 ), fc * hmulti * complex( -1.0, 0 ) ], dtype = np.complex128)
        gain_adjust = gain_adjust_lp * gain_adjust_hp * gain_adjust_twin
        gain = gain / gain_adjust
    elif lowpass:
        poles = np.array([ fc * complex( -1.0, 0 ) ], dtype = np.complex128)
        gain_adjust = gain_adjust_lp
        gain = gain / gain_adjust
    elif highpass:

        poles = np.array([ fc * complex( -1.0, 0 ) ], dtype = np.complex128)
        gain_adjust = gain_adjust_hp
        gain = gain / gain_adjust

    print('gain: {:0.6e} gain_adjust: {:0.6e}'.format(gain, gain_adjust))
    iir.setup(gain = gain,
              zeros=zeros, poles=poles,
              loops=loops)

    print('poles: {}'.format([po for po in iir.poles]))
    print('zeros: {}'.format([ze for ze in iir.zeros]))
    print('Filter sampling frequency: {:0.3f} MHz cut off frequency: {:0.6f} MHz'.format(125./iir.loops, fc / 1e6))
    print('poles: {}'.format([po / fc for po in iir.poles]))
    print('zeros: {}'.format([ze / fc for ze in iir.zeros]))

    # useful diagnostic functions
    print('IIR on: {}'.format(iir.on))
    print('IIR bypassed: {}'.format(iir.bypass))
    print('IIR loops: {} requested: {}'.format(iir.loops, loops))
    print('IIR overflows: {}'.format(iir.overflow))
    print('Coefficients (6 per biquad)\n{}'.format(iir.coefficients))

    # check if the filter is ok
    print('IIR overflows before: {}'.format(iir.overflow))

    # setup the network analyzer to measure tf of iir filter
    rbw = 1000
    points = 1001
    iir.input = na.iq
    na.setup(start_freq = start_freq, stop_freq = stop_freq, rbw = rbw,
             points = points,
             average_per_point = 1,
             amplitude = 0.1,
             input = 'iir', output_direct = 'off',
             logscale=True)

    tf = na.single()

    # check if the filter is still ok after measuring the transfer function
    print('IIR overflows after: {}'.format(iir.overflow))

    # retrieve designed transfer function
    designdata = iir.transfer_function(na.frequencies)

    # now make some measurements from the transfer function
    print('================= Measurements ======================')
    phases = np.angle(tf, deg = True)
    tf_abs = np.abs(tf)
    tf_max = tf_abs.max()
    amaxidx = np.argmax(tf_abs)
    dbs = 20 * np.log10(tf_abs)
    dbs_max = dbs.max()
    print('filter gain: {:0.3f} db'.format(dbs_max))
    if lowpass:
        if highpass:
            print('peak at: {:0.0f}'.format(na.frequencies[amaxidx]))

        phases_lp = phases[amaxidx:] if highpass else phases
        dbs_lp = dbs[amaxidx:] if highpass else dbs
        frequencies = na.frequencies[amaxidx:] if highpass else na.frequencies
        fminpidx, fmaxpidx = characterise_transfer_function(phases_lp, frequencies, phases_lp < phases_lp[0] - 45)
        fmindidx, fmaxdidx = characterise_transfer_function(dbs_lp, frequencies, dbs_lp < dbs_max - 3)
        print('low pass calc from phase then from amplitude: gain: {:0.6e} tf_max: {:0.6f} updated gain_adjust: {:0.6e}\nf between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            iir.gain,
            tf_max,
            gain_adjust * tf_max,
            na.frequencies[amaxidx + fminpidx], na.frequencies[amaxidx + fmaxpidx],
            dbs[amaxidx + fminpidx], dbs[amaxidx + fmaxpidx],
            phases[amaxidx + fminpidx], phases[amaxidx + fmaxpidx]))
        print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            na.frequencies[amaxidx + fmindidx], na.frequencies[amaxidx + fmaxdidx],
            dbs[amaxidx + fmindidx], dbs[amaxidx + fmaxdidx],
            phases[amaxidx + fmindidx], phases[amaxidx + fmaxdidx]))
    if highpass:
        fminpidx, fmaxpidx = characterise_transfer_function(phases, na.frequencies, phases < phases[0] - 45)
        fmindidx, fmaxdidx = characterise_transfer_function(dbs, na.frequencies, dbs_max - 3 < dbs)
        print('high pass calc from phase then from amplitude gain: {:0.6e} tf_max: {:0.6f} updated gain_adjust: {:0.6e}\nf between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            iir.gain,
            tf_max,
            gain_adjust * tf_max,
            na.frequencies[fminpidx], na.frequencies[fmaxpidx],
            dbs[fminpidx], dbs[fmaxpidx],
            phases[fminpidx], phases[fmaxpidx]))
        print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            na.frequencies[fmindidx], na.frequencies[fmaxdidx],
            dbs[fmindidx], dbs[fmaxdidx],
            phases[fmindidx], phases[fmaxdidx]))

    # plot the design data and the measured response
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (10, 6)

    from pyrpl.hardware_modules.iir.iir_theory import bodeplot
    bodeplot([(na.frequencies, designdata, 'designed system'),
              (na.frequencies, tf, 'measured system')],
             title = '', pause = False, xlog=True)
