#!/usr/bin/env python3

import numpy as np
import pyrpl

def characterise_transfer_function(data, frequencies, condition):
    idxs = np.where(condition)
    fminidx_hi = idxs[0].min()
    fminidx_lo = fminidx_hi - 1 if 0 < fminidx_hi else fminidx_hi

    return fminidx_lo, fminidx_hi

def omegac_sin(omegac, angle, delta = 0):
    val = np.sin((angle+delta) * np.pi / 180)
    if abs(omegac * val) < 1.0:
        val = 0

    return val

def omegac_cosz(omegac, angle, delta = 0):
    val = np.cos((angle+delta) * np.pi / 180)
    if abs(omegac * val) < 1.0:
        val = - 0.0001 / omegac

    return val

def omegac_cos(omegac, angle, delta = 0):
    val = np.cos((angle+delta) * np.pi / 180)
    if abs(omegac * val) < 1.0:
        val = 0

    return val

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'filters.notch'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False)
    na = p.networkanalyzer
    iir = p.rp.iir

    start_freq = 1e3
    stop_freq = 2e7

    gain_adjust = 1.0

    gain = 1.0

    loops = 4

    fc = 5.0e4

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
    poles = np.array([ ], dtype = np.complex128)

    angles = [90]
    for idx_z, angle in enumerate(angles):
        angle = angles[idx_z]
        zero = fc * complex( omegac_cosz(omegac, angle),
                                         omegac_sin(omegac, angle) )
        zeros = np.append(zeros, zero)

    angles = [179.99]
    for idx_p, angle in enumerate(angles):
        angle = angles[idx_p]
        pole = fc * complex( omegac_cosz(omegac, angle),
                                         omegac_sin(omegac, angle) )
        poles = np.append(poles, pole)

    print('gain: {:0.6e} gain_adjust: {:0.6e}'.format(gain, gain_adjust))
    iir.setup(gain = gain,
              zeros=zeros, poles=poles,
              loops=loops,
              output_direct = 'off')

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
             running_state = 'stopped',
             trace_average = 1,
             amplitude = 0.005,
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
    fminpidx, fmaxpidx = characterise_transfer_function(phases, na.frequencies, phases < -45)
    tf_abs = np.abs(tf)
    tf_max = tf_abs.max()
    aminidx = np.argmax(-tf_abs)
    dbs = 20 * np.log10(tf_abs)
    dbs_max = dbs.max()
    print('filter gain: {:0.3f} db'.format(dbs_max))
    print('notch frequency: {:0.0f} dbs: {:0.3f} phase: {:0.3f}'.format(na.frequencies[aminidx], dbs[aminidx], phases[aminidx]))
    fmindidx, fmaxdidx = characterise_transfer_function(dbs, na.frequencies, dbs < dbs_max - 3)
    print('low pass calc from phase then from amplitude: gain: {:0.6e} tf_max: {:0.6f} updated gain_adjust: {:0.6e}\nf between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
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

    # Now shift the attention to the second half of the filter output
    phases = phases[aminidx:]
    frequencies = na.frequencies[aminidx:]
    tf_abs = tf_abs[aminidx:]
    dbs = dbs[aminidx:]

    fminpidx, fmaxpidx = characterise_transfer_function(phases, frequencies, phases < 45)
    tf_max = tf_abs.max()
    dbs = dbs
    fmindidx, fmaxdidx = characterise_transfer_function(dbs, frequencies, dbs_max - 3 < dbs)
    print('high pass calc from phase then from amplitude gain: {:0.6e} tf_max: {:0.6f} updated gain_adjust: {:0.6e}\nf between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
        iir.gain,
        tf_max,
        gain_adjust * tf_max,
        frequencies[fminpidx], frequencies[fmaxpidx],
        dbs[fminpidx], dbs[fmaxpidx],
        phases[fminpidx], phases[fmaxpidx]))
    print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
        frequencies[fmindidx], frequencies[fmaxdidx],
        dbs[fmindidx], dbs[fmaxdidx],
        phases[fmindidx], phases[fmaxdidx]))

    print('notch width: {:0.0f}'.format(frequencies[fmaxdidx] - na.frequencies[fmindidx]))

    # plot the design data and the measured response
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (10, 6)

    from pyrpl.hardware_modules.iir.iir_theory import bodeplot
    bodeplot([(na.frequencies, designdata, 'designed system'),
              (na.frequencies, tf, 'measured system')],
             title = '', pause = False, xlog=True)
