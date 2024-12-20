#!/usr/bin/env python3

import numpy as np
import pyrpl

def characterise_transfer_function(data, frequencies, condition):
    idxs = np.where(condition)
    fminidx_hi = idxs[0].min()
    fminidx_lo = fminidx_hi - 1 if 0 < fminidx_hi else fminidx_hi

    return fminidx_lo, fminidx_hi

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'filters.resonance'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False)
    na = p.networkanalyzer
    iir = p.rp.iir

    start_freq = 1e3
    stop_freq = 2e7

    # setup a simple iir transfer function
    lowpass = True
    highpass = True

    gain_adjust = 3.694144e+19

    gain = 1.0

    fc = 2e4
    loops = 3
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

    zero = -1e-8 / fc

    angle = 91
    pole = complex( np.cos(angle * np.pi / 180),
                    np.sin(angle * np.pi / 180) )

    zeros = np.append( zeros, zero )
    poles = np.append( poles, pole )

    zeros = fc * zeros
    poles = fc * poles

    gain = gain / gain_adjust

    print('gain: {:0.6e} gain_adjust: {:0.6e}'.format(gain, gain_adjust))
    iir.setup(zeros = zeros, poles = poles, gain = gain,
              loops = loops,
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
    tf_abs = np.abs(tf)
    amaxidx = np.argmax(tf_abs)
    dbs = 20 * np.log10(tf_abs)
    dbs_max = dbs.max()

    idx_stable = amaxidx - 150 if 150 < amaxidx else amaxidx
    phases_stable = phases[idx_stable:]

    fmin_phase_rolloff_idx, fmax_phase_rolloff_idx = characterise_transfer_function(
        phases_stable, na.frequencies[idx_stable:], phases_stable < phases_stable[0] - 1.0)
    fmin_phase_rolloff_idx = fmin_phase_rolloff_idx + idx_stable
    fmax_phase_rolloff_idx = fmax_phase_rolloff_idx + idx_stable

    print('filter gain: {:0.3f} db at rolloff frequency {:0.3f} idx: {}'.format(
        dbs[fmin_phase_rolloff_idx], na.frequencies[fmin_phase_rolloff_idx], fmin_phase_rolloff_idx))
    print('peak at: {:0.3f} idx: {}'.format(na.frequencies[amaxidx], amaxidx))
    if lowpass:
        frequencies = na.frequencies[amaxidx:]
        phases_lo = phases[amaxidx:]
        idx_stable = amaxidx - fmin_phase_rolloff_idx
        idx_stable = idx_stable if 0 < idx_stable else amaxidx + 150
        idx_stable = idx_stable if phases_lo.shape[0] < idx_stable else phases_lo.shape[0] - 2
        dbs_lo = dbs[amaxidx:]
        fminpidx, fmaxpidx = characterise_transfer_function(
            phases_lo, frequencies, phases_lo < phases_lo[idx_stable] + 45)
        fmindidx, fmaxdidx = characterise_transfer_function(
            dbs_lo, frequencies, dbs_lo < dbs[fmin_phase_rolloff_idx])
        print('low pass calc from phase then from amplitude: gain: {:0.6e} tf_max: {:0.6f} updated gain_adjust: {:0.6e}\nf between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            iir.gain,
            tf_abs[fmindidx],
            gain_adjust * tf_abs[fmindidx],
            frequencies[fminpidx], frequencies[fmaxpidx],
            dbs_lo[fminpidx], dbs_lo[fmaxpidx],
            phases_lo[fminpidx], phases_lo[fmaxpidx]))
        print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            frequencies[fmindidx], frequencies[fmaxdidx],
            dbs_lo[fmindidx], dbs_lo[fmaxdidx],
            phases_lo[fmindidx], phases_lo[fmaxdidx]))
    if highpass:
        fminpidx, fmaxpidx = characterise_transfer_function(
            phases, na.frequencies, phases < phases_stable[0] - 45)
        fmindidx, fmaxdidx = characterise_transfer_function(
            dbs, na.frequencies, dbs[fmin_phase_rolloff_idx] < dbs)
        print('high pass calc from phase then from amplitude gain: {:0.6e} tf_max: {:0.6f} updated gain_adjust: {:0.6e}\nf between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            iir.gain,
            tf_abs[fmaxdidx],
            gain_adjust * tf_abs[fmaxdidx],
            na.frequencies[fminpidx], na.frequencies[fmaxpidx],
            dbs[fminpidx], dbs[fmaxpidx],
            phases[fminpidx], phases[fmaxpidx]))
        print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            na.frequencies[fmindidx], na.frequencies[fmaxdidx],
            dbs[fmindidx], dbs[fmaxdidx],
            phases[fmindidx], phases[fmaxdidx]))

    if lowpass and highpass:
        print('resonance width: {:0.0f}'.format(
            frequencies[fmaxdidx] - na.frequencies[fmindidx]))

    # plot the design data and the measured response
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (10, 6)

    from pyrpl.hardware_modules.iir.iir_theory import bodeplot
    bodeplot([(na.frequencies, designdata, 'designed system'),
              (na.frequencies, tf, 'measured system')], xlog=True)
