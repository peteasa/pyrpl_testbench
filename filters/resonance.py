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

    # setup a simple iir transfer function
    lowpass = False
    highpass = True
    gain = 5.5e-20

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
    factor = fc

    print('gain: {:0.1e} filter cutoff frequency: {:0.3f}'.format(gain, fc))

    zeros = np.array([ ], dtype = np.complex128)
    poles = np.array([ ], dtype = np.complex128)

    zero = -1e-8 / factor
    angle = 91
    pole = complex( np.cos(angle * np.pi / 180),
                    np.sin(angle * np.pi / 180) )
    #pole = complex( -1.0e-10 / factor, 1.0 )
    zeros = np.append( zeros, zero )
    poles = np.append( poles, pole )

    zeros = factor * zeros
    poles = factor * poles

    iir.setup(zeros = zeros, poles = poles, gain = gain,
              loops = loops,
              input = na.iq,
              output_direct = 'off')

    print('poles: {}'.format([po / factor for po in iir.poles]))
    print('zeros: {}'.format([ze / factor for ze in iir.zeros]))
    print('Filter sampling frequency: {:0.3f} MHz cut off frequency: {:0.6f} MHz'.format(125./iir.loops, fc / 1e6))

    # useful diagnostic functions
    print('IIR on: {}'.format(iir.on))
    print('IIR bypassed: {}'.format(iir.bypass))
    print('IIR loops: {} requested: {}'.format(iir.loops, loops))
    print('IIR overflows: {}'.format(iir.overflow))
    print('Coefficients (6 per biquad)\n{}'.format(iir.coefficients))

    # check if the filter is ok
    print('IIR overflows before: {}'.format(iir.overflow))

    # setup the network analyzer to measure tf of iir filter
    stop_freq = 5e7
    rbw = 1000
    points = 1001
    na.setup(start_freq = 3e3, stop_freq = stop_freq, rbw = rbw,
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

    if lowpass:
        phases = np.angle(tf, deg = True)
        phases = phases - phases[0]
        fminpidx, fmaxpidx = characterise_transfer_function(phases, na.frequencies, phases < -45)
        dbs = 20 * np.log10(np.abs(tf))
        dbs = dbs - dbs.max()
        fmindidx, fmaxdidx = characterise_transfer_function(dbs, na.frequencies, dbs < -3)
        print('low pass\nfrom phase f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            na.frequencies[fminpidx], na.frequencies[fmaxpidx],
            dbs[fminpidx], dbs[fmaxpidx],
            phases[fminpidx], phases[fmaxpidx]))
        print('from db f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            na.frequencies[fmindidx], na.frequencies[fmaxdidx],
            dbs[fmindidx], dbs[fmaxdidx],
            phases[fmindidx], phases[fmaxdidx]))
    if highpass:
        phases = np.angle(tf, deg = True)
        phases = phases - phases[0]
        fminpidx, fmaxpidx = characterise_transfer_function(phases, na.frequencies, phases < -45)
        dbs = 20 * np.log10(np.abs(tf))
        dbs = dbs - dbs.max()
        fmindidx, fmaxdidx = characterise_transfer_function(dbs, na.frequencies, -3 < dbs)
        print('high pass\nfrom phase f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            na.frequencies[fminpidx], na.frequencies[fmaxpidx],
            dbs[fminpidx], dbs[fmaxpidx],
            phases[fminpidx], phases[fmaxpidx]))
        print('from db f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
            na.frequencies[fmindidx], na.frequencies[fmaxdidx],
            dbs[fmindidx], dbs[fmaxdidx],
            phases[fmindidx], phases[fmaxdidx]))

    # plot the design data and the measured response
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (10, 6)

    from pyrpl.hardware_modules.iir.iir_theory import bodeplot
    bodeplot([(na.frequencies, designdata, 'designed system'),
              (na.frequencies, tf, 'measured system')], xlog=True)
