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
    CONFIG = 'basic.filters'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False)
    na = p.networkanalyzer

    with p.iirs.pop('user') as iir:
        unity = False
        lowpass = True
        highpass = False
        stop_freq = 5e6
        gain = 1e-9
        loops = 8
        TC = None # alternative is to specify the time constant: for example 1us
        if TC is None:
            fc = 1.5e5
        elif TC is None:
            fc = 1 / (2 * np.pi * TC)

        samples = round(125e6 / (loops * fc))
        fc = 125e6 / (loops * samples)
        # reduce the number of loops to increase the sampling rate
        # choose a suitable divisor
        sampling_rate_adjustment = 1.0 / samples
        loops = round(sampling_rate_adjustment * 125e6 / fc)
        print('loops: {:0.3f} samples: {:0.0f}'.format(loops, 1.0 / sampling_rate_adjustment))
        if 1023 < loops:
            loops = 1023
            print('frequency / time constant not possible - adjusted')
        if loops < 3:
            loops = 3
            print('frequency / time constant not possible - adjusted')

        fc = sampling_rate_adjustment * 125e6 / loops
        TC = 1 / (2 * np.pi * fc)
        omegac = 1 / TC

        print('filter cutoff frequency: {:0.3f} time constant: {:0.3f} us'.format(fc, TC * 1e6))

        if lowpass:
            # low pass filter at f
            stop_freq = 3e7
            zeros = np.array([], dtype = np.complex128)
            poles = np.array([ omegac * complex( -1, 0 ) ], dtype = np.complex128)
        elif highpass:
            # high pass filter at f
            stop_freq = 3e7
            zeros = np.array([ omegac * complex( -0.00001 / omegac, 0 ) ], dtype = np.complex128)
            poles = np.array([ omegac * complex( -1, 0 ) ], dtype = np.complex128)
        elif False:
            # 3rd order butterworth filter
            stop_freq = 3e7
            nps = 3
            start = 30
            end = 180 - start
            th = (end - start) / (nps - 1)
            angle = start
            poles = np.array([], dtype = np.complex128)
            for p in range(nps):
                print('angle: {}'.format(angle))
                imag = np.cos(angle * np.pi / 180)
                if abs(omegac * imag) < 1.0:
                    imag = 0

                poles = np.append(poles, omegac * complex( - np.sin(angle * np.pi / 180),
                                               imag ) )

                angle = angle + th # +0.000000001

            zeros = np.array([], dtype = np.complex128)
        elif False:
            # notch filter at f
            angle = 68

            p_zero = omegac * complex( 0, 1 )
            n_zero = omegac * complex( 0, -1 )
            p_pole = omegac * complex( -np.cos(angle * np.pi / 180),
                                      np.sin(angle * np.pi / 180) )
            n_pole = omegac * complex( -np.cos(angle * np.pi / 180),
                                      -np.sin((angle + 0.00001) * np.pi / 180) )
            zeros = np.array([ p_zero, n_zero ], dtype = np.complex128)
            poles = np.array([ p_pole, n_pole ], dtype = np.complex128)
        else:
            # band-pass filter at fc
            nps = 7
            loops = 2 * nps * 125e6 / fc
            th = 180 / (nps - 1)
            angle = 0
            pole = omegac * complex( 0, 1 )
            poles = np.array([], dtype = np.complex128)
            for p in range(nps):
                print('angle: {}'.format(angle))
                poles = np.append(poles, pole + omegac * complex( -np.sin(angle * np.pi / 180),
                                                     np.cos(angle * np.pi / 180) ) / 32)

                angle = angle + th

            angle = 0.000001
            pole = omegac * complex( 0, -1 )
            for p in range(nps):
                poles = np.append(poles, pole + omegac * complex( -np.sin(angle * np.pi / 180),
                                                     np.cos(angle * np.pi / 180) ) / 32)

                angle = angle + th

            zeros = np.array([], dtype = np.complex128)
        if unity:
            iir._setup_unity()
        else:
            iir.setup(gain = gain,
                      zeros=zeros, poles=poles,
                      loops=loops)

        print('poles: {}'.format([po / omegac for po in poles]))
        print('zeros: {}'.format([ze / omegac for ze in zeros]))
        print('Filter sampling frequency: {:0.3f} MHz cut off frequency: {} MHz'.format(125./iir.loops, fc / 1e6))

        # useful diagnostic functions
        print('IIR on: {}'.format(iir.on))
        print('IIR bypassed: {}'.format(iir.bypass))
        print('IIR loops: {} requested: {}'.format(iir.loops, loops))
        print('IIR overflows: {}'.format(iir.overflow))
        print('Coefficients (6 per biquad)\n{}'.format(iir.coefficients))

        # check if the filter is ok
        print('IIR overflows before: {}'.format(iir.overflow))

        # measure tf of iir filter
        rbw = 1000
        points = 1001
        iir.input = na.iq
        na.setup(start_freq = 1e4, stop_freq = stop_freq, rbw = rbw,
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

        if lowpass:
            phases = np.angle(tf, deg = True)
            fminpidx, fmaxpidx = characterise_transfer_function(phases, na.frequencies, phases < -45)
            dbs = 20 * np.log10(np.abs(tf))
            dbs = dbs - dbs.max()
            fmindidx, fmaxdidx = characterise_transfer_function(dbs, na.frequencies, dbs < -3)
            print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
                na.frequencies[fminpidx], na.frequencies[fmaxpidx],
                dbs[fminpidx], dbs[fmaxpidx],
                phases[fminpidx], phases[fmaxpidx]))
            print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
                na.frequencies[fmindidx], na.frequencies[fmaxdidx],
                dbs[fmindidx], dbs[fmaxdidx],
                phases[fmindidx], phases[fmaxdidx]))
        elif highpass:
            phases = np.angle(tf, deg = True)
            phases = phases - phases[0]
            fminpidx, fmaxpidx = characterise_transfer_function(phases, na.frequencies, phases < -45)
            dbs = 20 * np.log10(np.abs(tf))
            dbs = dbs - dbs.max()
            fmindidx, fmaxdidx = characterise_transfer_function(dbs, na.frequencies, -3 < dbs)
            print('f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
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
                  (na.frequencies, tf, 'measured system')], xlog=True)
