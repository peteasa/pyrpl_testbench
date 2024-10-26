#!/usr/bin/env python3

import numpy as np
import pyrpl

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'basic.filters'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False)
    na = p.networkanalyzer

    with p.iirs.pop('user') as iir:
        unity = False
        stop_freq = 5e6
        gain = 1e-9
        TC = None # alternative is to specify the time constant: for example 1us
        if TC is None:
            # alternative is to specify the cut off frequency: for example 1e6 Hz
            fc = 1.59235e6
        else:
            fc = 1 / (2 * np.pi * TC)

        loops = round(125e6 / fc)
        if 1023 < loops:
            loops = 1023
            print('frequency / time constant not possible - adjusted')
        if loops < 3:
            loops = 3
            print('frequency / time constant not possible - adjusted')

        fc = 125e6 / loops
        TC = 1 / (2 * np.pi * fc)
        omegac = 1 / TC

        print('filter cutoff frequency: {} time constant: {} us'.format(fc, TC * 1e6))

        rbw = 1000
        points = 1001
        if True:
            # low pass filter at f
            stop_freq = 3e7
            zeros = [ ]
            poles = [ omegac * complex( -1, 0 ) ]
            print('poles: {}'.format(poles))
        elif True:
            # high pass filter at f
            stop_freq = 3e7
            zeros = [ omegac * complex( 0, 0 ) ]
            poles = [ omegac * complex( -1, 0 ) ]
        elif False:
            # 3rd order butterworth filter
            stop_freq = 3e7
            nps = 3
            start = 30
            end = 180 - start
            th = (end - start) / (nps - 1)
            angle = start
            poles = []
            for p in range(nps):
                print('angle: {}'.format(angle))
                imag = np.cos(angle * np.pi / 180)
                if abs(omegac * imag) < 1.0:
                    imag = 0

                poles.append(omegac * complex( -np.sin(angle * np.pi / 180),
                                              imag ) )

                angle = angle + th # +0.000000001

            print('poles: {}'.format(poles))
            zeros = [ ]
        elif False:
            # notch filter at f
            angle = 68

            p_zero = omegac * complex( 0, 1 )
            n_zero = omegac * complex( 0, -1 )
            p_pole = omegac * complex( -np.cos(angle * np.pi / 180),
                                      np.sin(angle * np.pi / 180) )
            n_pole = omegac * complex( -np.cos(angle * np.pi / 180),
                                      -np.sin((angle + 0.00001) * np.pi / 180) )
            zeros = [ p_zero, n_zero ]
            poles = [ p_pole, n_pole ]
        else:
            # band-pass filter at f
            nps = 7
            loops = 2 * nps * 125e6 / f
            th = 180 / (nps - 1)
            angle = 0
            pole = omegac * complex( 0, 1 )
            poles = []
            for p in range(nps):
                print('angle: {}'.format(angle))
                poles.append(pole + omegac * complex( -np.sin(angle * np.pi / 180),
                                                     np.cos(angle * np.pi / 180) ) / 32)

                angle = angle + th

            angle = 0.000001
            pole = omegac * complex( 0, -1 )
            for p in range(nps):
                poles.append(pole + omegac * complex( -np.sin(angle * np.pi / 180),
                                                     np.cos(angle * np.pi / 180) ) / 32)

                angle = angle + th

            print('poles: {}'.format(poles))
            zeros = [ ]
        if unity:
            iir._setup_unity()
        else:
            iir.setup(gain = gain,
                      zeros=zeros, poles=poles,
                      loops=loops)

        print('Filter sampling frequency: {} MHz'.format(125./iir.loops))

        # useful diagnostic functions
        print('IIR on: {}'.format(iir.on))
        print('IIR bypassed: {}'.format(iir.bypass))
        print('IIR loops: {} requested: {}'.format(iir.loops, loops))
        print('IIR overflows: {}'.format(iir.overflow))
        print('Coefficients (6 per biquad)\n{}'.format(iir.coefficients))

        # check if the filter is ok
        print('IIR overflows before: {}'.format(iir.overflow))

        # measure tf of iir filter
        iir.input = na.iq
        na.setup(start_freq = 1e4, stop_freq = stop_freq, rbw = rbw,
                 points = points,
                 average_per_point = 1,
                 amplitude = 0.1,
                 input = 'iir', output_direct = 'off',
                 logscale=True)

        tf = na.single()

        # check if the filter is ok
        print('IIR overflows after: {}'.format(iir.overflow))

        # retrieve designed transfer function
        designdata = iir.transfer_function(na.frequencies)

        #plot with design data

        import matplotlib
        matplotlib.rcParams['figure.figsize'] = (10, 6)

        from pyrpl.hardware_modules.iir.iir_theory import bodeplot
        bodeplot([(na.frequencies, designdata, 'designed system'),
                  (na.frequencies, tf, 'measured system')], xlog=True)
