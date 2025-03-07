#!/usr/bin/env python3

import numpy as np
import pyrpl
from time import time

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
    HOSTNAME = 'rp-f0bd75' # change this to match your RedPitaya!
    CONFIG = 'basic.pid_dc'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False, reloadfpga = True) # False)
    r = p.rp
    s = r.scope

    pid = r.pid2
    asg = r.asg1

    # set input to asg1
    pid.input = asg.name

    pid.reg_integral = 0 # reset the integrator to zero

    # disable the input filters
    pid.inputfilter = [0, 0, 0, 0]

    # turn off the gains for now
    # proportional gain, integrator gain
    pid.p, pid.i = 0, 0

    print('proportional gain: {}'.format(pid.p))
    print('integral unity-gain frequency [Hz]: {}'.format(pid.i))

    # set asg to constant 0.1 Volts
    asg.setup(waveform='dc', offset = 0.1)
    asg.output_direct = "off"

    # set scope ch1 to pid2
    s.input1 = pid.name

    # turn on integrator to whatever negative gain
    pid.i = -10

    # set integral value above the maximum positive voltage
    pid.reg_integral = 1.5

    # take 1000 points - jitter of the ethernet delay will add a noise here but we dont care
    times, ivals, outputs = [], [], []
    for n in range(1000):
        times.append(time())
        ivals.append(pid.reg_integral)
        outputs.append(r.scope.voltage_in1)

    # plot
    prepare_to_show_plot(p)

    svg = False
    if svg:
        import matplotlib

        # enable svg plotting
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    times = np.array(times) - min(times)
    plt.plot(times, ivals, label='ivals')
    plt.plot(times, outputs, label='in1');
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.xlabel('Time [s]');
    plt.ylabel('Voltage');
    plt.title('PID with dc')
    plt.tight_layout()

    if svg:
        plt.savefig('integ_dc.svg')
    else:
        plt.show()
