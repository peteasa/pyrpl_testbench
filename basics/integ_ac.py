#!/usr/bin/env python3

import numpy as np
import pyrpl
from pyrpl.async_utils import sleep

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
    CONFIG = 'basic.pid_ac'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False, reloadfpga = True) # False)
    r = p.rp
    s = r.scope

    pid = r.pid0
    asg = r.asg1
    asg.setup(waveform = 'dc', offset = 0, trigger_source = 'off', output_direct = 'off')

    # set input to asg1
    pid.input = asg.name

    # reset the integrator to zero
    pid.reg_integral = 0

    # leave input filter disabled for now
    pid.inputfilter = [0, 0, 0, 0]

    # 4 input filters in series
    # can be either off (bandwidth=0)
    # lowpass (bandwidth positive)
    # highpass (bandwidth negative)
    # filter bandwidths that are powers of 2

    filtered = ''
    if False:
        # minimum cutoff frequency is 2 Hz, maximum 77 kHz (for now)
        # +ve low pass -ve high pass 0 off
        hi = 10
        lo = 10000
        pid.inputfilter = [-hi, lo]
        filtered = '_h{}_l{}'.format(hi, lo)
        print(pid.inputfilter)
    elif False:
        hi = 100
        lo = 5000
        pid.inputfilter = [-hi, lo]
        filtered = '_h{}_l{}'.format(hi, lo)
        print(pid.inputfilter)
    elif False:
        hi = 800
        lo = 2000
        pid.inputfilter = [-hi, lo]
        filtered = '_h{}_l{}'.format(hi, lo)
        print(pid.inputfilter)
    elif False:
        hi = 800
        lo = 10000
        pid.inputfilter = [-hi, lo]
        filtered = '_h{}_l{}'.format(hi, lo)
        print(pid.inputfilter)

    #turn off the gains for now
    # proportional gain, integrator gain
    pid.p, pid.i = 0, 0

    print('proportional gain: {}'.format(pid.p))
    print('integral unity-gain frequency [Hz]: {}'.format(pid.i))

    # prepare to turn on integrator at some negative gain
    pid_i = -400

    # set integral value above the maximum positive voltage
    pid_reg_integral = 0.2

    fg = 1000
    requested_samples_per_cycle = 300

    # s.sampling_time will auto call_setup
    s.sampling_time = 1 / (fg * requested_samples_per_cycle)
    samples_per_cycle = 1 / (fg * s.sampling_time)
    print('Sample time: {} Full buffer in: {} Selected samples per cycle: {} decimation: {}'.format(s.sampling_time, s.duration, samples_per_cycle, s.decimation))

    fs = 1 / s.sampling_time
    ag = 0.1
    ao = 0.01
    print('Generation {:.1f}V at {:.1f}Hz Sampling at {:.1f}Hz'.format(ag, fg, fs))

    #**** Note: this is not the way to configure the asg! ****
    # s.trigger_source will auto call_setup, starting the scope capture
    # asg.trigger_source = 'immediately'
    # Now change the waveform mid flow!
    # asg.waveform = 'halframp'
    # Now set offset mid flow!
    # asg.offset = ao

    #asg.delay_between_bursts = 1500
    asg.bursts = 0
    asg.setup(
        frequency = fg,
        amplitude = ag,
        start_phase = 0,
        cycles_per_burst = 0,
        trigger_source = 'off',
        waveform = 'halframp',
        offset = ao)

    # set scope ch1 to pid2
    s.input1 = pid.name
    s.input2 = asg.name

    # trig at zero volt crossing
    s.threshold = 0.05

    # positive/negative slope is detected by waiting for input to
    # sweept through hysteresis around the trigger threshold in
    # the right direction
    s.hysteresis = 0.01

    # only 1 trace average
    s.trace_average = 1

    # trigger on the input signal positive slope (s.trigger_source would auto call_setup)
    trigger_source = 'ch2_positive_edge'

    # seconds to delay (s.trigger_delay would auto call_setup)
    trigger_delay = 0.010

    # Can't pre capture more than s.duration / 2
    trigger_delay = trigger_delay if -s.duration / 2 < trigger_delay else -s.duration / 2
    s.setup(trigger_source = trigger_source,
            trigger_delay = trigger_delay)

    # setup the scope for an acquisition
    curve = s.single_async()

    for i in range(2):
        # wait for pre trigger buffer to fill
        arm_time = s.duration / 2
        trigger_prep = - s.trigger_delay if s.trigger_delay < 0 else 0
        sleep(arm_time + trigger_prep)

        # trigger should still be armed
        if i == 0 and s.curve_ready():
            discard = curve.result()
            curve = s.single_async()
            continue

        break

    print('Curve ready before generation: {}'.format(s.curve_ready()))

    sequence_time = 1.0
    asg.trigger_source = 'immediately'

    # turn on pid module
    pid.i, pid.reg_integral = pid_i, pid_reg_integral

    sleep(sequence_time)
    asg.trigger_source = 'off'

    # wait for scope to finish
    finish_time = s.trigger_delay + s.duration / 2 - sequence_time
    if 0 < finish_time:
        sleep(finish_time)

    # check that the trigger has been disarmed
    print('After turning on asg:')
    print('Curve ready: {}'.format(s.curve_ready()))
    print('Trigger event age [ms]: {}'.format(8e-9 * ((s.current_timestamp & 0xFFFFFFFFFFFFFFFF) - s.trigger_timestamp) * 1000))

    # plot
    prepare_to_show_plot(p)

    svg = False
    if svg:
        import matplotlib

        # enable svg plotting
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    # plot the data
    result = curve.result()
    fig = plt.figure()
    plt.plot(s.times*1e3, result[0], label='curve[0]')
    plt.plot(s.times*1e3, result[1], label='curve[1]');

    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.xlabel('Time [ms]');
    plt.ylabel('Voltage');
    plt.title('PID with halframp {}'.format(filtered))
    plt.tight_layout()

    if svg:
        plt.savefig('integ_ac{}.svg'.format(filtered))
    else:
        plt.show()
