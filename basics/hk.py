#!/usr/bin/env python3

#
# for this example connect the scope input to connector E1 pin 17 PL_DIO7_P
#

import numpy as np
import pyrpl
from pyrpl.async_utils import sleep
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

def sequencer(loopcount = 500, sleep_time = 0.01):
    sleep_time_min = 0.002
    sleep_time = sleep_time_min if sleep_time < sleep_time_min else sleep_time

    on = False
    count = 0
    counter = 0
    max_count = loopcount / 10
    mark_space = max_count / 2
    now = time()
    for i in range(loopcount):
        if counter < 1:
            hk.expansion_P7 = on
            if on:
                counter = mark_space
            else:
                counter = max_count - mark_space

            on = not on
            mark_space += 3
            mark_space = mark_space % max_count

        counter -= 1
        sleep(sleep_time)

    # end of sequence marker
    for i in range(6):
        hk.expansion_P7 = on
        on = not on
        sleep(0.01)

    hk.expansion_P7 = False
    duration = time() - now
    print('sequence duration: {:0.3f} on / off: {} / {}'.format(duration, max_count - mark_space, mark_space))

    return duration

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75' # change this to match your RedPitaya!
    CONFIG = 'basic.hk'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False, reloadfpga = True) #False)
    r = p.rp
    s = r.scope
    hk = r.hk

    if False:
        # trivial led test
        for i in range(1025):
            hk.led = i
            sleep(0.005)

    # setup scope input
    s.input1 = 'in1'
    probe_attenuation = 1
    input_attenuation = 20 * probe_attenuation

    # simple pulse generator
    sleep_time = 0.005
    fg = 0.25 / sleep_time

    # prepare the pulse generator
    hk.expansion_P7_output = True
    hk.expansion_P7 = False

    # scope capture rate
    acquire_length = 20000
    requested_samples_per_cycle = 10

    if True:
        trigger_source = 'ch1_positive_edge'
    else:
        trigger_source = 'immediately'

    # s.sampling_time will auto call_setup
    s.sampling_time = 1 / (fg * requested_samples_per_cycle)
    samples_per_cycle = 1 / (fg * s.sampling_time)
    print('Sample time: {} Full buffer in: {} Selected samples per cycle: {} decimation: {}'.format(s.sampling_time, s.duration, samples_per_cycle, s.decimation))

    samples_to_plot = int(acquire_length * samples_per_cycle)

    # trigger at
    s.threshold = 0.7 / input_attenuation
    s.hysteresis = 0.01
    cycles_to_delay = 4 / (s.sampling_time * samples_per_cycle)

    # only 1 trace average
    s.trace_average = 1

    # seconds to delay
    # s.trigger_delay would auto call_setup
    trigger_delay = cycles_to_delay * s.sampling_time * samples_per_cycle

    # Can't pre capture more than s.duration / 2
    trigger_delay = trigger_delay if -s.duration / 2 < trigger_delay else -s.duration / 2
    s.setup(trigger_source = trigger_source,
            trigger_delay = trigger_delay,
            rolling_mode = True)

    # setup the scope for an acquisition
    results = s.single_async()

    # wait for pre trigger buffer to fill
    arm_time = s.duration / 2
    trigger_prep = - s.trigger_delay if s.trigger_delay < 0 else 0
    sleep(arm_time + trigger_prep)

    print('Curve ready before generation: {}'.format(s.curve_ready()))
    sequence_time = sequencer(loopcount = 1000, sleep_time = sleep_time)

    # wait for scope to finish
    finish_time = s.trigger_delay + s.duration / 2 - sequence_time
    if 0 < finish_time:
        sleep(finish_time)

    print('Curve ready after generation: {}'.format(s.curve_ready()))
    print('input: {} active: {} trigger: {} trigger_delay: {:0.3f} duration: {:0.3f}'.format(
        s.input1, s.ch1_active, s.trigger_source, s.trigger_delay, s.duration))
    print('rolling_mode: {} running_state: {}'.format(s.rolling_mode, s.running_state))
    if s.data_avg is not None:
        data = s.data_avg[0]
        print('data: {}'.format(data))

    curve = results.result()
    times = s.times * 1e3

    zoom = False
    idx = 0
    end = times.shape[0]
    if zoom:
        start_time = 6000 # ms
        time_to_plot = 2000 # ms
        samples_to_plot = int(time_to_plot / (1000 * s.sampling_time))
        idx = np.searchsorted(times, start_time)
        end = idx + samples_to_plot
        end = times.shape[0] if times.shape[0] < end else end
        idx = end - samples_to_plot
        idx = 0 if idx < 0 else idx

    # plot
    prepare_to_show_plot(p)

    svg = False
    if svg:
        import matplotlib

        # enable svg plotting
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    # plot the data
    fig = plt.figure()
    plt.plot(times[idx:end], curve[0,idx:end] * input_attenuation, label = s.input1)
    #plt.plot(times[idx:end], curve[1,idx:end], label = s.input2)

    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.xlabel('Time [ms]');
    plt.ylabel('Voltage');
    #plt.title('ASG with {} phi: {}'.format(asg.waveform, asg.start_phase))
    plt.tight_layout()

    if svg:
        plt.savefig('ams.svg')
    else:
        plt.show()
