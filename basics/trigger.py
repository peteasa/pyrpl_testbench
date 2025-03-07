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
    HOSTNAME = 'rp-f0bd75' # change this to match your RedPitaya!
    CONFIG = 'basic.trigger'
    p = pyrpl.Pyrpl(config=CONFIG, hostname=HOSTNAME, gui=False, reloadfpga = True) # False)
    r = p.rp
    s = r.scope
    asg = r.asg0
    trig = r.trig

    if trig.armed:
        print('WARNING: trigger is already armed')
        print('  consider reloading fpga to clear, or removing Pyrpl config')
        print('At start of run: trig armed: {} trigger timestamp: {} current timestamp: {}'.format(
            trig.armed, trig.trigger_timestamp, trig.current_timestamp))

    fg = 1000
    requested_samples_per_cycle = 300

    # s.sampling_time will auto call_setup
    s.sampling_time = 1 / (fg * requested_samples_per_cycle)
    samples_per_cycle = 1 / (fg * s.sampling_time)
    print('Sample time: {} Full buffer in: {:0.3e} Selected samples per cycle: {:0.3f} decimation: {}'.format(s.sampling_time, s.duration, samples_per_cycle, s.decimation))

    fs = 1 / s.sampling_time
    ag = 0.4
    ao = 0.0
    print('Generation {:.1f}V at {:.1f}Hz Sampling at {:.1f}Hz'.format(ag, fg, fs))

    # Prepare signal generator
    asg.bursts = 0
    asg.setup(
        frequency = fg,
        amplitude = ag,
        start_phase = 0,
        cycles_per_burst = 0,
        trigger_source = 'off',
        waveform = 'sin',
        offset = ao,
        output_direct = 'off')

    # set scope ch1 to pid2
    s.input1 = asg.name
    s.input2 = trig.name

    # only 1 trace average
    s.trace_average = 1

    # trigger from Trig module
    trigger_source = 'dsp'

    # seconds to delay (s.trigger_delay would auto call_setup)
    trigger_delay = 0

    # Can't pre capture more than s.duration / 2
    trigger_delay = trigger_delay if -s.duration / 2 < trigger_delay else -s.duration / 2

    trig_source = 'pos_edge'
    rng = np.random.default_rng()
    trig_threshold = ao + ag * (rng.random() - 0.5)
    trig_hysteresis = 1e-3

    # Now arm the trigger
    trig.setup(input = asg,
               output_direct = 'off',
               threshold = trig_threshold,
               hysteresis = trig_hysteresis,
               phase_offset = 0,
               auto_rearm = False,
               trigger_source = 'pos_edge',
               output_signal = '{}_phase'.format(asg.name))

    print('signal: {} output_direct: {} trigger_source: {} output_signal: {}'.format(
        trig.signal(), trig.output_direct, trig.trigger_source, trig.output_signal))
    print('after trig setup: current_output: {:0.3e} trig armed: {} trigger timestamp: {} current timestamp: {}'.format(
        trig.current_output_signal,
        trig.armed, trig.trigger_timestamp, trig.current_timestamp))

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

    print('Curve ready before generation: {} trig armed: {}'.format(s.curve_ready(), trig.armed))

    sequence_time = 0.01
    start_trigger_time = trig.current_timestamp
    asg.trigger_source = 'immediately'
    sleep(sequence_time)
    asg.trigger_source = 'off'
    end_trigger_time = trig.current_timestamp
    triggered_after_absolute = trig.trigger_timestamp - start_trigger_time
    triggered_after = sequence_time * triggered_after_absolute / (end_trigger_time - start_trigger_time)

    # wait for scope to finish
    finish_time = s.trigger_delay + s.duration / 2 - sequence_time
    if 0 < finish_time:
        sleep(finish_time)

    print('trigger threshold was: {:06f} sampler.trig is: {:0.6f} output_signal: {:0.6f}'.format(
        trig_threshold, r.sampler.trig, trig.current_output_signal))
    asg_phase = trig.output_signal_to_phase(r.sampler.trig)

    # check that the trigger has been disarmed
    print('At end of run: trig armed: {} triggered_after: {:0.6f}s absolute time: {}'.format(
        trig.armed, triggered_after, triggered_after_absolute))
    print('After turning on asg phase from trigger is: {:0.3f} degrees'.format(asg_phase))
    print('Curve ready: {} trig armed: {}'.format(s.curve_ready(), trig.armed))
    print('Trigger event age [ms]: {:0.3f}'.format(8e-9 * ((s.current_timestamp & 0xFFFFFFFFFFFFFFFF) - s.trigger_timestamp) * 1000))

    results = curve.result()
    times = s.times * 1e3
    print(times)

    zoom = False
    idx = 0
    end = times.shape[0]
    if zoom:
        start_time = -1 # ms
        time_to_plot = 5 # ms
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
    plt.plot(times[idx:end], results[0,idx:end], label = s.input1)
    plt.plot(times[idx:end], results[1,idx:end], label = s.input2)

    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.xlabel('Time [ms]');
    plt.ylabel('Voltage');
    plt.tight_layout()

    if svg:
        plt.savefig('trigger.svg')
    else:
        plt.show()
