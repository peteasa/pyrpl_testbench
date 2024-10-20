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
    CONFIG = 'basic.asg_burst'
    p = pyrpl.Pyrpl(config=CONFIG, hostname=HOSTNAME, gui=False)
    r = p.rp
    s = r.scope

    pid = r.pid0
    asg = r.asg0

    asg.setup(waveform='dc', offset = 0, amplitude = 0, trigger_source = 'off', output_direct = 'off')

    # reset the integrator to zero, disable filters and switch off gains
    pid.setup(inputfilter = [0, 0, 0, 0], p = 0, i = 0)

    # set scope ch1 to pid2
    s.input1 = pid.name
    s.input2 = asg.name

    # turn on integrator to whatever negative gain
    pid.i = -200.0

    # set integral value above the maximum positive voltage
    pid.reg_integral = 0.2 # -0.5

    start_phase = 45
    fg = 17000

    acquire_length = 6
    requested_samples_per_cycle = 490

    # s.sampling_time will auto call_setup
    s.sampling_time = 1 / (fg * requested_samples_per_cycle)
    samples_per_cycle = 1 / (fg * s.sampling_time)
    print('Sample time: {} Full buffer in: {} Selected samples per cycle: {} decimation: {}'.format(s.sampling_time, s.duration, samples_per_cycle, s.decimation))

    samples_to_plot = int(acquire_length * samples_per_cycle)

    fs = 1 / s.sampling_time
    ag = 0.1
    print('Generation {:.1f}V at {:.1f}Hz Sampling at {:.1f}Hz'.format(ag, fg, fs))

    # write the waveform before arming the trigger
    # start_phase set to whatever is required to get zero volts!
    waveform = 'sin'
    cycles_to_delay = 20
    ia = ag
    io = -ag
    ip = 0.0
    ao = 0.0
    if 'sin' == waveform:
        ia = 0.0
        io = 0.0
        ao = 0.0
        start_phase = 0.0
        cycles_to_delay = 5
    if 'ramp' == waveform:
        io = ia
        ao = ia
        cycles_to_delay = 5

    # trig at
    s.threshold = 0.05

    # positive/negative slope is detected by waiting for input to
    # sweept through hysteresis around the trigger threshold in
    # the right direction
    s.hysteresis = 0.01

    # only 1 trace average
    s.trace_average = 1

    # trigger on the input signal positive slope (s.trigger_source would auto call_setup)
    trigger_source = 'ch2_positive_edge'

    # seconds to delay
    # s.trigger_delay would auto call_setup
    trigger_delay = cycles_to_delay * s.sampling_time * samples_per_cycle
    s.setup(trigger_source = trigger_source, trigger_delay = trigger_delay)

    asg.delay_between_bursts = 2000000 / fg

    # demonstrate ASG wrong number of bursts https://github.com/lneuhaus/pyrpl/issues/493
    # depending on the number of cycles in each burst you can get extra bursts
    asg.bursts = 1
    asg.setup(frequency = fg,
              amplitude = ia, offset = io, start_phase = ip,
              trigger_source = 'off',
              waveform = waveform, cycles_per_burst = 3)

    # setup the scope for an acquisition
    results = s.single_async()
    sleep(0.001)

    # trigger should still be armed
    if s.curve_ready():
        discard = curve.result()
        results = s.single_async()
        sleep(0.001)

    print('Curve ready: {}'.format(s.curve_ready()))

    # set input to asg1
    pid.input = asg.name

    # start_phase causes an undesireable initial offset to be configured
    # during the setup phase two unwanted partial bursts occur followed by the configured bursts
    asg.start_phase = start_phase
    asg.amplitude = ag
    asg.offset = ao

    # similar to asg.trig()
    for attempt in range(5):
        if s.curve_ready():
            break

        asg.trigger_source = 'immediately'
        sleep(1.0)
        asg.trigger_source = 'off'
        sleep(0.1)

    # check that the trigger has been disarmed
    print('After turning on asg:')
    print('Curve ready: {}'.format(s.curve_ready()))
    print('Trigger event age [ms]: {}'.format(8e-9 * ((s.current_timestamp & 0xFFFFFFFFFFFFFFFF) - s.trigger_timestamp) * 1000))

    curve = results.result()
    times = s.times * 1e3

    zoom = False
    idx = 0
    end = times.shape[0]
    if zoom:
        print(s.sampling_time * samples_per_cycle)
        idx = np.searchsorted(times, - s.sampling_time * samples_per_cycle * 1e3)
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
    plt.plot(times[idx:end], curve[0,idx:end], label = s.input1)
    plt.plot(times[idx:end], curve[1,idx:end], label = s.input2)

    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.xlabel('Time [ms]');
    plt.ylabel('Voltage');
    plt.title('ASG with {} phi: {}'.format(asg.waveform, asg.start_phase))
    plt.tight_layout()

    if svg:
        plt.savefig('asg_burst_{}.svg'.format(asg.start_phase))
    else:
        plt.show()
