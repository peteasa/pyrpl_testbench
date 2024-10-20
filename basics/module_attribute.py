#!/usr/bin/env python3

import numpy as np
import pyrpl
from pyrpl.async_utils import sleep
from pyrpl.hardware_modules.asg import Asg0, Asg1

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'basic.module_attribute'
    p = pyrpl.Pyrpl(config='', hostname=HOSTNAME, gui=False)
    r = p.rp
    s = r.scope
    asg = r.asg0

    waveforms = asg.waveforms
    waveforms.append('custom')
    Asg0.waveform.change_options(asg, waveforms)

    print('waveforms: {}'.format(asg.waveforms))

    # define the custom waveform
    x1 = np.linspace(0, 2 * np.pi, asg.data_length, endpoint=False, dtype = float)
    x2 = np.linspace(0, 16 * np.pi, asg.data_length, endpoint=False, dtype = float)

    y = (4 * np.sin(x1) + np.sin(x2)) / 5.0
    asg.data = y

    fg = 100
    asg.delay_between_bursts = 2000000 / fg
    asg.bursts = 1
    asg.setup(waveform='custom',
              frequency = fg,
              offset = 0,
              amplitude = 0.9,
              start_phase = 0,
              trigger_source = 'off',
              output_direct = 'out1',
              cycles_per_burst = 4)

    acquire_length = 6
    requested_samples_per_cycle = 20

    trigger = -0.03
    hysteresis = 0.01

    # s.trigger_delay would auto call_setup
    trigger_delay = 0.0 # -0.016 # 0.079348

    # s.sampling_time would auto call_setup
    sampling_time = 1 / ( fg * requested_samples_per_cycle )
    s.setup(threshold = trigger,
            hysteresis = hysteresis,
            trigger_source = 'ch1_negative_edge',
            trigger_delay = trigger_delay,
            trace_average = 1,
            input1 = 'in1')

    samples_per_cycle = 1 / (fg * s.sampling_time)
    fs = 1 / s.sampling_time

    samples_to_plot = int(acquire_length * samples_per_cycle)

    results = s.single_async()
    if s.curve_ready():
        discard = curve.result()
        results = s.single_async()
        sleep(0.001)

    print('Curve ready: {}'.format(s.curve_ready()))

    # similar to asg.trig()
    for attempt in range(5):
        if s.curve_ready():
            break

        asg.trigger_source = 'immediately'
        sleep(1.0)
        asg.trigger_source = 'off'
        sleep(0.1)

    print('After asg start: curve ready: {}'.format(s.curve_ready()))
    curve = results.result()

    times = s.times * 1e3

    zoom = True
    idx = 0
    end = times.shape[0]
    if zoom:
        print('Zoom in to {}s'.format(s.sampling_time * samples_per_cycle * acquire_length))
        idx = np.searchsorted(times, - s.sampling_time * samples_per_cycle * 1e3)
        end = idx + samples_to_plot
        end = times.shape[0] if times.shape[0] < end else end
        idx = end - samples_to_plot
        idx = 0 if idx < 0 else idx

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(times[idx:end], curve[0,idx:end], label = s.input1)
    plt.legend(loc = 'center left', bbox_to_anchor = (1.04, 0.5))
    plt.xlabel('Time [ms]')
    plt.tight_layout()

    plt.show()
