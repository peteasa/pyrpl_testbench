#!/usr/bin/env python3

#
# for this example connect the scope input to connector E2 with one of the pins
#     dac0: pin 17 AO0, dac1: pin 18 AO1, dac2: pin 19 AO2, dac3: pin 19 AO3
#
# now set the associated dac:
DAC = 'dac0'

import numpy as np
import pyrpl
from pyrpl.async_utils import sleep
from pyrpl.attributes import LongRegister
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

def sinwave(minv, maxv, freq, samples_per_cycle, modules, module_name, reg):
    # crude python signal generator
    samples_to_generate = 1000
    sleepmin = 0.002
    maxfreq = 1 / (samples_per_cycle * sleepmin)
    freq_to_generate = freq
    if maxfreq < freq:
        freq_to_generate = maxfreq
        print('python generator - too fast - updated frequency: {:0.3f}'.format(freq_to_generate))

    sleeptime = 1 / (samples_per_cycle * freq_to_generate)
    maxrange = maxv - minv
    module = getattr(modules['rp'], modules[dac][module_name])
    start = time()
    for i in range(samples_to_generate):
        setattr(module, reg, minv + maxrange * (1 + np.sin(2 * np.pi * i / samples_per_cycle)) / 2)
        sleep(sleeptime)

    setattr(module, reg, minv + maxrange / 2)
    duration = time() - start
    revolutions = samples_to_generate / samples_per_cycle
    print('generated: {} cycles: {:0.1f} samples_per_cycle: {:0.1f} sleeptime: {:0.6f} duration: {:0.3f} freq: {:0.3f}'.format(
        samples_to_generate,
        revolutions,
        samples_to_generate / revolutions,
        sleeptime,
        duration,
        revolutions / duration))

    return duration

def to_float(val):
    return val / 2 ** 13

def combine(msv, lsv):
    # ref: red_pitaya_ams.v
    # bit 13
    msb = msv & 0x80

    # bits 13:6
    msv_m = msv & 0xFF

    # bits 5:2
    lsv_m = lsv & 0b1111
    # bits 1:0 are ignored

    return np.int32(msv_m * 2 ** 6 + lsv_m * 2 ** 2)

def split_dac(val):
    PWM_LOW_BITS = 16
    PWM_HIGH_BITS = 8
    value = np.int32(val)

    # cfg / cfg_b: see red_pitaya_ams.v
    # ~pwm_i[13]
    mask = 2 ** (PWM_HIGH_BITS + PWM_LOW_BITS - 1)
    msb = np.int32((value & mask) / mask)

    # pwm_i[13-1:6]
    low_mask = 2 ** PWM_LOW_BITS
    mask = (2 ** (PWM_HIGH_BITS-1) - 1) * low_mask
    high = np.int32((value & mask) / low_mask)

    # 0,b3,b2,b3, b1,b3,b2,b3, b0,b3,b2,b3, b1,b3,b2,b3
    low_mask -= 1
    low = np.int32(value & low_mask)

    return (msb, high, low)

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'basic.ams'
    p = pyrpl.Pyrpl(config=CONFIG, hostname=HOSTNAME, gui=False, reloadfpga = True) # False)
    r = p.rp
    s = r.scope

    modules = {'rp' : r,
               'dac0': {'module_name': 'pwm0', 'input': 'pid0', 'reg': 'ival'}, # pin 17
               'dac1': {'module_name': 'pwm1', 'input': 'pid0', 'reg': 'ival'}, # pin 18
               'dac2': {'module_name': 'ams', 'reg': 'dac2'}, # pin 19
               'dac3': {'module_name': 'ams', 'reg': 'dac3'} # pin 20
               }

    dac = DAC

    module_name = 'module_name'
    reg = modules[dac]['reg']
    if 'input' in modules[dac].keys():
        module = getattr(r, modules[dac]['module_name'])
        setattr(module, 'input', modules[dac]['input'])
        module_name = 'input'

    # setup scope input
    s.input1 = 'in1'
    probe_attenuation = 1
    input_attenuation = 20 * probe_attenuation

    # simple signal generator
    fg = 40
    generator_samples_per_cycle = 20

    # scope capture rate
    acquire_length = 20000
    requested_samples_per_cycle = generator_samples_per_cycle * 4

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
    s.threshold = 0.5 / input_attenuation
    s.hysteresis = 0.01
    cycles_to_delay = 2 / (s.sampling_time * samples_per_cycle)

    # only 1 trace average
    s.trace_average = 1

    # seconds to delay
    # s.trigger_delay would auto call_setup
    trigger_delay = cycles_to_delay * s.sampling_time * samples_per_cycle

    # Can't pre capture more than s.duration / 2
    trigger_delay = trigger_delay if -s.duration / 2 < trigger_delay else -s.duration / 2
    s.setup(trigger_source = trigger_source, trigger_delay = trigger_delay, rolling_mode = True)

    # setup the scope for an acquisition
    results = s.single_async()

    # wait for pre trigger buffer to fill
    arm_time = s.duration / 2
    trigger_prep = - s.trigger_delay if s.trigger_delay < 0 else 0
    sleep(arm_time + trigger_prep)

    print('Curve ready before generation: {}'.format(s.curve_ready()))
    if True:
        generation_time = sinwave(0, 1.8, fg, generator_samples_per_cycle, modules, module_name, reg)
    elif False:
        module = getattr(modules['rp'], modules[dac][module_name])
        for i in range(1000):
            setattr(module, reg, 1.8 * (1 + np.sin(2 * np.pi * i / 47)) / 2)
            sleep(0.002)

        generation_time = 1000 * 0.002
    else:
        # this test to configure / explore a single duty cycle
        pwm_number = int(dac[3])
        module = getattr(modules['rp'], modules[dac][module_name])
        if 1 < pwm_number:
            # value we expect from the dac register
            lower = 0x4
            upper = 0x5A

            # PWMRegister does the conversion for us
            test_val = float(upper * 0x10 + lower) * 1.8 / 2 ** 12
        else:
            # value we expect from the dac register
            lower = 0x4
            upper = 0xA5

            # IValAttribute is a simple Float register type
            # so we have to do the conversion
            msb = 0x80 & upper
            negative = msb == 0x80
            upper = upper ^ msb
            lower = 0xF ^ lower - 1 if negative else lower
            upper = 0x7F ^ upper if negative else upper
            pwm_config = combine(upper, lower)
            test_val = to_float(pwm_config)
            test_val = -test_val if negative else test_val
            print('test_val: {} pwm_config: 0x{:x}'.format(test_val, pwm_config))

        setattr(module, reg, test_val)

        # Treat the dac register as an unsigned 32bit register
        reg = 0x20 + pwm_number * 4
        dac_reg = LongRegister(reg, doc='PWM{} config'.format(pwm_number))
        dac = dac_reg._read(r.ams, dac_reg.address)
        dac = np.int32(dac)
        dac_split = split_dac(dac)
        dac_name = 'dac{}'.format(pwm_number)
        dac_f = getattr(r.ams, dac_name)
        print('test_val: {} ~pwm{}_i[13]: {:x} pwm{}_i[12:6]: 0x{:x} cfg[7:0]: 0x{:04x} {}: 0x{:06x}'.format(
            test_val, pwm_number, dac_split[0], pwm_number, dac_split[1], dac_split[2], dac_name, dac))

        generation_time = 1000

    # wait for scope to finish
    finish_time = s.trigger_delay + s.duration / 2 - generation_time + 2
    if 0 < finish_time:
        sleep(finish_time)

    print('Curve ready after generation: {}'.format(s.curve_ready()))
    print('input: {} active: {} trigger: {} trigger_delay: {:0.3f} duration: {:0.3f}'.format(
        s.input1, s.ch1_active, s.trigger_source, s.trigger_delay, s.duration))
    print('rolling_mode: {} running_state: {}'.format(s.rolling_mode, s.running_state))
    print('current value in1: {}'.format(s.voltage_in1 * input_attenuation))
    if s.data_avg is not None:
        data = s.data_avg[0]
        print('data: {}'.format(data))

    if s.curve_ready():
        # scope not expected to trigger for some of the options above
        curve = results.result()
        times = s.times * 1e3

        zoom = False
        idx = 0
        end = times.shape[0]
        if zoom:
            start_time = 1000 # ms
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
        plt.title('AMS with {}'.format(dac))
        plt.tight_layout()

        if svg:
            plt.savefig('ams.svg')
        else:
            plt.show()

    # Stop pwm
    module = getattr(modules['rp'], modules[dac][module_name])
    if 'input' in modules[dac].keys():
        setattr(module, 'input', 'off')
    else:
        lower = 0
        upper = 0x80
        setattr(module, reg, float(upper * 0x10 + lower) * 1.8 / 2 ** 12)
