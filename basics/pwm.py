#!/usr/bin/env python3

#
# for this example connect the scope input to connector E2 pin 17 AO0
#

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

def to_hex(val):
    return np.int32(round(val * 2 ** 13))

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

def split(val):
    # ref: red_pitaya_ams.v pwm0_i
    # bits 13:6
    msv = np.int32((val & 0b11111111000000) / 2 ** 6)
    # bits 5:2
    lsv = np.int32((val & 0b111100) / 2 ** 2)
    # bits 1:0 are ignored

    return (msv, lsv)

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

    # sequence bytes
    # 0,b3,b2,b3, b1,b3,b2,b3, b0,b3,b2,b3, b1,b3,b2,b3
    low_mask -= 1
    low = np.int32(value & low_mask)

    return (msb, high, low)

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75' # change this to match your RedPitaya!
    CONFIG = 'basic.pwm'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False, reloadfpga = True) # False) #
    r = p.rp
    s = r.scope

    pwm_number = 0
    pwm = getattr(r, 'pwm{}'.format(pwm_number))

    # only pid.ival and asg.offset provide suitable registers for the pwm config
    #print('input options: {}'.format(pwm.input_options))

    # configure the pwm mark space ration using the output from the asg0 register
    # or from the output of the r.pid0 register
    pwm.input = r.pid0 # r.asg0 #

    lower = 0b1000
    pwm_test_config = combine(0x20, lower)
    pwm_test_config_float = pwm_test_config / 2 ** 13
    pwm_split = split(pwm_test_config)

    print('pwm_input: {}'.format(pwm.input))
    print('pwm_test_config: (0x{:x}, 0x{:x}) 0x{:x} {}'.format(pwm_split[0], pwm_split[1], pwm_test_config, pwm_test_config_float))

    s.input1 = 'in1'
    probe_attenuation = 1
    input_attenuation = 20 * probe_attenuation

    if pwm.input == 'pid0':
        s.input2 = 'pid0'

    if pwm.input == 'asg0':
        s.input2 = 'asg0'

    fg = 30000
    generator_samples_per_cycle = 10

    acquire_length = 6
    requested_samples_per_cycle = generator_samples_per_cycle * 4

    # s.sampling_time will auto call_setup
    s.sampling_time = 0.01 / (fg * requested_samples_per_cycle)
    samples_per_cycle = 1 / (fg * s.sampling_time)
    print('Sample time: {} Full buffer in: {} Selected samples per cycle: {} decimation: {}'.format(
        s.sampling_time, s.duration, samples_per_cycle, s.decimation))

    samples_to_plot = int(acquire_length * samples_per_cycle)

    cycles_to_delay = (0.00006) / (s.sampling_time * samples_per_cycle)

    fs = 1 / s.sampling_time
    #print('Generation ?? at {:.1f}Hz Sampling at {:.1f}Hz'.format(fg, fs))

    # trig at
    s.threshold = 0.6 / input_attenuation

    # positive/negative slope is detected by waiting for input to
    # sweept through hysteresis around the trigger threshold in
    # the right direction
    s.hysteresis = 0.01

    # only 1 trace average
    s.trace_average = 1

    # trigger on the input signal positive slope (s.trigger_source would auto call_setup)
    trigger_source = 'ch2_positive_edge' # 'ch1_positive_edge' # 'ch2_positive_edge' #

    # seconds to delay
    # s.trigger_delay would auto call_setup
    trigger_delay = cycles_to_delay * s.sampling_time * samples_per_cycle

    # Can't pre capture more than s.duration / 2
    print(-s.duration / 2, trigger_delay)
    trigger_delay = trigger_delay if -s.duration / 2 < trigger_delay else -s.duration / 2
    s.setup(trigger_source = trigger_source, trigger_delay = trigger_delay)

    # Prepare the scope for an acquisition
    results = s.single_async()
    for i in range(2):
        # wait for pre trigger buffer to fill
        arm_time = s.duration / 2
        trigger_prep = - s.trigger_delay if s.trigger_delay < 0 else 0
        sleep(arm_time + trigger_prep)

        # trigger should still be armed
        if i == 0 and s.curve_ready():
            discard = results.result()
            results = s.single_async()
            continue

        break

    print('Curve ready before turning on pwm: {}'.format(s.curve_ready()))

    freq = fg
    samples_to_generate = 10
    sleepmin = 0.020
    maxfreq = 1 / (generator_samples_per_cycle * sleepmin)
    freq_to_generate = freq
    if maxfreq < freq:
        freq_to_generate = maxfreq
        print('python generator - too fast - updated frequency: {:0.3f}'.format(freq_to_generate))

    sleeptime = 1 / (samples_per_cycle * freq_to_generate)
    start = time()
    for i in range(samples_to_generate):
        # to trigger the scope we alternate between two pwm values
        # each produces a different analogue voltage output depending on the
        # mark space ratio of the pwm output and the RC low pass filter
        if i%2:
            # vary this output to produce step in output voltage to trigger the scope
            cfg = to_float(combine((0x20 - i*7) % 0x80, lower))
        else:
            # this is the first pwm that we use to explore the pwm output
            cfg = pwm_test_config_float

        if pwm.input == 'asg0':
            r.asg0.setup(waveform='dc', offset = cfg, output_direct = 'off')

        if pwm.input == 'pid0':
            r.pid0.ival = cfg
            ival = to_hex(r.pid0.ival)

        print('{:x} '.format(to_hex(cfg)), end = '')
        sleep(sleeptime)

    duration = time() - start

    # set the last value after the sequence to provide final readings for the test
    cfg = pwm_test_config_float
    if pwm.input == 'asg0':
        r.asg0.setup(waveform='dc', offset = cfg, output_direct = 'off')

    if pwm.input == 'pid0':
        r.pid0.ival = cfg
        ival = to_hex(r.pid0.ival)

    revolutions = samples_to_generate / samples_per_cycle
    print('generated: {} cycles: {:0.1f} samples_per_cycle: {:0.1f} sleeptime: {:0.6f} duration: {:0.3f} freq: {:0.3f}'.format(
        samples_to_generate,
        revolutions,
        samples_to_generate / revolutions,
        sleeptime,
        duration,
        revolutions / duration))

    sequence_time = duration

    # wait for scope to finish
    finish_time = s.trigger_delay + s.duration / 2 - sequence_time + 2
    if 0 < finish_time:
        sleep(finish_time)

    # check that the trigger has been disarmed
    print('After turning on pwm:')
    print('Curve ready: {}'.format(s.curve_ready()))
    print('Trigger event age [ms]: {}'.format(8e-9 * ((s.current_timestamp & 0xFFFFFFFFFFFFFFFF) - s.trigger_timestamp) * 1000))
    print('Trigger threshold: {}'.format(s.threshold * input_attenuation))

    if pwm.input == 'asg0':
        offset = to_hex(r.asg0.offset)
        offset_split = split(offset)
        print('offset: (0x{:x}, 0x{:x}) 0x{:x} {}'.format(offset_split[0], offset_split[1], offset, r.asg0.offset))

    if pwm.input == 'pid0':
        #ival_addr = 0x100
        #ival_reg = LongRegister(ival_addr, doc='pid0.ival')
        #ival_reg._write(r.pid0, ival_addr, pwm_test_config)
        ival = to_hex(r.pid0.ival)
        ival_split = split(ival)
        print('ival: (0x{:x}, 0x{:x}) 0x{:x} {}'.format(ival_split[0], ival_split[1], ival, r.pid0.ival))

    # Treat the dac register as an unsigned 32bit register
    reg = 0x20 + pwm_number * 4
    dac_reg = LongRegister(reg, doc='PWM{} config'.format(pwm_number))
    dac = dac_reg._read(r.ams, dac_reg.address)
    dac = np.int32(dac)
    dac_split = split_dac(dac)
    dac_name = 'dac{}'.format(pwm_number)
    dac_f = getattr(r.ams, dac_name)
    print('~pwm{}_i[13]: {:x} pwm{}_i[12:6]: 0x{:x} cfg[7:0]: 0x{:04x} {}: 0x{:06x}'.format(
        pwm_number, dac_split[0], pwm_number, dac_split[1], dac_split[2], dac_name, dac))

    # Stop pwm
    pwm.input = 'off'

    curve = results.result()
    times = s.times * 1e3

    zoom = True
    idx = 0
    end = times.shape[0]
    if zoom:
        start_time = .04 # ms
        time_to_plot = .002 # ms
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
    plt.title('PWM')
    plt.tight_layout()

    if svg:
        plt.savefig('pwm_{}.svg'.format(s.input))
    else:
        plt.show()
