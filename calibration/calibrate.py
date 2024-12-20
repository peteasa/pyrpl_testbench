#!/usr/bin/env python3

import argparse
import json
import numpy as np
import pyrpl
from pyrpl.async_utils import sleep

DEBUG = False

class CalData(object):
    def __init__(self):
        offsets = np.zeros((8), dtype=float)
        scales = np.ones((8), dtype=float)
        volts_per_steps = np.ones((8), dtype=float)
        self._volts_per_bit_ideal = 1 / 2. ** 13  # 122.07uV
        self._curve_set = 4
        volts_per_steps = volts_per_steps * self.volts_per_bit_ideal

        self.caldata = {
            'VPERSTEPS': volts_per_steps,
            'SCALES': scales,
            'OFFSETS': offsets
            }

    def __str__(self):
        rtn = 'volts_per_steps\n{}'.format(self.volts_per_steps)
        rtn = '{}\nscales\n{}'.format(rtn, self.scales)
        rtn = '{}\noffsets\n{}'.format(rtn, self.offsets)

        return rtn

    @property
    def data(self):
        return self.caldata

    @property
    def scales(self):
        return self.data["SCALES"]

    def scale(self, val):
        discard, rtn = self._convert("SCALES", 1.0, val)
        return rtn

    @property
    def offsets(self):
        return self.data["OFFSETS"]

    def offset(self, val):
        discard, rtn = self._convert("OFFSETS", 0.0, val)
        return rtn

    @property
    def volts_per_steps(self):
        return self.data["VPERSTEPS"]

    def volts_per_step(self, val):
        discard, rtn = self._convert("VPERSTEPS", self.volts_per_bit_ideal, val)
        return rtn

    @property
    def in1(self):
        return 0
    @property
    def in2(self):
        return 1
    @property
    def out1(self):
        return 2
    @property
    def out2(self):
        return 3
    @property
    def volts_per_bit_ideal(self):
        return self._volts_per_bit_ideal

    def cal(self, io, ao, ag = 1.0):
        stritem, offset = self._convert("OFFSETS", 0.0, io)
        if DEBUG: print('calibrate io: {} ao: {} ag: {}'.format(stritem, ao, ag))
        if 'in' in stritem:
            scale = self.scale(stritem)
            outo = offset * self.volts_per_step(stritem) + ao * scale
            outg = ag * scale
        elif 'out' in stritem:
            scale = self.scale(stritem)
            outo = offset + ao * scale
            outg = ag * scale
        else:
            outo = ao
            outg = ag

        return outo, outg

    def reverse(self, io, so, sg = 1.0):
        stritem, offset = self._convert("OFFSETS", 0.0, io)
        if DEBUG: print('calibrate reverse io: {} so: {} sg: {}'.format(io, so, sg))
        if 'in' in stritem:
            scale = self.scale(stritem)
            ino = (so - offset * self.volts_per_step(stritem)) / scale
            ing = sg / scale
        elif 'out' in stritem:
            scale = self.scale(stritem)
            ino = (so - offset) / scale
            ing = sg / scale
        else:
            ino = so
            ing = sg

        return ino, ing

    def dc(self, io, data):
        stritem, offset = self._convert("OFFSETS", 0.0, io, cuo = self._curve_set)
        if DEBUG: print('calibrate curve io: {} data: {}'.format(stritem, ao))
        if 'in' in stritem:
            discard, scale = self._convert("SCALES", 1.0, stritem, cuo = self._curve_set)
            outo = offset + data * scale
        else:
            outo = data

        return outo

    def ac(self, io, data):
        stritem, scale = self._convert("SCALES", 1.0, io, cuo = self._curve_set)
        if 'in' in stritem:
            outo = data * scale
        else:
            outo = data

        return outo

    def dc_raw(self, io, data):
        stritem, offset = self._convert("OFFSETS", 0.0, io)
        if DEBUG:
            print('calibrate raw: {} io: {}'.format(data, io))
            print('stritem: {} offset: {} volts_per_step: {}'.format(stritem, offset, self.volts_per_step(stritem)))

        if 0 < len(stritem):
            outo = (data - offset) * self.volts_per_step(stritem)
            if DEBUG: print(outo)
        else:
            outo = data

        return outo

    def load(self, name = 'caldata'):
        try:
            with open("{}.json".format(name), 'r') as fh:
                d = fh.read(60000)
                j = json.loads(d)
        except:
            return False

        if isinstance(j ,dict):
            for k, v in j.items():
                if v is type(list):
                    self.data[k] = np.asarray(v, dtype=float)
                else:
                    self.data[k] = v

        return True

    def save(self, name):
        with open("{}.json".format(name), 'w') as fh:
            io = {}
            for k in self.data.keys():
                if isinstance(self.data[k], np.ndarray):
                    io[k] = self.data[k].tolist()
                else:
                    io[k] = self.data[k]

            if DEBUG: print(io)
            fh.write( json.dumps( io ) )

    def _get_val(self, callist, default, val, cuo):
        rtn = default
        stritem = val
        found = False
        if type(stritem) is str and hasattr(self, stritem):
            rtn = self.data[callist][getattr(self, stritem) + cuo]
            found = True
            if DEBUG: print('io port: {} rtn: {}'.format(stritem, rtn))
        else:
            stritem = ''

        return found, stritem, rtn

    def _convert(self, callist, default, val, cuo = 0):
        found, stritem, rtn = self._get_val(callist, default, val, cuo)
        if not type(val) is str:
            stritem = ''
            attrs = dir(val)
            if 'output_direct' in attrs:
                found, stritem, rtn = self._get_val(callist, default, val.output_direct, cuo)
                if DEBUG: print('output_direct: {} rtn: {}'.format(val.output_direct, rtn))

            if 'input' in attrs:
                if not found:
                    found, stritem, rtn = self._get_val(callist, default, val.input, cuo)
                else:
                    print('WARNING CalData {} also contains input'.format(val.__class__))

        return stritem, rtn

class Meta(object):
    def __init__(self):
        self._startdir = None
        self._finish_init()

    def _finish_init(self):
        self._startdir = dir(self)

    def get(self, name):
        """get the attribute"""
        return self.getcreate(name)

    def set(self, name, val):
        """create / update the attribute value"""
        setattr(self, name, val)
        return getattr(self, name)

    def getcreate(self, name, default = None):
        """get the attribute or create if not None"""
        _inst = None
        if hasattr(self, name):
            _inst = getattr(self, name)
        elif not default is None:
            setattr(self, name, default)
            _inst = getattr(self, name)

        return _inst

    def pop(self, name):
        _inst = self.get(name)
        if not name in self._startdir:
            if hasattr(self, name):
                delattr(self, name)

        return _inst

    def dump(self):
        return ((a, self.get(a)) for a in dir(self) if not a in self._startdir)

class MetaTest(Meta):
    def __init__(self, testname, tag):
        self._fname = ''.join(s for s in testname if s.isalnum and not s == ' ')
        super().__init__()
        self.getcreate('testname', default = testname)
        self.getcreate('tag', default = tag)

    @property
    def name(self):
        return self.testname
    @property
    def filename(self):
        return self.create_filename(self.tag)

    def create_filename(self, tag):
        return '{}_{}'.format(self._fname, tag)

    def clear(self):
        # create the list of attributes to delete
        names = [k for (k, v) in self.dump()]
        for k in names:
            self.pop(k)

    def load(self):
        try:
            with open('{}.json'.format(self.filename), 'r') as fh:
                d = fh.read(60000)
                j = json.loads(d)
        except:
            return False

        if isinstance(j, dict):
            for k in j:
                self.set(k, j[k])

        return True

    def save(self, fname = None):
        if fname is None:
            fname = self.filename

        with open('{}.json'.format(fname), 'w') as fh:
            fh.write( json.dumps( dict(self.dump()) ) )

class DummyR(object):
    class asg(object):
        def __init__(self, n):
            setattr(self, 'name', 'asg{}'.format(n))
            setattr(self, 'frequency', 0)
        def setup(self, *arg, **kwargs):
            print('{} setup: {} kwargs: {}'.format(self.name, arg, kwargs))
            for k in kwargs:
                setattr(self, k, kwargs[k])

    class iq(object):
        def __init__(self, n):
            setattr(self, 'name', 'iq{}'.format(n))
        def setup(self, *arg, **kwargs):
            print('{} setup: {} kwargs: {}'.format(self.name, arg, kwargs))
            for k in kwargs:
                setattr(self, k, kwargs[k])

    class scope(object):
        def __init__(self):
            setattr(self, 'name', 'scope')
            attrs = ['sampling_time', 'duration', 'decimation', 'input1', 'input2']
            for k in attrs:
                setattr(self, k, 0)

        def setup(self, *arg, **kwargs):
            print('{} setup: {} kwargs: {}'.format(self.name, arg, kwargs))
            for k in kwargs:
                setattr(self, k, kwargs[k])

    def __init__(self):
        for n in range(2):
            setattr(self, 'asg{}'.format(n), self.asg(n))
        for n in range(3):
            setattr(self, 'iq{}'.format(n), self.iq(n))

        setattr(self, 'scope', self.scope())


class DummyP(object):
    class spectrumanalyzer(object):
        def __init__(self):
            attrs = ['center', 'decimation', 'rbw']
            for k in attrs:
                setattr(self, k, 0)

            setattr(self, 'frequencies', np.arange(1000, 10000, 100, dtype=float))
            setattr(self, 'data_length', len(self.frequencies))
            setattr(self, 'scope', DummyR.scope())

        def setup(self, *arg, **kwargs):
            print('spectrumanalyzer setup: {} kwargs: {}'.format(arg, kwargs))
            for k in kwargs:
                setattr(self, k, kwargs[k])

        def single(self):
            dummy = np.zeros((4, self.frequencies.shape[0]), dtype = float)
            dummy[0, 5] = 1.0
            return dummy

        def curve_ready(self):
            return True

        def data_to_display_unit(self, x, rbw):
            return x

    def __init__(self):
        setattr(self, 'spectrumanalyzer', self.spectrumanalyzer())

class TestBench(Meta):
    def __init__(self, test):
        HOSTNAME = 'rp-f0bd75'
        #self._p = DummyP()
        #self._r = DummyR()
        self._p = pyrpl.Pyrpl(config = test.rpconf, hostname = HOSTNAME, gui = False)
        self._r = self.p.rp
        #start_logging()
        self._name = test.name
        self._test = test

        # only call __init__ after all attributes have been created
        super().__init__()

    @property
    def name(self):
        return self._name
    @property
    def p(self):
        return self._p
    @property
    def r(self):
        return self._r
    @property
    def test(self):
        return self._test

    def set_test_ifcal(self, name, default = None):
        if not hasattr(self.test, 'calibrate'):
            rtn = self.test.set(name, default)
        elif self.test.calibrate:
            rtn = self.test.set(name, default)
        else:
            rtn = self.test.getcreate(name, default)

        return rtn

    def getcreate(self, name, default = None):
        if hasattr(self, name):
            # TestBench overrides others
            _inst = getattr(self, name)
        elif hasattr(self.p, name):
            # attribute from Pyrpl overrides others
            _inst = getattr(self.p, name)
        elif hasattr(self.r, name):
            # attribute from RedPitaya
            _inst = getattr(self.r, name)
        else:
            _inst = super().getcreate(name, default)

        return _inst

    def run(self, seq):
        for func in seq:
            globals()[func](self)

def start_logging():
    import logging

    formatter = logging.Formatter('%(levelname)s : %(name)s : %(message)s')

    # setup optional console logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # get the root logger
    logger = logging.getLogger('')

    logger.handlers.clear()
    #logger.addHandler(console)

    filename = 'pyrpl.log'

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)

    # send debug logs to filehandler
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    msg = 'pyrpl version: {}'.format(pyrpl.__version__)
    logger.info(msg)

    levels = {'pyrpl.hardware_modules.iir': logging.DEBUG,
              'pyrpl.hardware_modules.iir.iir': logging.DEBUG,
              'pyrpl.hardware_modules.iir.iir_theory': logging.DEBUG,
              'pyrpl.modules': logging.DEBUG,
              'pyrpl': logging.WARNING,
              'pyrpl.async_utils': logging.WARNING,
              'pyrpl.redpitaya': logging.WARNING,
              'pyrpl.memory': logging.WARNING,
              'pyrpl.sshshell': logging.WARNING, # set level in __init__.py
              'pyrpl.widgets.startup_widget': logging.WARNING}

    # switch module logging level
    for module in levels.keys():
        logger = logging.getLogger(module)
        logger.setLevel(levels[module])
        if levels[module] == logging.INFO:
            logger.addHandler(console)

def prepare_to_show_plot(p):
    import matplotlib
    for i in range(1):
        try:
            matplotlib.use('TkAgg')
            break
        except:
            #print('{} qt is still running!'.format(10-i))
            pass

def calc_mean(stream):
    samples = len(stream)
    mean = np.sum(stream) / samples
    std = np.std(stream)
    #print(mean, std)

    return mean, std, samples

def remove_outliers(stream, mean, std):
    core = stream[np.nonzero(stream < (mean + 3*std))]
    #print(len(core))
    core = core[np.nonzero((mean - 3*std) < core)]
    if 0 < len(core):
        #print(len(core))
        mean, std, samples = calc_mean(core)
    else:
        samples = len(stream[np.nonzero(stream == mean)])

    return mean, std, samples

def calc_range(stream):
    smax = stream.max()
    smin = stream.min()
    mean = (smax + smin) / 2
    above = stream[np.nonzero(mean < stream)]
    if 0 < len(above):
        #print(mean, len(above))
        upper, ustd, samples_above = calc_mean(above)
    else:
        upper = mean
        ustd = 0
        samples_above = len(above)

    # now remove outliers
    upper, ustd, samples_above = remove_outliers(stream, upper, ustd)

    below = stream[np.nonzero(stream < mean)]
    if 0 < len(below):
        #print(mean, len(below))
        lower, lstd, samples_below = calc_mean(below)
    else:
        lower = mean
        ustd = 0
        samples_below = len(below)

    # now remove outliers
    lower, lstd, samples_below = remove_outliers(stream, lower, lstd)

    mid = (upper + lower) / 2

    # print(lower, mid, upper, samples_above, samples_below)
    if not len(stream) == samples_above + samples_below:
        if DEBUG: print('signal / noise: {:.2f}'.format(
            (samples_above + samples_below) / (len(stream) - samples_above - samples_below)))

    ratio = 0
    if 0 < samples_above + samples_below:
        ratio = samples_above / (samples_above + samples_below)

    return lower, mid, upper - lower, samples_above, samples_below, ratio

def setup_asg(t):
    out = 'out{}'.format(t.counter + 1)
    asg_o, asg_a = t.c.cal(out, t.test.asg_raw_o, ag = t.asg_raw_a)
    if DEBUG: print('{} offset: {} asg amplitude: {}'.format(t.asg.name, asg_o, asg_a))
    t.asg.setup(frequency = t.test.fg, waveform = t.waveform,
              offset = asg_o, amplitude = asg_a,
              output_direct = out,
              trigger_source = 'immediately'
              )

    if t.test.verbose:
        print('{} offset: {} asg amplitude: {}'.format(t.asg.name, t.asg.offset, t.asg.amplitude))
        print('generator frequency: {} waveform: {} offset: {} amplitude: {}'.format(
            t.asg.frequency, t.asg.waveform, t.test.asg_raw_o, t.asg_raw_a))

    t.test.fg = t.asg.frequency

def setup_scope(t):
    acquire_length = 50
    requested_samples_per_cycle = 200

    t.s.sampling_time = 1 / (t.test.fg * requested_samples_per_cycle)
    samples_per_cycle = 1 / (t.test.fg * t.s.sampling_time)
    fs = 1 / t.s.sampling_time

    if t.test.verbose:
        print('Sampling rate: {:0.0f}'.format(fs))
        print('Sample time: {} Full buffer in: {} Selected samples per cycle: {} decimation: {}'.format(
            t.s.sampling_time, t.s.duration, samples_per_cycle, t.s.decimation))

    samples_in_buffer = 16384
    t.set('samples_acquired' , int(acquire_length * samples_per_cycle))
    if samples_in_buffer < t.samples_acquired:
        print('WARNING: buffer overflow: {}'.format(t.samples_acquired))

    trigger = 0.0
    hysteresis = 0.01
    trigger_delay = 0.0
    t.s.setup(threshold_ch1 = trigger,
            hysteresis_ch1 = hysteresis,
            trigger_source = 'immediately',
            trigger_delay = trigger_delay,
            trace_average = 1)

def setup_channels(t):
    t.getcreate('s', t.getcreate('scope'))
    t.run(['setup_scope'])
    ip = []
    for t.counter in range(2):
        asg = 'asg{}'.format(t.counter)
        t.getcreate('asg')
        t.asg = t.getcreate(asg)
        t.run(['setup_asg'])
        port = 1 - t.counter if t.test.reverse_ports else t.counter
        ip.append('in{}'.format(port + 1))

    t.s.input1 = ip[0]
    t.s.input2 = ip[1]
    t.getcreate('rinput', default = t.s.input1)
    if t.test.verbose: print('scope inputs: {} {}'.format(t.s.input1, t.s.input2))

def switch_waveform(t):
    if t.test.verbose: print('switch waveform {}'.format(t.asg.name))
    asg_o, asg_a = t.c.cal(t.asg.output_direct, t.test.asg_raw_o, ag = t.asg_raw_a)
    if DEBUG: print('{} offset: {} asg amplitude: {}'.format(t.asg.name, asg_o, asg_a))
    t.asg.setup(offset = asg_o, amplitude = asg_a, waveform = t.waveform)
    if t.test.verbose:
        print('{} offset: {} asg amplitude: {}'.format(t.asg.name, t.asg.offset, t.asg.amplitude))
        print('generator frequency: {} waveform: {} offset: {} amplitude: {}'.format(
            t.asg.frequency, t.asg.waveform, t.test.asg_raw_o, t.asg_raw_a))

def update_asg(t):
    asg_o, asg_a = t.c.cal(t.asg.output_direct, t.test.asg_raw_o, ag = t.asg_raw_a)
    if DEBUG: print('{} offset: {} asg amplitude: {}'.format(t.asg.name, asg_o, asg_a))
    t.asg.setup(offset = asg_o, amplitude = asg_a)
    if t.test.verbose:
        print('{} offset: {} asg amplitude: {}'.format(t.asg.name, t.asg.offset, t.asg.amplitude))
        print('generator frequency: {} waveform: {} offset: {} amplitude: {}'.format(
            t.asg.frequency, t.asg.waveform, t.test.asg_raw_o, t.asg_raw_a))

def config_channels(t):
    for t.counter in range(2):
        asg = 'asg{}'.format(t.counter)
        t.asg = t.getcreate(asg)
        t.run([t.oper])

def switch_waveforms(t):
    t.oper = 'switch_waveform'
    config_channels(t)

    # give time for the generator to change waveforms
    sleep(2)

def update_asgs(t):
    t.oper = 'update_asg'
    config_channels(t)

def channel_oper(t):
    for t.counter in range(2):
        channel = 'input{}'.format(t.counter + 1)
        if hasattr(t.s, channel):
            t.rinput = getattr(t.s, channel)
            t.run([t.oper])

def capture_raw_trace(t):
    stream1 = np.zeros([0], dtype=np.int16)
    stream2 = np.zeros([0], dtype=np.int16)
    t.s._start_acquisition_rolling_mode()

    n = 0
    w1 = 0
    while True:
        if n % 10 == 0:
            if t.test.verbose: print('w1: {} Time = {:.4f}'.format(w1, len(stream1) * t.s.sampling_time))

        w0 = w1
        w1 = t.s._write_pointer_current
        stream1 = np.concatenate((stream1, t.s._rawdata_ch1[w0:w1]))
        stream2 = np.concatenate((stream2, t.s._rawdata_ch2[w0:w1]))

        if t.samples_acquired < w1:
            if t.test.verbose: print('w1: {} Time = {:.4f}'.format(w1, len(stream1) * t.s.sampling_time))
            break

        n += 1

    duration = len(stream1) * t.s.sampling_time
    if t.test.verbose: print('Finished in = {:.4f}s'.format(duration))

    samples_acquired = len(stream1) if len(stream1) < t.samples_acquired else t.samples_acquired

    t.set('times', np.arange(len(stream1)) * t.s.sampling_time)
    results = np.stack((stream1, stream2), axis=0)
    t.set('results', results)
    t.set('cuo', 0)
    t.set('results_to_plot', results)
    t.set('samples_to_plot', samples_acquired)
    t.set('ylabel', 'raw')
    t.set('fncal', t.c.dc_raw)
    t.set('plotfn', 'raw')

def capture_curve_trace(t):
    # setup the scope for an acquisition
    curve = t.s.single()

    # trigger should fire immediately
    if t.s.curve_ready():
        curve = t.s.single()

    # check that the trigger has been disarmed
    if t.test.verbose:
        print('After trigger:')
        print('Curve ready: {}'.format(t.s.curve_ready()))
        print('Trigger event age [ms]: {}'.format(
            8e-9 * ((t.s.current_timestamp & 0xFFFFFFFFFFFFFFFF) - t.s.trigger_timestamp) * 1000))

        print('Sample time: {} Full buffer in: {} decimation: {}'.format(
            t.s.sampling_time, t.s.duration, t.s.decimation))

    t.set('samples_to_plot', 0)
    if t.s.curve_ready():
        # calibrate the results_to_plot
        times = t.s.times * 1e3

        samples_acquired = len(times) if len(times) < t.samples_acquired else t.samples_acquired
        t.set('times', times)
        t.set('results', curve)
        t.set('cuo', t.c._curve_set)
        t.set('results_to_plot', curve)
        t.set('samples_to_plot', samples_acquired)
        t.set('ylabel', 'Volts')
        t.set('fncal', t.c.dc)
        t.set('plotfn', 'curve')
    else:
        print('WARNING capture_curve_trace: no results captured')

def get_index(val, cuo = 0):
    rtn = 999
    stritem = val
    found = False
    if type(stritem) is str and hasattr(t.c, stritem):
        rtn = getattr(t.c, stritem) + cuo
        found = True
    else:
        stritem = ''

    return found, stritem, rtn

def analyse_result(t):
    found = False
    if 0 < t.samples_to_plot:
        if t.asg_raw_a == 0.0:
            mean, std, samples = calc_mean(t.results[t.counter, :])
            found, stritem, idx = get_index(t.rinput, cuo = t.cuo)
            if found:
                t.zeroffsets[idx] = mean
                if t.test.verbose: print('{} mean: {} std: {}'.format(stritem, mean, std))
        else:
            lower, mid, irange, samples_above, samples_below, ratio = calc_range(t.results[t.counter, :])
            found, stritem, idx = get_index(t.rinput, cuo = t.cuo)
            if found:
                t.squareoffsets[idx] = mid
                t.ranges[idx] = irange
                if t.test.verbose: print('{} mid point: {} input range: {}'.format(stritem, mid, irange))

    if not found:
        print('WARNING analyse_result: no results to process')

def analyse_results(t):
    t.oper = 'analyse_result'
    channel_oper(t)

def plot_calibrated_results(t):
    r1 = t.fncal(t.s.input1, t.results_to_plot[0,:]) * t.test.input_attenuation
    r2 = t.fncal(t.s.input2, t.results_to_plot[1,:]) * t.test.input_attenuation
    t.results_to_plot = np.stack((r1, r2), axis=0)

    plot_results(t)
    print('measurements for {}'.format(t.plotfn))
    print('======================')
    if t.asg_raw_a == 0.0:
        mean, std, samples = calc_mean(r1)
        print('{} mean: {} std: {}'.format(t.s.input1, mean, std))
        mean, std, samples = calc_mean(r2)
        print('{} mean: {} std: {}'.format(t.s.input2, mean, std))
    else:
        lower, mid, irange, samples_above, samples_below, ratio = calc_range(r1)
        print('{} mid point: {} input range: {}'.format(t.s.input1, mid, irange))
        lower, mid, irange, samples_above, samples_below, ratio = calc_range(r2)
        print('{} mid point: {} input range: {}'.format(t.s.input2, mid, irange))

def plot_results(t):
    if 0 < t.samples_to_plot:
        import matplotlib.pyplot as plt

        # plot the data
        plt.close()
        fig = plt.figure()
        plt.ion()
        plt.plot(t.times[0:t.samples_to_plot], t.results_to_plot[0,0:t.samples_to_plot], label = t.s.input1)
        plt.plot(t.times[0:t.samples_to_plot], t.results_to_plot[1,0:t.samples_to_plot], label = t.s.input2)

        plt.legend(loc = 'center left', bbox_to_anchor = (1.04, 0.5))
        plt.xlabel('Time [ms]')
        plt.ylabel(t.ylabel)
        plt.title(t.ylabel)
        plt.tight_layout()

        if hasattr(t, 'svg') and t.svg:
            plt.savefig('calibrate_{}.svg'.format(t.plotfn))
        else:
            plt.pause(3.0)

def prep_zero_reading(t):
    print('Attach scope to earth and DC meter to signal generators')
    resp = input('Press RTN when ready or "q" to quit: ')
    if 'q' in resp:
        raise StopIteration

    t.asg_raw_a = 0.0
    t.waveform = 'dc'

def get_zero_reading(t):
    resp = input('Enter voltage readings (1lo, 1hi, 2lo, 2hi) or "q" to quit: ')
    if 'q' in resp:
        raise StopIteration

    parts = resp.split(',')
    print('readings: {}'.format(parts))
    if 3 < len(parts):
        for count in range(2):
            asg = 'asg{}'.format(count)
            out = t.getcreate(asg).output_direct
            found, stritem, idx = get_index(out, cuo = 0)
            if found:
                t.zeroffsets[idx] = (float(parts[count*2]) + float(parts[count*2 + 1])) / 2.0
                if t.test.verbose: print('{} offset: {}'.format(stritem, t.zeroffsets[idx]))

                # calculate the offset correction
                t.zeroffsets[idx] = t.c.offsets[idx] - t.zeroffsets[idx]
            else:
                print('WARNING get_zero_reading: no results to process')

def prep_square_reading(t):
    print('Attach scope to signal generators and AC meter to generators')
    resp = input('Press RTN when ready or "q" to quit: ')
    if 'q' in resp:
        raise StopIteration

    t.asg_raw_a = t.test.asg_raw_a
    t.waveform = 'square'

def get_squarewave_reading(t):
    resp = input('Enter voltage readings (1lo, 1hi, 2lo, 2hi) or "q" to quit: ')
    if 'q' in resp:
        raise StopIteration

    parts = resp.split(',')
    print('readings: {}'.format(parts))
    if 3 < len(parts):
        for count in range(2):
            asg = 'asg{}'.format(count)
            out = t.getcreate(asg).output_direct
            found, stritem, idx = get_index(out, cuo = 0)
            if DEBUG: print(out, idx)
            if found:
                t.ranges[idx] = float(parts[count*2]) + float(parts[count*2 + 1])
                if t.test.verbose: print('{} range: {}'.format(stritem, t.ranges[idx]))
                # asg required output was calibrated earlier
                # so we need to adjust the existing calibration
                t.scales[idx] = 2.0 * t.asg_raw_a * t.c.scales[idx] / t.ranges[idx]
            else:
                print('WARNING get_squarewave_reading: no results to process')

def calculate_scale(t):
    for count in range(2):
        asg = 'asg{}'.format(count)
        out = t.getcreate(asg).output_direct
        found, stritem, oidx = get_index(out, cuo = 0)
        channel = 'input{}'.format(count + 1)
        if hasattr(t.s, channel):
            sinput = getattr(t.s, channel)
        else:
            print('WARNING calculate_scale: cant calibrate')
            continue

        found, stritem, idx = get_index(sinput, cuo = 0)
        if DEBUG: print(sinput, idx)
        if found:
            t.volts_per_steps[idx] = t.ranges[oidx] / (t.test.input_attenuation * t.ranges[idx])
            t.scales[idx] = t.volts_per_steps[idx] / t.c.volts_per_bit_ideal

        found, stritem, idx = get_index(sinput, cuo = t.c._curve_set)
        if DEBUG: print(sinput, idx)
        if found:
            t.scales[idx] = t.ranges[oidx] / (t.test.input_attenuation * t.ranges[idx])

            # we add the offset so apply the scale to the offset
            # and change the sign to cancel the reading
            t.zeroffsets[idx] = - t.zeroffsets[idx] * t.scales[idx]

def update_calibration_data(t):
    for idx in range(t.volts_per_steps.shape[0]):
        t.c.volts_per_steps[idx] = t.volts_per_steps[idx]
        t.c.scales[idx] = t.scales[idx]
        t.c.offsets[idx] = t.zeroffsets[idx]

    print('Existing calibration data')
    print('=========================')
    print(t.c)
    print('New calibration data')
    print('====================')
    print('volts_per_steps\n{}'.format(t.volts_per_steps))
    print('scales\n{}'.format(t.scales))
    print('offsets\n{}'.format(t.zeroffsets))
    print('square wave mid points\n{}'.format(t.squareoffsets))
    print('ranges\n{}'.format(t.ranges))

def save_calibration(t):
    t.c.save(t.test.dataname)

def do_calibration(t):
    try:
        t.run(['prep_zero_reading', 'setup_channels',
               'capture_raw_trace', 'analyse_results',
               'capture_curve_trace', 'analyse_results',
               'get_zero_reading',
               'prep_square_reading', 'switch_waveforms',
               'capture_raw_trace', 'analyse_results',
               'capture_curve_trace', 'analyse_results',
               'get_squarewave_reading',
               'calculate_scale', 'update_calibration_data',
               'save_calibration'])
    except StopIteration:
        print('stopping')

    t.set('svg', True)
    update_asg(t)
    sleep(1)
    capture_raw_trace(t)
    plot_calibrated_results(t)
    sleep(2)
    capture_curve_trace(t)
    plot_calibrated_results(t)

def get_run_conf(**kargs):
    progname = 'calibrate'
    parser = argparse.ArgumentParser(
        prog = progname,
        description = 'Simple calibration for pyrpl')
    parser.add_argument('--verbose', help='verbose output', action = 'store_true', default = False)
    parser.add_argument('--square', help='square wave test', action = 'store_true', default = False)
    parser.add_argument('--zero', help='zero voltage test', action = 'store_true', default = False)
    parser.add_argument('--reverse', help='reverse the ports in*', action = 'store_true', default = False)
    args = parser.parse_args()
    if DEBUG: print('calibration args: {}'.format(args))

    # command line arguments override kargs overrides config file
    testname = kargs['testname'] if 'testname' in kargs else 'cal'
    tag = ''
    test = MetaTest(testname, tag)

    probe_attenuation = 1
    test.set('input_attenuation', 20 * probe_attenuation)
    test.set('dataname', 'caldata')
    test.set('asg_raw_a', 0.9)
    test.set('asg_raw_o', 0.0)
    test.set('fg', 50)
    test.set('reverse_ports', False)
    reverse = True if args.reverse else test.reverse_ports
    test.reverse_ports = reverse

    RPCONFIG = 'calibrate'
    test.set('rpconf', RPCONFIG)

    test.set('verbose', False)
    verbose = kargs['verbose'] if 'verbose' in kargs else test.verbose
    verbose = True if args.verbose else verbose
    test.verbose = verbose

    test.set('verbose', False)
    verbose = kargs['verbose'] if 'verbose' in kargs else test.verbose
    verbose = True if args.verbose else verbose
    test.verbose = verbose

    test.set('squaretest', False)
    squaretest = True if args.square else test.squaretest
    test.squaretest = squaretest

    test.set('zerotest', False)
    zerotest = True if args.zero else test.zerotest
    test.zerotest = zerotest

    if test.verbose: print('test config: {}'.format([(k, v) for k,v in test.dump()]))

    return test

if __name__ == '__main__':
    test = get_run_conf(testname = 'calibration test')

    c = CalData()
    c.load()

    t = TestBench(test)
    t.set('c', c)

    t.set('volts_per_steps', np.ones((8), dtype=float))
    t.volts_per_steps = t.volts_per_steps * t.c.volts_per_bit_ideal
    t.set('ranges', np.ones((8), dtype=float))
    t.ranges = t.ranges * t.test.asg_raw_a * 2.0
    t.ranges[0] = t.ranges[0] / (t.c.volts_per_bit_ideal * t.test.input_attenuation)
    t.ranges[1] = t.ranges[1] / (t.c.volts_per_bit_ideal * t.test.input_attenuation)
    t.set('scales', np.ones((8), dtype=float))
    t.set('zeroffsets', np.zeros((8), dtype=float))
    t.set('squareoffsets', np.zeros((8), dtype=float))

    t.set('asg_raw_a', 0.0)
    t.set('waveform', 'dc')

    if t.test.squaretest or t.test.zerotest:
        # test the results
        try:
            if t.test.squaretest:
                t.asg_raw_a = t.test.asg_raw_a
                t.waveform = 'square'

            t.run(['setup_channels', 'switch_waveforms',
                   'capture_raw_trace', 'plot_calibrated_results',
                   'capture_curve_trace', 'plot_calibrated_results'])
            sleep(2)
        except StopIteration:
            print('stopping')
    else:
        do_calibration(t)
