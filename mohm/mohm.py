#!/usr/bin/env python3

import argparse
import datetime
import json
import math
import numpy as np
from pathlib import Path
import pyrpl
from pyrpl.async_utils import sleep
import string

DEBUG = False

class CalData(object):
    def __init__(self):
        self._volts_per_bit_ideal = 1 / 2. ** 13  # 122.07uV
        self._curve_set = 4
        self.caldata = {}

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
        if 0 < len(stritem):
            outo = (data - offset) * self.volts_per_step(stritem)
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
        if not name == 'caldata':
            with open("{}.json".format(name), 'w') as fh:
                io = {}
                for k in self.data.keys():
                    if isinstance(self.data[k], np.ndarray):
                        io[k] = self.data[k].tolist()
                    else:
                        io[k] = self.data[k]

                fh.write( json.dumps( io ) )

    def _get_val(self, callist, default, val, cuo):
        rtn = default
        stritem = val
        found = False
        if type(stritem) is str and hasattr(self, stritem):
            rtn = self.data[callist][getattr(self, stritem) + cuo]
            found = True
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

    logger = logging.getLogger('pyrpl')
    #logger = logging.getLogger('pyrpl.sshshell')
    #print('logger handlers: {}'.format(logger.handlers))
    logger.handlers.clear()
    fh = logging.FileHandler('pyrpl.log')
    #fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    #print('logger handlers: {}'.format(logger.handlers))

    logger.info('pyrpl version: {}'.format(pyrpl.__version__))

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

def calc_freq(fi_low = None, fir = None, fis = None, fi_high = None):
    #print('>calc_freq fi_low: {} fir: {} fis: {} fi_high: {}'.format(fi_low, fir, fis, fi_high))

    fr = fir
    if not fr is None:
        r = round(1.25e8 / fr)
        fr = 1.25e8 / r

    fs = fis
    if not fs is None:
        s = round(1.25e8 / fs)
        fs = 1.25e8 / s

    f_low = fi_low
    f_high = fi_high
    if fs is None and fr is None:
        s = round((f_low + f_high)/2.0)
        fs = 1.25e8 / s
        r = round((f_high - f_low)/2.0)
        fr = 1.25e8 / r

    if f_low is None and fr is None:
        r = round(1.25e8 / (f_high - fs))
        fr = 1.25e8 / r

    if f_high is None and fr is None:
        r = round(1.25e8 / (fs - f_low))
        fr = 1.25e8 / r

    if fs is None and f_high is None:
        s = round(1.25e8 / (f_low + fr))
        fs = 1.25e8 / s

    if fs is None and f_low is None:
        s = round(1.25e8 / (f_high - fr))
        fs = 1.25e8 / s

    f_low = fs - fr
    f_high = fs + fr

    f_low = 0 if f_low < 0 else f_low
    fr = 0 if fr < 0 else fr
    fs = 0 if fs < 0 else fs
    f_high = 0 if f_high < 0 else f_high
    if DEBUG: print('<calc_freq f_low: {:0.0f} fr: {:0.0f} fs: {:0.0f} f_high: {:0.0f}'.format(f_low, fr, fs, f_high))

    return f_low, fr, fs, f_high

def calc_decimation(fi_low = None, fir = None, fis = None, fi_high = None):
    if DEBUG: print('>calc_decimation fi_low: {} fir: {} fis: {} fi_high: {}'.format(fi_low, fir, fis, fi_high))

    f_low = fi_low
    l = None
    if not f_low is None:
        l = round(1.25e8 / f_low)

    fr = fir
    r = None
    if not fr is None:
        r = round(1.25e8 / fr)
        fr = 1.25e8 / r

    fs = fis
    s = None
    if not fs is None:
        s = round(1.25e8 / fs)
        fs = 1.25e8 / s

    f_high = fi_high
    h = None
    if not f_high is None:
        h = round(1.25e8 / f_high)

    if fs is None and fr is None:
        s = round((f_low + f_high)/2.0)
        fs = 1.25e8 / s
        r = round((f_high - f_low)/2.0)
        fr = 1.25e8 / r

    if f_low is None and fr is None:
        r = round(1.25e8 / (f_high - fs))
        fr = 1.25e8 / r

    if fs is None and f_high is None:
        s = round(1.25e8 / (f_low + fr))
        fs = 1.25e8 / s

    f_low = fs - fr
    f_high = fs + fr

    l = 0 if f_low < 0 else 1.25e8 / f_low
    f_low = 0 if f_low < 0 else f_low
    r = 0 if fr < 0 else 1.25e8 / fr
    fr = 0 if fr < 0 else fr
    s = 0 if fs < 0 else 1.25e8 / fs
    fs = 0 if fs < 0 else fs
    h = 0 if f_high < 0 else 1.25e8 / f_high
    f_high = 0 if f_high < 0 else f_high
    if DEBUG:
        print('<calc_decimation f_low: {:0.0f} fr: {:0.0f} fs: {:0.0f} f_high: {:0.0f}'.format(f_low, fr, fs, f_high))
        print('<calc_decimation l: {:0.0f} r: {:0.0f} s: {:0.0f} h: {:0.0f}'.format(l, r, s, h))

    return l, r, s, h

def characterise_spectrum(data, frequencies, condition):
    idxs = np.where(condition)
    fminidx = idxs[0].min()
    fmaxidx = idxs[0].max()
    amaxidx = fminidx + np.argmax(data[fminidx:fmaxidx])
    amaxidx_lo = amaxidx - 1 if 0 < amaxidx else 0
    amaxidx_hi = amaxidx + 1 if amaxidx + 1 < len(data) else len(data) - 1

    #print(idxs, amaxidx_lo, amaxidx, amaxidx_hi)

    return amaxidx_lo, amaxidx, amaxidx_hi

def is_similar(f, cf):
    d = 1.25e8 / f
    c = 1.25e8 / cf

    return c - 2 < d and d < c + 2

def load_calibration(t, seq):
    filename = '{}_{}_cal.npz'.format(t.test.dataname, t.test.tag)
    rtn = seq if t.test.testype == 'freq' else t.test.fs
    try:
        with np.load(filename) as data:
            if t.test.testype == 'freq' and 'test_vector' in data and data['test_vector'].shape[0] == seq.shape[0]:
                rtn = freq_calibration(t, seq, data)
            elif 'test_vector' in data and 0 < data['test_vector'].shape[0]:
                rtn = amp_calibration(t, seq, data)
    except:
        pass

    return rtn

def amp_calibration(t, asg_raw_a, data):
    fc = data['test_vector'][0, t.freq_idx]
    similar = is_similar(t.test.fs, fc)
    rtnfs = fc if similar else t.test.fs
    if similar and 'results' in data:
        t.getcreate('results_cal', data['results'])
        print('calibration data: {}'.format(t.results_cal))

    return rtnfs

def freq_calibration(t, fs, data):
    f0 = data['test_vector'][0, t.freq_idx]
    fe = data['test_vector'][-1, t.freq_idx]
    #print(f0, fe)
    similar = is_similar(fs[0], f0) and is_similar(fs[-1], fe)
    rtnfs = data['test_vector'][:, t.freq_idx] if similar else fs
    if similar and 'results' in data:
        t.getcreate('results_cal', data['results'])
        if DEBUG: print('calibration data: {}'.format(t.results_cal))

    return rtnfs

def freq_list(start, end, count):
    interval = (end - start) / count

    freqs = np.array([], dtype = float)

    freq = start
    f = -1.0
    n = 0
    while freq < end:
        fo = f
        r = round(1.25e8 / freq)
        f = 1.25e8 / r
        freq = freq + interval
        if not f == fo:
            freqs = np.append(freqs, [f])
        else: print('skip {}'.format(freq))

        n += 1
        if not n < count:
            break

    return freqs

def freq_log_list(start, end, count):
    start_l = math.log10(start)
    end_l = math.log10(end)
    interval = (end_l - start_l) / count

    freqs = np.array([], dtype = float)

    freq = start
    f = -1.0
    n = 0
    while freq < end:
        fo = f
        fl = math.log10(freq)
        r = round(1.25e8 / freq)
        f = 1.25e8 / r
        fl = fl + interval
        freq = math.pow(10, fl)
        if not f == fo:
            freqs = np.append(freqs, [f])
        else: print('skip {}'.format(freq))

        n += 1
        if not n < count:
            break

    return freqs

def test_ranges():
    multiplef = False
    if multiplef:
        fss = 10000
        fse = 100000
        count = 10
        fs = freq_list(fss, fse, count)
        fr = np.ones(fs.shape, dtype = float)
        fir = 6000
        r = round(1.25e8 / fir)
        fr = fr * 1.25e8 / r
    else:
        frs = 5000
        fre = 7000
        count = 10
        fr = freq_list(frs, fre, count)
        fs = np.ones(fr.shape, dtype = float)
        fis = 30000
        s = round(1.25e8 / fis)
        fs = fs * 1.25e8 / s

    count = 100
    ars = 0.1
    are = 0.98
    step = (are - ars) / count
    asg_raw_a = np.arange(ars, are, step, dtype=float)
    print(len(asg_raw_a), asg_raw_a[-1])

def create_signal_amplitudes(t):
    t.getcreate('amp_idx', 0)
    t.getcreate('freq_idx', 1)

    t.test.getcreate('count', 4)
    t.test.getcreate('asg_raw_a_start', 0.1)
    t.test.getcreate('asg_raw_a_end', 0.9)
    step = (t.test.asg_raw_a_end - t.test.asg_raw_a_start) / t.test.count

    asg_raw_a = np.arange(t.test.asg_raw_a_start, t.test.asg_raw_a_end, step, dtype = float)
    test_vector = np.ones((asg_raw_a.shape[0], 2), dtype = float)
    test_vector[:, t.amp_idx] = asg_raw_a
    if not t.test.calibrate:
        test_vector[:, t.freq_idx] = load_calibration(t, asg_raw_a)
    else:
        test_vector[:, t.freq_idx] = t.test.fs

    t.set('iter_count', 0)
    t.set('test_vector', test_vector)

def create_signal_freqs(t):
    t.getcreate('amp_idx', 0)
    t.getcreate('freq_idx', 1)

    t.test.getcreate('count', 4)
    t.test.getcreate('fs_start', 10000)
    t.test.getcreate('fs_end', 100000)
    fs = freq_log_list(t.test.fs_start, t.test.fs_end, t.test.count)
    if not t.test.calibrate:
        fs = load_calibration(t, fs)

    test_vector = np.ones((fs.shape[0], 2), dtype = float)
    test_vector[:, t.amp_idx] = t.test.asg_raw_a
    test_vector[:, t.freq_idx] = fs

    t.set('iter_count', 0)
    t.set('test_vector', test_vector)

def get_run_conf(**kargs):
    today = '{}'.format(''.join(datetime.date.today().isoformat().split('-')))
    progname = 'mohm'
    parser = argparse.ArgumentParser(
        prog = progname,
        description = 'Calculate voltage ratio')
    parser.add_argument('--amp', action = 'store_true', required = False)
    parser.add_argument('--freq', action = 'store_true', required = False)
    parser.add_argument('--cal', action = 'store_true', required = False)
    parser.add_argument('--tag', type = str, required = False)
    parser.add_argument('--plot', action = argparse.BooleanOptionalAction, required = False)
    args = parser.parse_args()
    # print('test args: {}'.format(args))

    # command line arguments override kargs overrides config file
    testname = kargs['testname'] if 'testname' in kargs else 'test'
    tag = kargs['tag'] if 'tag' in kargs else today
    tag = args.tag if not args.tag is None else tag
    test = MetaTest(testname, tag)
    test.load()

    # print('test config: {}'.format(test.data))

    dataname = test.getcreate('dataname', default = 'data')
    test.set('dataname', kargs['dataname'] if 'dataname' in kargs else dataname)
    rpconf = test.getcreate('rpconf', default = progname)
    test.set('rpconf', kargs['rpconf'] if 'rpconf' in kargs else rpconf)

    testype = test.getcreate('testype', default = 'amp')
    testype = kargs['testype'] if 'testype' in kargs else testype
    testype = 'freq' if args.freq else testype
    testype = 'amp' if args.amp else testype
    test.set('testype', testype)

    test.set('calibrate', True if args.cal else False)

    plot = test.getcreate('plot', default = True)
    test.set('plot', args.plot if not args.plot is None else plot)

    # prevent overwrite of previous days results
    test.tag = today

    print('test config: {}'.format([(k, v) for k,v in test.dump()]))

    return test

def test_sequence(t):
    t.set('counter', 0)
    for t.counter in range(t.test_vector.shape[0]):
        # update asg setup
        t.asg_raw_a = t.test_vector[t.counter, t.amp_idx]
        t.fs = t.test_vector[t.counter, t.freq_idx]
        asg_o, asg_a = t.c.cal(t.asg.output_direct, t.test.asg_raw_o, ag = t.asg_raw_a)
        t.asg.setup(frequency = t.fs, offset = asg_o, amplitude = asg_a)
        t.fs = t.asg.frequency
        if t.test.calibrate:
            t.test_vector[t.counter, t.freq_idx] = t.asg.frequency
        elif not t.test_vector[t.counter, t.freq_idx] == t.fs:
            print('WARNING test_sequence {} fs correction test_vector - fs: {}'.format(
                t.counter, t.test_vector[t.counter, t.freq_idx] - t.fs))

        # update iq setup
        t.f_low, fref, fs, t.f_high = calc_freq(fir = t.fref, fis = t.fs)
        flow_pass = t.test_vector[t.counter, t.freq_idx]
        applyfilter = True
        if applyfilter:
            acbandwidth = flow_pass / 2
            bandwidth = [ flow_pass, flow_pass ]
        else:
            acbandwidth = 0
            bandwidth = []

        aiq = t.test_vector[t.counter, t.amp_idx] / t.test.input_attenuation
        t.iq.setup(frequency = t.fref, bandwidth = bandwidth,
                   acbandwidth = acbandwidth, amplitude = aiq
                   )
        t.set('fref', t.iq.frequency)
        print('iq acbandwidth: {} bandwidth: {}'.format(t.iq.acbandwidth, t.iq.bandwidth))

        if t.f_low < t.iq.acbandwidth or t.f_high < flow_pass:
            if 0 < len(bandwidth):
                flow_pass = bandwidth[0]

            print('WARNING try to increase low pass filter frequency flow_pass: {} acbandwidth: {}'.format(
                flow_pass, t.iq.acbandwidth))

        if applyfilter:
            t.set('acbandwidth_iq', t.iq.acbandwidth)
        else:
            t.set('acbandwidth_iq', 1000)

        # allow time for iq to settle
        sleep(0.01)

        print('test_sequence {} amplitude: {} fs: {}'.format(t.counter,
            t.test_vector[t.counter, t.amp_idx], t.test_vector[t.counter, t.freq_idx]))

        spec_setup(t)

        run_test(t)

        print_results(t)

        t.getcreate('plot_samples', [int(t.test_vector.shape[0]/2), ])
        plotspec(t)

def asg_setup(t):
    n = 18000
    n = round(1.25e8 / 8264)
    t.getcreate('f_low', 1.25e8 / n)
    m = 72000
    t.set_test_ifcal('fref', 1.25e8 / m)
    t.getcreate('fref', t.test.fref)
    t.getcreate('f_high', 0)
    t.f_low, t.fref, fts, t.f_high = calc_freq(fi_low = t.f_low, fir = t.fref)

    # fetch configuration
    t.set_test_ifcal('fs', fts)
    t.getcreate('fs', t.test.fs)

    probe_attenuation = 1
    t.test.set('input_attenuation', 20 * probe_attenuation)

    out1 = 'out1'
    t.getcreate('asg', t.get('asg0'))

    t.set_test_ifcal('asg_raw_a', 0.1)
    t.getcreate('asg_raw_a', t.test.asg_raw_a)
    t.set_test_ifcal('asg_raw_o', 0.0)

    asg_o, asg_a = t.c.cal(out1, t.test.asg_raw_o, ag = t.asg_raw_a)
    t.asg.setup(frequency = t.fs, waveform = 'sin',
              offset = asg_o, amplitude = asg_a,
              output_direct = out1,
              trigger_source = 'immediately'
              )

    # at this point the signal frequency is fixed
    t.fs = t.asg.frequency
    t.test.fs = t.asg.frequency
    print('asg offset: {} asg amplitude: {}'.format(t.asg.offset, t.asg.amplitude))
    print('generator frequency: {} waveform: {} offset: {} amplitude: {}'.format(t.fs, t.asg.waveform, t.test.asg_raw_o, t.asg_raw_a))

def iq_setup(t):
    t.f_low, t.fref, fs, t.f_high = calc_freq(fir = t.fref, fis = t.fs)
    flow_pass = t.fs
    applyfilter = True
    if applyfilter:
        acbandwidth = flow_pass / 2
        bandwidth = [ flow_pass, flow_pass ]
    else:
        acbandwidth = 0
        bandwidth = []

    aiq = t.asg_raw_a / t.test.input_attenuation

    t.getcreate('iq', t.get('iq0'))
    t.iq.setup(frequency = t.fref, bandwidth = bandwidth, gain = 1.0,
             phase = 0, acbandwidth = acbandwidth, amplitude = aiq,
             input = 'in1', output_direct = 'off',
             output_signal = 'output_direct',
             #output_signal = 'quadrature',
             quadrature_factor = 1
             )

    # at this point the reference frequency is fixed
    t.fref = t.iq.frequency
    t.test.fref = t.iq.frequency

    print('iq quadrature frequency: {} amplitude: {}'.format(t.fref, t.iq.amplitude))
    print('iq acbandwidth: {} bandwidth: {}'.format(t.iq.acbandwidth, t.iq.bandwidth))
    print('quadrature_factor: {} iq gain: {} '.format(t.iq.quadrature_factor, t.iq.gain))
    t.f_low, fref, fs, t.f_high = calc_freq(fir = t.fref, fis = t.fs)

    if t.f_low < t.iq.acbandwidth or t.f_high < flow_pass:
        if 0 < len(bandwidth):
            flow_pass = bandwidth[0]

        print('WARNING try to increase low pass filter frequency flow_pass: {} acbandwidth: {}'.format(
            flow_pass, t.iq.acbandwidth))

    if applyfilter:
        t.set('acbandwidth_iq', t.iq.acbandwidth)
    else:
        t.set('acbandwidth_iq', 1000)

    # allow time for iq to settle
    sleep(0.01)

def spec_setup(t):
    sa_input = t.iq.name

    print('spec_setup fs: {} fref: {}'.format(t.fs, t.fref))
    t.getcreate('sa', t.get('spectrumanalyzer'))
    t.sa.setup(
        input1_baseband = sa_input,
        span = 5 * (t.fs + t.fref + t.fs / 10),
        window = 'blackman',
        display_unit = 'Vpk',
        acbandwidth = t.fs / 10
    )

    # set baseband after centre
    t.sa.baseband = True

    print('Sample time: {} Full buffer in: {} decimation: {}'.format(
        t.sa.scope.sampling_time, t.sa.scope.duration, t.sa.scope.decimation))
    print('spectrum analyser centre: {} window: {} span: {} baseband: {} decimation: {} acbandwidth: {}'.format(
        t.sa.center, t.sa.window, t.sa.span, t.sa.baseband, t.sa.decimation, t.sa.acbandwidth))

    print('before sample: spec curve ready: {}'.format(t.sa.curve_ready()))

def run_test(t):
    # the return format is (spectrum for channel 1, spectrum for channel 2,
    # real part of cross spectrum, imaginary part of cross spectrum)
    t.set('res', t.sa.single())
    t.set('label', t.sa.input1_baseband)
    t.set('ylabel', '')
    if 'iq' in t.sa.input1_baseband:
        t.ylabel = '{} real (uncalibrated)'.format(t.sa.display_unit)
        if 'quadrature' in t.iq.output_signal:
            t.label = '{} x {:0.0f}'.format(t.iq.output_signal, t.iq.quadrature_factor)
        elif 'output_direct' == t.iq.output_signal:
            t.label = 'iq.out'
        else:
            t.label = '{}'.format(t.iq.output_signal)

    print('after sample: spec curve ready: {} data length: {}'.format(t.sa.curve_ready(), t.sa.data_length))

def print_results(t):
    # calibrate the results
    to_units = lambda x:t.sa.data_to_display_unit(x, t.sa.rbw)
    t.set('spec_ch1', to_units(t.res[0, :])) #t.c.ac(sa.scope.input1, res[0, :])

    if 'iq' in t.sa.input1_baseband:
        if 'quadrature' == t.iq.output_signal:
            threepeak = False
            freflim = 1.25e8 / ((1.25e8 / t.fref) - 10000)
            if threepeak:
                amaxidx_1_lo, amaxidx_1, amaxidx_1_hi = characterise_spectrum(t.spec_ch1, t.sa.frequencies,
                             np.logical_and( t.acbandwidth_iq < t.sa.frequencies, t.sa.frequencies < freflim))
            amaxidx_2_lo, amaxidx_2, amaxidx_2_hi = characterise_spectrum(t.spec_ch1, t.sa.frequencies,
                             np.logical_and( freflim < t.sa.frequencies, t.sa.frequencies < (t.f_low + t.f_high)/2.0 ))
            amaxidx_3_lo, amaxidx_3, amaxidx_3_hi = characterise_spectrum(t.spec_ch1, t.sa.frequencies,
                             (t.f_low + t.f_high)/2.0 < t.sa.frequencies)
            print('{} DC: {:0.4e} @ {:0.0f}'.format(t.label, t.spec_ch1[0], t.sa.frequencies[0]))
            if threepeak:
                print('{} peak 1: {:0.4e} @ {:0.0f}'.format(t.label, t.spec_ch1[amaxidx_1], t.sa.frequencies[amaxidx_1]))
            print('{} peak 2: {:0.4e} @ {:0.0f}'.format(t.label, t.spec_ch1[amaxidx_2], t.sa.frequencies[amaxidx_2]))
            print('{} peak 3: {:0.4e} @ {:0.0f}'.format(t.label, t.spec_ch1[amaxidx_3], t.sa.frequencies[amaxidx_3]))
            print('{} acbandwidth: {:0.0f} peak 2 f range: {:0.0f} -> {:0.0f} -> {:0.0f}'.format(
                t.label, t.acbandwidth_iq, t.sa.frequencies[amaxidx_2_lo], t.sa.frequencies[amaxidx_2], t.sa.frequencies[amaxidx_2_hi]))
            print('{} peak 3 f range: {:0.0f} -> {:0.0f} -> {:0.0f}'.format(
                t.label, t.sa.frequencies[amaxidx_3_lo], t.sa.frequencies[amaxidx_3], t.sa.frequencies[amaxidx_3_hi]))
            print('INFO peak 2 frequency difference peak 2: {:0.3f} peak 3: {:0.3f}'.format(
                t.f_low - t.sa.frequencies[amaxidx_2], t.f_high - t.sa.frequencies[amaxidx_3]))

        if 'output_direct' == t.iq.output_signal:
            freflim = 1.25e8 / ((1.25e8 / t.fref) + 10000)
            amaxidx_1_lo, amaxidx_1, amaxidx_1_hi = characterise_spectrum(t.spec_ch1, t.sa.frequencies,
                             np.logical_and( freflim < t.sa.frequencies, t.sa.frequencies < t.fref ))
            amaxidx_2_lo, amaxidx_2, amaxidx_2_hi = characterise_spectrum(t.spec_ch1, t.sa.frequencies,
                             (t.fref + t.f_high)/2.0 < t.sa.frequencies)

            print('{} DC: {:0.4e} @ {:0.0f}'.format(t.label, t.spec_ch1[0], t.sa.frequencies[0]))
            print('{} peak 1: {:0.4e} @ {:0.0f}'.format(t.label, t.spec_ch1[amaxidx_1], t.sa.frequencies[amaxidx_1]))
            print('{} peak 2: {:0.4e} @ {:0.0f}'.format(t.label, t.spec_ch1[amaxidx_2], t.sa.frequencies[amaxidx_2]))
            print('{} acbandwidth: {:0.0f} peak 1 f range: {:0.0f} -> {:0.0f} -> {:0.0f}'.format(
                t.label, t.acbandwidth_iq, t.sa.frequencies[amaxidx_1_lo], t.sa.frequencies[amaxidx_1], t.sa.frequencies[amaxidx_1_hi]))
            print('{} peak 2 f range: {:0.0f} -> {:0.0f} -> {:0.0f}'.format(
                t.label, t.sa.frequencies[amaxidx_2_lo], t.sa.frequencies[amaxidx_2], t.sa.frequencies[amaxidx_2_hi]))
            print('INFO peak 2 frequency difference: {:0.3f}'.format(t.fs - t.sa.frequencies[amaxidx_2]))

            if hasattr(t, 'counter'):
                # store results
                aref = 0; bref = 1; cref = 2;
                results = t.getcreate('results', np.zeros([0, 3], dtype = float))
                cal = t.getcreate('results_cal', np.zeros([0, 3], dtype = float))
                print('#{} fs: {:0.0f} peak 2 {:0.4e} @ {:0.0f}'.format(
                    t.counter, t.fs, t.spec_ch1[amaxidx_2], t.sa.frequencies[amaxidx_2]))
                if t.test.testype == 'freq':
                    result = np.array([[t.fs,
                                        t.spec_ch1[amaxidx_2],
                                        t.fs - t.sa.frequencies[amaxidx_2]],], dtype = float)
                    if 0 < cal.shape[0]:
                        idxs = np.where(cal[:, aref] == t.fs)
                        if 0 < len(idxs) and 0 < len(idxs[0]):
                            print('#{} idx: {} @ fs: {:0.0f} calibration amplitude: {}'.format(
                                t.counter, idxs[0][0], cal[idxs[0][0], aref], cal[idxs[0][0], bref]))
                            result = np.array([[t.fs,
                                                t.spec_ch1[amaxidx_2] / cal[idxs[0][0], bref],
                                                abs(cal[idxs[0][0], cref]) + abs(t.fs - t.sa.frequencies[amaxidx_2])
                                                ],], dtype = float)
                else:
                    result = np.array([[t.asg_raw_a,
                                        t.spec_ch1[amaxidx_2],
                                        t.fs - t.sa.frequencies[amaxidx_2]],], dtype = float)
                    if 0 < cal.shape[0]:
                        idxs = np.where(cal[:, aref] == t.asg_raw_a)
                        if 0 < len(idxs) and 0 < len(idxs[0]):
                            print('#{} idx: {} @ amplitude: {} calibration amplitude: {}'.format(
                                t.counter, idxs[0][0], cal[idxs[0][0], aref], cal[idxs[0][0], bref]))
                            result = np.array([[t.asg_raw_a,
                                                t.spec_ch1[amaxidx_2] / cal[idxs[0][0], bref],
                                                abs(cal[idxs[0][0], cref]) + abs(t.fs - t.sa.frequencies[amaxidx_2])
                                                ],], dtype = float)

                t.results = np.concatenate((results, result))
    else:
        amaxidx_lo, amaxidx, amaxidx_hi = characterise_spectrum(t.spec_ch1, t.sa.frequencies,
                         np.logical_and( t.acbandwidth_iq < t.sa.frequencies, t.sa.frequencies < (t.fs + t.f_high) / 2.0 + t.fref ))
        print('{} DC: {:0.4e} @ {:0.0f} start: {:0.4e} @ {:0.0f} end: {:0.4e} @ {:0.0f}'.format(
            t.label, t.spec_ch1[0], t.sa.frequencies[0], t.spec_ch1[10], t.sa.frequencies[10], t.spec_ch1[-1], t.sa.frequencies[-1]))
        print('{} peak: {:0.4e} @ {:0.0f}'.format(
            t.label, t.spec_ch1[amaxidx], t.sa.frequencies[amaxidx]))
        print('{} peak f range: {:0.0f} -> {:0.0f} -> {:0.0f}'.format(
            t.label, t.sa.frequencies[amaxidx_lo], t.sa.frequencies[amaxidx], t.sa.frequencies[amaxidx_hi]))

        print('INFO peak frequency difference: {:0.3f}'.format(t.fs - t.sa.frequencies[amaxidx]))

def plotspec(t):
    if hasattr(t, 'counter') and hasattr(t, 'plot_samples'):
        if not t.counter in t.plot_samples:
            return

    # plot
    prepare_to_show_plot(t.p)

    svg = False
    if svg:
        import matplotlib

        # enable svg plotting
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    # plot the data
    fig = plt.figure()
    plt.semilogx(t.sa.frequencies, t.spec_ch1, label = t.label)

    plt.xlabel('frequency [Hz]');
    plt.ylabel(t.ylabel);

    plt.legend(loc = 'center left', bbox_to_anchor = (1.04, 0.5))
    plt.title('{} amp: {:0.3f} fs: {:0.0f} fref: {:0.0f}'.format(t.name, t.asg_raw_a, t.fs, t.fref))
    plt.tight_layout()

    filename = t.test.filename
    if t.test.testype == 'amp':
        filename = t.test.create_filename('{:0.3f}'.format(t.asg_raw_a))
    elif t.test.testype == 'freq':
        filename = t.test.create_filename('{:0.0f}'.format(t.fs))

    if svg:
        plt.savefig('{}_{}.svg'.format(t.test.dataname, filename))
    else:
        plt.show()

if __name__ == '__main__':
    RPCONFIG = 'mohm.mohm'
    test = get_run_conf(testname = 'mohm test', rpconf = RPCONFIG, dataname = 'data')

    c = CalData()
    c.load()
    c.save(test.dataname)

    t = TestBench(test)
    t.set('c', c)
    t.run(['asg_setup'])
    if t.test.testype == 'amp':
        create_signal_amplitudes(t)
    else:
        create_signal_freqs(t)

    t.run(['iq_setup', 'spec_setup', 'test_sequence'])

    gen = t.test.dump()
    for (k, v) in gen:
        print(k, v)

    if hasattr(t, 'test_vector'):
        print('test_vector\n', t.test_vector)
    if hasattr(t, 'results_cal'):
        print('test calibration\n', t.results_cal)
    if hasattr(t, 'results'):
        print('test results\n', t.results)

    test.save(test.create_filename(test.tag))
    cal = '_cal' if t.test.calibrate else ''
    filename = '{}_{}{}.npz'.format(t.test.dataname, t.test.tag, cal)
    if hasattr(t, 'test_vector') and hasattr(t, 'results'):
        np.savez_compressed(filename, test_vector = t.test_vector, results = t.results)
    elif hasattr(t, 'test_vector'):
        np.savez_compressed(filename, test_vector = t.test_vector)
