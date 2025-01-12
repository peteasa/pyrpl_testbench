#!/usr/bin/env python3

import argparse
from collections import defaultdict   # plotting poles and zeros
import datetime
import json
import logging
import math
from matplotlib import patches
import numpy as np
import pyrpl
from pyrpl.async_utils import sleep
import scipy.signal as sig
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
        #print('calibrate io: {} ao: {} ag: {}'.format(stritem, ao, ag))
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
        #print('calibrate reverse io: {} so: {} sg: {}'.format(io, so, sg))
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
        #print('calibrate curve io: {} data: {}'.format(stritem, ao))
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
        #print('calibrate raw: {} io: {}'.format(data, io))
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

                #print(io)
                fh.write( json.dumps( io ) )

    def _get_val(self, callist, default, val, cuo):
        rtn = default
        stritem = val
        found = False
        if type(stritem) is str and hasattr(self, stritem):
            rtn = self.data[callist][getattr(self, stritem) + cuo]
            found = True
            #print('io port: {} rtn: {}'.format(stritem, rtn))
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
                #print('output_direct: {} rtn: {}'.format(val.output_direct, rtn))

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

    class networkanalyzer(object):
        def __init__(self):
            attrs = ['iq']
            for k in attrs:
                setattr(self, k, 0)

        @property
        def frequencies(self):
            return np.ones((6), dtype = float)

        def setup(self, *arg, **kwargs):
            print('networkanalyzer setup: {} kwargs: {}'.format(arg, kwargs))
            for k in kwargs:
                setattr(self, k, kwargs[k])

        def single(self):
            dummy = np.zeros((4, self.frequencies.shape[0]), dtype = float)
            dummy[0, 5] = 1.0
            return dummy

    class iirc(object):
        class iirtheory(object):
            def __init__(self):
                attrs = ['loops', 'dt']
                for k in attrs:
                    setattr(self, k, 1)

            @property
            def proper_sys(self):
                return ([0,], [1,], 2)
            @property
            def rescaled_sys(self):
                return ([3,], [4,], 5)
            @property
            def rp_discrete(self):
                return ([6,], [7,], 8)

        def __init__(self):
            attrs = ['overflow', 'coefficients']
            for k in attrs:
                setattr(self, k, 0)

            setattr(self, 'iirfilter', self.iirtheory())

        def setup(self, *arg, **kwargs):
            print('iir setup: {} kwargs: {}'.format(arg, kwargs))
            for k in kwargs:
                setattr(self, k, kwargs[k])

        def transfer_function(self, frequencies):
            rtn = np.empty(frequencies.shape, dtype=np.complex128)
            theta = 2 * np.pi / 3
            i = 0.1
            for f in frequencies:
                rtn = np.append(rtn, i * complex(np.cos(theta), np.sin(theta)))
                theta -= np.pi / (2 * frequencies.shape[0])
                i += 0.3

            return rtn

    def __init__(self):
        setattr(self, 'spectrumanalyzer', self.spectrumanalyzer())
        setattr(self, 'networkanalyzer', self.networkanalyzer())
        setattr(self, 'iir', self.iirc())

class TestBench(Meta):
    def __init__(self, test):
        HOSTNAME = 'rp-f0bd75'

        # for debug replace with DummyP / DummyR
        #self._p = DummyP()
        #self._r = DummyR()
        self._p = pyrpl.Pyrpl(config = test.rpconf, hostname = HOSTNAME, gui = False)
        self._r = self.p.rp
        self._name = test.name
        self._test = test

        start_logging(self)

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

class MeasureTransferFunction(Meta):
    def __init__(self, t, tf):
        if hasattr(t, 'logger'):
            self.set('logger', t.logger)

        self.set('tf', tf)
        self.set('frequencies', t.frequencies)

        ttype = gen_test_type(t.test)
        self.set('highpass', True if ttype == 'highpass' else False)
        self.set('lowpass',  True if ttype == 'lowpass' else False)
        self.set('lowpass',  True if ttype == 'butter_lp' else False)
        self.set('bandpass', True if ttype == 'bandpass' else False)
        self.set('notch', True if ttype == 'notch' else False)
        self.set('resonance', True if ttype == 'resonance' else False)

        self.set('phases', np.angle(self.tf, deg = True))
        self.set('tf_abs', np.abs(self.tf))
        self.set('tf_max', self.tf_abs.max())
        self.set('amaxidx', np.argmax(self.tf_abs))
        self.set('dbs', 20 * np.log10(self.tf_abs))
        self.set('dbs_max', self.dbs.max())

        self.set('gain', t.iirf_gain if hasattr(t, 'iirf_gain') else 1.0)
        self.set('gain_adjust', t.gain_adjust if hasattr(t, 'gain_adjust') else 1.0)

        self.set('gain_adjust_correction', gain_adjust_correction(t, self.gain_adjust, self.tf_max))

        super().__init__()

    def peak_gain(self):
        msg = 'low pass calc from phase then from amplitude: gain: {:0.6e} tf_max: {:0.6f} updated gain_adjust: {:0.6e}'.format(
            self.gain,
            self.tf_max,
            self.gain_adjust_correction)
        log_msg(self, msg)

        self.gain_from_phase()
        self.gain_from_amplitude()

    def gain_from_phase(self):
        if self.lowpass:
            if self.highpass:
                msg = 'peak at: {:0.0f}'.format(self.frequencies[self.amaxidx])
                log_msg(self, msg)

            self.lp_gain_from_phase()

        if self.highpass:
            self.hp_gain_from_phase()

    def gain_from_amplitude(self):
        if self.lowpass:
            self.lp_gain_from_amplitude()

        if self.highpass:
            self.hp_gain_from_amplitude()

    def lp_gain_from_phase(self):
        phases_lp = self.phases[self.amaxidx:] if self.highpass else self.phases
        frequencies = self.frequencies[self.amaxidx:] if self.highpass else self.frequencies
        amaxidx = self.amaxidx if self.highpass else 0
        try:
            fminpidx, fmaxpidx = characterise_transfer_function(phases_lp, frequencies, phases_lp < phases_lp[0] - 45)
            self.set('lp_fminpidx', amaxidx + fminpidx)
            self.set('lp_fmaxpidx', amaxidx + fmaxpidx)
            msg = 'p:f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
                self.frequencies[self.lp_fminpidx], self.frequencies[self.lp_fmaxpidx],
                self.dbs[self.lp_fminpidx], self.dbs[self.lp_fmaxpidx],
                self.phases[self.lp_fminpidx], self.phases[self.lp_fmaxpidx])
            log_msg(self, msg)
        except (ValueError, IndexError):
            pass

    def lp_gain_from_amplitude(self):
        dbs_lp = self.dbs[self.amaxidx:] if self.highpass else self.dbs
        frequencies = self.frequencies[self.amaxidx:] if self.highpass else self.frequencies
        amaxidx = self.amaxidx if self.highpass else 0
        try:
            fmindidx, fmaxdidx = characterise_transfer_function(dbs_lp, self.frequencies, dbs_lp < self.dbs_max - 3)
            self.set('lp_fmindidx', amaxidx + fmindidx)
            self.set('lp_fmaxdidx', amaxidx + fmaxdidx)
            msg = 'a:f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
                self.frequencies[self.lp_fmindidx], self.frequencies[self.lp_fmaxdidx],
                self.dbs[self.lp_fmindidx], self.dbs[self.lp_fmaxdidx],
                self.phases[self.lp_fmindidx], self.phases[self.lp_fmaxdidx])
            log_msg(self, msg)
        except ValueError:
            pass

    def hp_gain_from_phase(self):
        try:
            fminpidx, fmaxpidx = characterise_transfer_function(self.phases, self.frequencies, self.phases < self.phases[0] - 45)
            self.set('hp_fminpidx', fminpidx)
            self.set('hp_fmaxpidx', fmaxpidx)
            msg = 'p:f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
                self.frequencies[self.hp_fminpidx], self.frequencies[self.hp_fmaxpidx],
                self.dbs[self.hp_fminpidx], self.dbs[self.hp_fmaxpidx],
                self.phases[self.hp_fminpidx], self.phases[self.hp_fmaxpidx])
            log_msg(self, msg)
        except ValueError:
            pass

    def hp_gain_from_amplitude(self):
        try:
            fmindidx, fmaxdidx = characterise_transfer_function(self.dbs, self.frequencies, self.dbs_max - 3 < self.dbs)
            self.set('hp_fmindidx', fmindidx)
            self.set('hp_fmaxdidx', fmaxdidx)
            msg = 'a:f between: {:0.0f} and {:0.0f} dbs: {:0.3f} and {:0.3f} phase: {:0.3f} and {:0.3f}'.format(
                self.frequencies[self.hp_fmindidx], self.frequencies[self.hp_fmaxdidx],
                self.dbs[self.hp_fmindidx], self.dbs[self.hp_fmaxdidx],
                self.phases[self.hp_fmindidx], self.phases[self.hp_fmaxdidx])
            log_msg(self, msg)
        except ValueError:
            pass

def start_logging(t):
    from pathlib import Path

    formatter = logging.Formatter('%(levelname)s : %(name)s : %(message)s')

    # setup optional console logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # get the root logger
    t.set('logger', logging.getLogger(''))

    t.logger.handlers.clear()
    #logger.addHandler(console)

    filename = '{}.log'.format(t.test.filename)

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)

    # send debug logs to filehandler
    fh.setLevel(logging.DEBUG)
    t.logger.setLevel(logging.INFO)
    t.logger.addHandler(fh)

    t.logger.info('Running: {}'.format(Path(__file__).name))
    msg = 'pyrpl version: {}'.format(pyrpl.__version__)
    t.logger.info(msg)

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

def prepare_to_show_plot():
    import matplotlib

    for i in range(4):
        try:
            matplotlib.use('TkAgg')
            break
        except:
            print('{} qt is still running!'.format(10-i))
            pass

# ref https://gist.github.com/endolith/4625838
def plot_unit_circle(t, maxv = 1.0):
    fig = t.plt.figure(t.plotspec['fig'])
    ax = fig.add_subplot( t.plotspec['gs'] )

    unit_circle = patches.Circle((0,0), radius=1.0, fill=False,
                                 color='black', ls='solid', alpha=0.1)
    ax.add_patch(unit_circle)
    if 1.0 < maxv:
        ax.add_patch(patches.Circle((0,0), radius=maxv, fill=False,
                                 color='grey', ls='dotted', alpha=0.1))
    t.plt.axvline(0, color='0.7')
    t.plt.axhline(0, color='0.7')

    t.plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
    t.plt.title('{}'.format(t.plotspec['title']))

    # t.plt.show()
    # t.plt.pause(3.0)

def plot_points(t, points, symbol):
    # print(points, symbol)
    t1 = t.plt.plot(points.real, points.imag, symbol, markersize=10.0, alpha=0.5)
    mark_overlapping(points)

def plot_clip_points(t, points, limit, symbols):
    val = np.abs(points)
    clip = np.nonzero(limit < val)
    noclip = np.nonzero(val <= limit)
    if 0 < len(noclip): plot_points(t, points[noclip], symbols[0])
    if 0 < len(clip):
        angles = np.angle(points[clip])
        clipped = np.empty(angles.shape, dtype=np.complex128)
        clipped.real = np.cos(angles)
        clipped.imag = np.sin(angles)
        plot_points(t, clipped * limit, symbols[1])

def plot_zepo(t, zeros, poles):
    maxv = np.abs(zeros).max() if 0 < len(zeros) else 0
    absv = np.abs(poles).max() if 0 < len(zeros) else 0
    maxv = absv if maxv < absv else maxv
    limit = 1.5
    maxv = limit if limit < maxv else maxv
    maxv = 1.0 if maxv < 1.0 else maxv
    plot_unit_circle(t, maxv)
    if maxv < limit:
        plot_points(t, zeros, 'o')
        plot_points(t, poles, 'x')
    else:
        # print(zeros, poles)
        plot_clip_points(t, zeros, limit, ['o', 's'])
        plot_clip_points(t, poles, limit, ['x', '*'])

def mark_overlapping(items):
    """
    Given `items` as a list of complex coordinates, make a tally of identical
    values, and, if there is more than one, plot a superscript on the graph.
    """
    d = defaultdict(int)
    for i in items:
        d[i] += 1
    for item, count in d.items():
        if count > 1:
            t.plt.text(item.real, item.imag, r' ${}^{' + str(count) + '}$', fontsize=13)

    # t.plt.pause(2.0)
    # t.plt.show()

def plotting_init(t):
    if not hasattr(t, 'plt'):
        if hasattr(t, 'svg') and t.svg:
            import matplotlib
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt
        t.set('plt', plt)

def create_plot(t, figsize, rows, cols):
    t.plt.close()
    fig = t.plt.figure(figsize=figsize)
    t.set('plot_rows', rows)
    t.set('plot_cols', cols)
    t.set('gs', fig.add_gridspec(t.plot_rows,t.plot_cols, wspace=0.4, hspace=0.45, left = 0.15, right = 0.8))
    t.set('plotspec', {'row': 0, 'col': 0, 'fig': t.plt.gcf().number})
    t.plt.ion()

def plotting_config(t):
    nrws = 0
    ncls = 0
    for item in t.plot_items:
        for key in t.plot_sizes.keys():
            keystr = '{}_'.format(key)
            if keystr == item[0:len(keystr)]:
                ncls += t.plot_sizes[key]['ncols'] if 'ncols' in t.plot_sizes[key] else 0
                nrws += t.plot_sizes[key]['nrows'] if 'nrows' in t.plot_sizes[key] else 0

    ncols = ncls if ncls < 3 else 3
    nrows = nrws + int(np.ceil(ncls / ncols))
    figsize = (nrows * 2, ncols * 3 + 1) ; rows = nrows; cols = ncols
    create_plot(t, figsize, rows, cols)

def plot_zp(t, zeros, poles, title):
    t.plotspec['gs'] = t.gs[t.plotspec['row'], t.plotspec['col']]
    t.plotspec['title'] = title
    plot_zepo(t, zeros, poles)
    if hasattr(t, 'svg'):
        if not t.svg:
            t.plt.pause(3.0)
    else:
        t.plt.pause(3.0)

    t.plotspec['col'] += 1
    if t.plot_cols < t.plotspec['col']:
        t.plotspec['col'] = 0
        t.plotspec['row'] += 1

    if t.plotspec['row'] > t.plot_rows: t.plotspec['row'] = 0

def plt_freq_response(t, values, title, ylabels):
    t.plotspec['title'] = title
    t.plotspec['log'] = True

    if 0 < t.plotspec['col']:
        t.plotspec['col'] = 0
        t.plotspec['row'] += 1

    t.plotspec['gs'] = t.gs[t.plotspec['row'], :]

    fig = t.plt.figure(t.plotspec['fig'])
    ax = fig.add_subplot( t.plotspec['gs'] )
    if 'log' in t.plotspec and t.plotspec['log']:
        ax.set_xscale('log')

    datasets = values.shape[1] if 1 < len(values.shape) else 1
    for idx in range(datasets):
        to_plot = values[:, idx] if 1 < datasets else values
        t.plt.plot(t.frequencies, to_plot, label = ylabels[idx])

    t.plt.ylabel(ylabels[-1]);

    t.plt.legend(loc = 'center left', bbox_to_anchor = (1.04, 0.5))
    t.plt.xlabel('Hz');
    t.plt.title(t.plotspec['title'])

    if hasattr(t, 'svg'):
        if not t.svg:
            t.plt.pause(3.0)
    else:
        t.plt.pause(3.0)

    t.plotspec['row'] += 1
    if t.plotspec['row'] > t.plot_rows: t.plotspec['row'] = 0

def log_msg(t, msg, title = '', loglevel = logging.INFO):
    logfn = print
    if hasattr(t, 'logger'):
        logfn = t.logger.info
        if loglevel == logging.WARNING:
            logfn = t.logger.warning
        elif loglevel == logging.ERROR:
            logfn = t.logger.error
        elif loglevel == logging.CRITICAL:
            logfn = t.logger.critical
        elif loglevel == logging.DEBUG:
            logfn = t.logger.debug

    if 0 < len(title):
        logfn(title)
    logfn(msg)

def log_gain_adjust(t, gain, gain_adjust, title = ''):
    logfn = print
    if hasattr(t, 'logger'):
        logfn = t.logger.info

    logfn(title)
    msg = 'gain: {} gain_adjust: {:0.6e}'.format(gain, gain_adjust)
    logfn(msg)

def log_zp(t, zeros, poles, title = ''):
    logfn = print
    if hasattr(t, 'logger'):
        logfn = t.logger.info

    logfn(title)
    msg = 'zeros: {}'.format([ze for ze in zeros])
    logfn(msg)
    msg = 'poles: {}'.format([po for po in poles])
    logfn(msg)

def log_basics(t):
    samples = round(125e6 / (t.loops * t.fc))

    log_msg(t, 'fc: {:0.0f} loops: {} samples: {}'.format(t.fc, t.loops, samples))

    # get the calculated s-plane poles zeros and gain
    log_gain_adjust(t, t.gain, t.gain_adjust, 'gain and gain adjust')
    log_zp(t, t.zeros, t.poles, 's-plane zeros and poles')

def log_extended(t):
    if not hasattr(t, 'proper_gain'):
        s_plane_proper_init(t)

    log_msg(t, 'adjusted gain: {}'.format(t.proper_gain))
    log_zp(t, t.proper_zeros, t.proper_poles, 'proper zeros and poles')

    log_msg(t, 'rescaled_sys k: {}'.format(t.k))
    log_zp(t, t.zd, t.pd, 'z-plane zeros and poles')

    if not hasattr(t, 'rd'):
        partial_init(t)

    log_msg(t, 'rd: {} pd: {} cd: {}'.format(t.rd, t.pd, t.cd), 'partial fraction expansion')
    log_msg(t, '{}'.format(t.iirf.coefficients), 'Coefficients (6 per biquad)')
    #if hasattr(t, 'coef_trad_raw'):
    #    log_msg(t, '{}'.format(t.coef_trad_raw), 'Raw Traditional Coefficients (6 per biquad)')
    if hasattr(t, 'coef_trad'):
        log_msg(t, '{}'.format(t.coef_trad), 'Rounded Traditional Coefficients (6 per biquad)')
    if hasattr(t, 'fpga_coef_trad'):
        log_msg(t, '{}'.format(t.fpga_coef_trad), 'Traditional FPGA Coefficients (6 per biquad)')
    #if hasattr(t, 'coef_comp_raw'):
    #    log_msg(t, '{}'.format(t.coef_comp_raw), 'Raw Compact Coefficients (6 per biquad)')
    if hasattr(t, 'coef_comp'):
        log_msg(t, '{}'.format(t.coef_comp), 'Rounded Compact Coefficients (6 per biquad)')
    if hasattr(t, 'fpga_coef_comp'):
        log_msg(t, '{}'.format(t.fpga_coef_comp), 'Compact FPGA Coefficients (6 per biquad)')

def omegac_sin(omegac, angle, delta = 0):
    val = np.sin((angle+delta) * np.pi / 180)
    if abs(omegac * val) < 1.0:
        val = 0

    return val

def omegac_cosz(omegac, angle, delta = 0):
    val = np.cos((angle+delta) * np.pi / 180)
    if abs(omegac * val) < 1.0:
        val = - 0.0001 / omegac

    return val

def omegac_cos(omegac, angle, delta = 0):
    val = np.cos((angle+delta) * np.pi / 180)
    if abs(omegac * val) < 1.0:
        val = 0

    return val

def generate_f(start, stop, count):
    return np.logspace(
        np.log10(start),
        np.log10(stop),
        count,
        endpoint=True)

def s_plane_proper_clear(t):
    if hasattr(t, 'proper_zeros'):
        t.pop('proper_zeros')
        t.pop('proper_poles')
        t.pop('proper_gain')

def s_plane_proper_init(t):
    if hasattr(t, 'iirf_gain') and not hasattr(t, 'proper_zeros'):
        (zeros, poles, gain) = t.iirf.iirfilter.proper_sys
        t.set('proper_zeros', np.asarray(zeros, dtype=np.complex128) / t.fc)
        t.set('proper_poles', np.asarray(poles, dtype=np.complex128) / t.fc)
        t.set('proper_gain', gain)

def plot_s_plane_proper(t):
    if hasattr(t, 'proper_zeros'):
        plot_zp(t, t.proper_zeros, t.proper_poles,
                'proper ze: {} po: {}'.format(
                    len(t.proper_zeros), len(t.proper_poles)))

def z_plane_pz_clear(t):
    if hasattr(t, 'zd'):
        t.pop('zd')
        t.pop('pd')
        t.pop('k')

def z_plane_pz_init(t):
    if hasattr(t, 'iirf_gain') and not hasattr(t, 'zd'):
        z, p, k = t.iirf.iirfilter.rescaled_sys

        # Expose internals zd to test harness
        zd = np.exp(np.asarray(z, dtype=np.complex128) * t.dt * t.loops)
        pd = np.exp(np.asarray(p, dtype=np.complex128) * t.dt * t.loops)
        # Improve stability: add zeros at s = \infty fix up IirFilter.coefficients
        # applying the following and uncomment the following
        #while len(zd) < len(pd):
        #    zd = np.append(zd, complex(-1, 0))

        t.set('zd', zd)
        if hasattr(t, 'pd') and not np.array_equal(t.pd, pd):
            logmsg(t, 'WARNING z_plane_pz_init pd: {} old pd: {}'.format(pd, t.pd))

        t.set('pd', pd)
        t.set('k', k)

def plot_z_plane_pz(t):
    if hasattr(t, 'zd'):
        plot_zp(t, t.zd, t.pd, 'zplane ze: {} po: {}'.format(len(t.zd), len(t.pd)))

def partial_clear(t):
    if hasattr(t, 'rd'):
        t.pop('rd')
        t.pop('pd')
        t.pop('cd')

def partial_init(t):
    if hasattr(t, 'iirf_gain') and not hasattr(t, 'rd'):
        rd, pd, cd = t.iirf.iirfilter.rp_discrete
        t.set('rd', rd)
        t.set('cd', cd)
        if hasattr(t, 'pd') and not np.array_equal(t.pd, pd):
            logmsg(t, 'WARNING partial_init pd: {} old pd: {}'.format(pd, t.pd))

        t.set('pd', pd)

def tf_partial_generation(t):
    if not hasattr(t, 'rd'):
        partial_init(t)

    t.set('tf_partial', np.empty(t.frequencies.shape, dtype=np.complex128))
    for fidx, fz in enumerate(t.frequencies):
        angle = np.pi * fz * t.dt * t.loops
        zc = complex( np.cos(angle),
                      np.sin(angle) )

        t.tf_partial[fidx] = t.cd
        for idx, rd in enumerate(t.rd):
            t.tf_partial[fidx] += rd / (zc - t.pd[idx])

def tf_partial_clear(t):
    if hasattr(t, 'tf_partial'):
        partial_clear(t)
        t.pop('tf_partial')
        t.pop('plot_partial_dbs')
        t.pop('plot_partial_phases')

def tf_partial_init(t):
    if not hasattr(t, 'tf_partial'):
        # generate the transfer function from the partial fractions
        tf_partial_generation(t)

    tf_abs = np.abs(t.tf_partial)
    t.set('plot_partial_dbs', 20 * np.log10(tf_abs))
    t.set('plot_partial_phases', np.angle(t.tf_partial, deg = True))

def rp2coefficients(t):
    ## copied from iir_theory with minor changes
    coef_raw = 'coef_{}_raw'.format(t.coef_type)
    if hasattr(t, 'rd') and not hasattr(t, coef_raw):
        t.getcreate('tol', default = 1e-3)
        N = int(np.ceil(float(len(t.pd)) / 2.0))
        if t.cd != 0:
            N += 1
        if N == 0:
            t.logmsg(t,
                'Warning: No poles or zeros defined. Filter will be turned off!')
            coeff = t.set(coef_raw, np.zeros((1, 6), dtype=np.float64))
            coeff[0, 0] = 0
            coeff[:, 3] = 1.0
            return t.get(coef_raw)

        coeff = t.set(coef_raw, np.zeros((N, 6), dtype=np.float64))
        coeff[0, 0] = 0
        coeff[:, 3] = 1.0

        rc = list(t.rd)
        pc = list(t.pd)
        complexp = []
        complexr = []
        realp = []
        realr = []
        while (len(pc) > 0):
            pp = pc.pop(0)
            rr = rc.pop(0)
            if np.imag(pp) == 0:
                realp.append(pp)
                realr.append(rr)
            else:
                # find closest-matching index
                diff = np.abs(np.asarray(pc) - np.conjugate(pp))
                index = np.argmin(diff)
                if diff[index] > t.tol:
                    t.logmsg(
                        t,
                        'Conjugate partner for pole {} deviates from expected value by {} > {}'.format(
                            pp, diff[index], t.tol))
                complexp.append((pp + np.conjugate(pc.pop(index))) / 2.0)
                complexr.append((rr + np.conjugate(rc.pop(index))) / 2.0)

        complexp = np.asarray(complexp, dtype=np.complex128)
        complexr = np.asarray(complexr, dtype=np.complex128)
        invert = -1.0
        # THIS IS THE CHANGE: introduce b2 - [b0, b1, b2, 1.0, a1, a2]
        coeff[:len(complexp), t.coef_idx] = 2.0 * np.real(complexr)
        coeff[:len(complexp), t.coef_idx + 1] = -2.0 * np.real(complexr * np.conjugate(complexp))
        coeff[:len(complexp), 4] = 2.0 * np.real(complexp) * invert
        coeff[:len(complexp), 5] = -1.0 * np.abs(complexp) ** 2 * invert
        if len(realp) % 2 != 0:
            realp.append(0)
            realr.append(0)

        realp = np.asarray(np.real(realp), dtype=np.float64)
        realr = np.asarray(np.real(realr), dtype=np.float64)
        for i in range(len(realp) // 2):
            p1, p2 = realp[2 * i], realp[2 * i + 1]
            r1, r2 = realr[2 * i], realr[2 * i + 1]
            coeff[len(complexp)+i, t.coef_idx] = r1 + r2
            coeff[len(complexp)+i, t.coef_idx + 1] = -r1 * p2 - r2 * p1
            coeff[len(complexp)+i, 4] = (p1 + p2) * invert
            coeff[len(complexp)+i, 5] = (-p1 * p2) * invert

        # finish the design by adding a constant term if needed
        if t.cd != 0:
            coeff[-1, 0] = t.cd

        return t.get(coef_raw)

def sos2zpk(sos):
    ## copied from iir_theory
    sos = np.asarray(sos)
    n_sections = sos.shape[0]
    z = np.empty(n_sections*2, np.complex128)
    p = np.empty(n_sections*2, np.complex128)
    k = 1.
    for section in range(n_sections):
        b, a = sos[section, :3], sos[section, 3:]
        # remove leading zeros from numerator to avoid badcoefficient warning in scipy.signal.normaize
        while b[0] == 0:
            b = b[1:]
        # convert to transfer function
        zpk = sig.tf2zpk(b, a)
        z[2*section:2*(section+1)] = zpk[0]
        p[2*section:2*(section+1)] = zpk[1]
        k *= zpk[2]

    return z, p, k

def minimize_delay(t):
    ## copied from iir_theory
    coef_raw = 'coef_{}_raw'.format(t.coef_type)
    if hasattr(t, coef_raw):
        ranks = list()
        for c in list(t.get(coef_raw)):
            # empty sections (numerator is 0) are ranked 0
            if (c[0:3] == 0).all():
                ranks.append(0)
            else:
                z, p, k = sos2zpk([c])
                # compute something proportional to the frequency of the pole
                ppp = [np.abs(np.log(pp)) for pp in p if pp != 0]
                if not ppp:
                    f = 1e20  # no pole -> superfast
                else:
                    f = np.max(ppp)
                ranks.append(f)

        newcoefficients = [c for (rank, c) in
                           sorted(zip(ranks, list(t.get(coef_raw))),
                                  key=lambda pair: -pair[0])]

        t.set(coef_raw, np.array(newcoefficients))

        return t.get(coef_raw)

def finiteprecision(t):
    ## copied from iir_theory finiteprecision
    coef_raw = 'coef_{}_raw'.format(t.coef_type)
    coef = 'coef_{}'.format(t.coef_type)
    if hasattr(t, coef_raw) and not hasattr(t, coef):
        t.getcreate('iirbits', default = 32)
        t.getcreate('iirshift', default = 29)
        coeff = t.set(coef, np.zeros(t.get(coef_raw).shape, dtype=np.float64))
        coeff += t.get(coef_raw)
        for x in np.nditer(coeff, op_flags=['readwrite']):
            xr = np.round(x * 2 ** t.iirshift)
            xmax = 2 ** (t.iirbits - 1)
            if xr == 0 and xr != 0:
                t.logmsg(t,
                         'One value was rounded off to zero: Increase shiftbits in fpga design if this is a problem!',
                         loglevel = logging.WARNING)
            elif xr > xmax - 1:
                xr = xmax - 1
                t.logmsg(t, 'One value saturates positively: Increase totalbits or decrease gain!',
                         loglevel = logging.WARNING)
            elif xr < -xmax:
                xr = -xmax
                t.logmsg(t, 'One value saturates negatively: Increase totalbits or decrease gain!',
                         loglevel = logging.WARNING)

            x[...] = 2 ** (-t.iirshift) * xr

    return t.get(coef)

def generate_fpga_coefficients(t):
    ## adapted from iir coefficients
    coef = 'coef_{}'.format(t.coef_type)
    fpga_coef = 'fpga_coef_{}'.format(t.coef_type)
    invert = -1
    if hasattr(t, coef) and not hasattr(t, fpga_coef):
        t.getcreate('iirbits', default = 32)
        t.getcreate('iirshift', default = 29)
        coeff = t.get(coef)
        f = coeff * 2 ** t.iirshift
        fpga_coeff = t.set(fpga_coef, f.astype(int))
        for i in range(fpga_coeff.shape[0]):
            for j in range(4, fpga_coeff.shape[1], 1):
                fpga_coeff[i, j] *= invert

        return t.get(fpga_coef)

def extend_to_loops(t):
    coef_raw = 'coef_{}_raw'.format(t.coef_type)
    if hasattr(t, coef_raw) and t.get(coef_raw).shape[0] < t.loops:
        for i in range(t.loops - t.get(coef_raw).shape[0]):
            bq = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype = float)
            t.set(coef_raw, np.concatenate((t.get(coef_raw), bq)))

def generate_coefficients(t):
    coef_raw = 'coef_{}_raw'.format(t.coef_type)
    if not hasattr(t, coef_raw):
        t.run(['partial_init', 'rp2coefficients', 'minimize_delay', 'extend_to_loops', 'finiteprecision'])

def fpga_coef_trad_clear(t):
    tf_coef_trad_raw_clear(t)

def fpga_coef_trad_init(t):
    t.set('coef_idx', 1)
    t.set('coef_type', 'trad')
    t.set('coef_format', 'trad')
    generate_coefficients(t)
    generate_fpga_coefficients(t)

def fpga_coef_comp_clear(t):
    tf_coef_comp_raw_clear(t)

def fpga_coef_comp_init(t):
    t.set('coef_idx', 0)
    t.set('coef_type', 'comp')
    t.set('coef_format', 'trad')
    generate_coefficients(t)
    generate_fpga_coefficients(t)

def fpga_coef_clear(t):
    fpga_coef_trad_clear(t)
    fpga_coef_comp_clear(t)

def fpga_coef_init(t):
    fpga_coef_trad_init(t)
    fpga_coef_comp_init(t)

def tf_coef_generation(t):
    generate_coefficients(t)

    coef = 'coef_{}'.format(t.coef_format)
    tf_coef = 'tf_coef_{}'.format(t.coef_format)
    tf_coeff = t.set(tf_coef, np.empty(t.frequencies.shape, dtype=np.complex128))
    invert = -1.0
    for fidx, fz in enumerate(t.frequencies):
        angle = np.pi * fz * t.dt * t.loops
        zc = complex( np.cos(angle),
                      np.sin(angle) )

        tf_coeff[fidx] = 0.0
        for biquad in t.get(coef):
            bq = zc * zc * biquad[0] + zc * biquad[1] + biquad[2]
            bq /= zc * zc * biquad[3] - invert * zc * biquad[4] - invert * biquad[5]
            tf_coeff[fidx] += bq

def tf_coef_trad_raw_generation(t):
    t.set('coef_idx', 1)
    t.set('coef_type', 'trad')
    t.set('coef_format', 'trad_raw')
    tf_coef_generation(t)

def tf_coef_comp_raw_generation(t):
    t.set('coef_idx', 0)
    t.set('coef_type', 'comp')
    t.set('coef_format', 'comp_raw')
    tf_coef_generation(t)

def tf_coef_clear(t):
    if hasattr(t, 'coef_{}_raw'.format(t.coef_type)):
        t.pop('coef_{}_raw'.format(t.coef_type))
        t.pop('coef_{}'.format(t.coef_type))
        t.pop('fpga_coef_{}'.format(t.coef_type))
        t.pop('tf_coef_{}_raw'.format(t.coef_type))
        t.pop('tf_coef_{}'.format(t.coef_type))
        t.pop('plot_coef_{}_raw_dbs'.format(t.coef_type))
        t.pop('plot_coef_{}_raw_phases'.format(t.coef_type))
        t.pop('plot_coef_{}_dbs'.format(t.coef_type))
        t.pop('plot_coef_{}_phases'.format(t.coef_type))

def tf_coef_init(t):
    # generate the transfer function from the z-plane poles and zeros
    tf_coef_generation(t)

    tf_coef = 'tf_coef_{}'.format(t.coef_format)
    tf_abs = np.abs(t.get(tf_coef))
    t.set('plot_coef_{}_dbs'.format(t.coef_format), 20 * np.log10(tf_abs))
    t.set('plot_coef_{}_phases'.format(t.coef_format),
          np.angle(t.get('tf_coef_{}'.format(t.coef_format)), deg = True))

def tf_coef_trad_raw_clear(t):
    t.set('coef_idx', 1)
    t.set('coef_type', 'trad')
    tf_coef_clear(t)

def tf_coef_trad_raw_init(t):
    t.set('coef_idx', 1)
    t.set('coef_type', 'trad')
    t.set('coef_format', 'trad_raw')
    tf_coef_init(t)

def tf_coef_trad_clear(t):
    t.set('coef_idx', 1)
    t.set('coef_type', 'trad')
    tf_coef_clear(t)

def tf_coef_trad_init(t):
    t.set('coef_idx', 1)
    t.set('coef_type', 'trad')
    t.set('coef_format', 'trad')
    tf_coef_init(t)

def tf_coef_comp_raw_clear(t):
    t.set('coef_idx', 0)
    t.set('coef_type', 'comp')
    tf_coef_clear(t)

def tf_coef_comp_raw_init(t):
    t.set('coef_idx', 0)
    t.set('coef_type', 'comp')
    t.set('coef_format', 'comp_raw')
    tf_coef_init(t)

def tf_coef_comp_clear(t):
    t.set('coef_idx', 0)
    t.set('coef_type', 'comp')
    tf_coef_clear(t)

def tf_coef_comp_init(t):
    t.set('coef_idx', 0)
    t.set('coef_type', 'comp')
    t.set('coef_format', 'comp')
    tf_coef_init(t)

def tf_pz_generation(t):
    t.set('tf_pz', np.empty(t.frequencies.shape, dtype=np.complex128))
    for fidx, fz in enumerate(t.frequencies):
        angle = np.pi * fz * t.dt * t.loops
        zc = complex( np.cos(angle),
                      np.sin(angle) )

        t.tf_pz[fidx] = t.k
        for idx, po in enumerate(t.pd):
            if idx < t.zd.shape[0]:
                t.tf_pz[fidx] *= zc - t.zd[idx]

            t.tf_pz[fidx] /= zc - po

def tf_pz_clear(t):
    if hasattr(t, 'tf_pz'):
        t.pop('tf_pz')
        t.pop('plot_pz_dbs')
        t.pop('plot_pz_phases')

def tf_pz_init(t):
    if not hasattr(t, 'tf_pz'):
        # generate the transfer function from the z-plane poles and zeros
        tf_pz_generation(t)

    tf_abs = np.abs(t.tf_pz)
    t.set('plot_pz_dbs', 20 * np.log10(tf_abs))
    t.set('plot_pz_phases', np.angle(t.tf_pz, deg = True))

def tf_desgn_clear(t):
    if hasattr(t, 'designdata'):
        t.pop('designdata')
        t.pop('plot_desgn_dbs')
        t.pop('plot_desgn_phases')

def tf_desgn_init(t):
    t.set('designdata', t.iirf.transfer_function(t.frequencies))
    tf_abs = np.abs(t.designdata)
    t.set('plot_desgn_dbs', 20 * np.log10(tf_abs))
    t.set('plot_desgn_phases', np.angle(t.designdata, deg = True))

def characterise_transfer_function(data, frequencies, condition):
    idxs = np.where(condition)
    fminidx_hi = idxs[0].min()
    fminidx_lo = fminidx_hi - 1 if 0 < fminidx_hi else fminidx_hi

    return fminidx_lo, fminidx_hi

def tf_meas_clear(t):
    if hasattr(t, 'plot_meas_phases'):
        t.pop('plot_meas_phases')
        t.pop('plot_meas_dbs')

def tf_meas_init(t):
    if not hasattr(t, 'tf_measurement'):
        t.set('tf_measurement', MeasureTransferFunction(t, t.tf))

    t.set('plot_meas_phases', t.tf_measurement.phases)
    t.set('plot_meas_dbs', t.tf_measurement.dbs)

def tf_characterisation_clear(t):
    t.pop('tf_measurement')

def tf_characterisation_init(t):
    if not hasattr(t, 'tf_measurement'):
        t.set('tf_measurement', MeasureTransferFunction(t, t.tf))

    t.tf_measurement.peak_gain()

def tf_dbs_clear(t):
    for item in t.tf_items:
        t.run(['tf_{}_clear'.format(item)])

def tf_dbs_init(t):
    for item in t.tf_items:
        if not hasattr(t, 'plot_{}_dbs'.format(item)):
            initfn = 'tf_{}_init'.format(item)
            if DEBUG: print('tf_dbs_init: {}'.format(initfn))
            t.run([initfn])

def plot_tf_dbs(t):
    labels = []
    to_plot = []
    for item in t.tf_items:
        plot_data = 'plot_{}_dbs'.format(item)
        to_plot.append(t.get(plot_data))

        labels.append(item)

    to_plot = np.stack(to_plot, axis = -1)

    labels.append('db')
    plt_freq_response(t, to_plot, 'frequency response', labels)

def tf_phases_clear(t):
    for item in t.tf_items:
        t.run(['tf_{}_clear'.format(item)])

def tf_phases_init(t):
    for item in t.tf_items:
        if not hasattr(t, 'plot_{}_phases'.format(item)):
            initfn = 'tf_{}_init'.format(item)
            if DEBUG: print('tf_phases_init: {}'.format(initfn))
            t.run([initfn])

def plot_tf_phases(t):
    labels = []
    to_plot = []
    for item in t.tf_items:
        plot_data = 'plot_{}_phases'.format(item)
        to_plot.append(getattr(t, plot_data))

        labels.append(item)

    to_plot = np.stack(to_plot, axis = -1)

    labels.append('deg')
    plt_freq_response(t, to_plot, 'phase response', labels)

def plot_results(t):
    plotting_init(t)
    plotting_config(t)

    for item in t.plot_items:
        initfn = '{}_init'.format(item)
        plotfn = 'plot_{}'.format(item)
        if DEBUG: print('plot_results: {}'.format(initfn))
        t.run([initfn, plotfn])

    if hasattr(t, 'svg') and t.svg:
        freq = '{:08.0f}'.format(t.fc)
        t.plt.savefig('{}_{}.svg'.format(t.test.filename, freq))

def is_similar(f, cf):
    d = 1.25e8 / f
    c = 1.25e8 / cf

    return c - 2 < d and d < c + 2

def gain_calibration(t, new_vectors, data):
    similar = True # is_similar(t.test.fc, fc)
    rtnfs = data['results'] if similar else new_vectors
    if similar and 'results' in data:
        t.getcreate('results_cal', data['results'])
        print('calibration data: {}'.format(t.results_cal))

    return rtnfs

def load_calibration(t, seq):
    filename = '{}_{}_cal.npz'.format(t.test.dataname, t.test.tag)
    try:
        with np.load(filename) as data:
            rtn = gain_calibration(t, seq, data)
    except:
        rtn = seq

    return rtn

def configure_gain(t):
    loops_samples = round(125e6 / t.test.fc)
    samples = loops_samples / t.test.loops

    t.getcreate('fc', 125e6 / (t.test.loops * samples))
    t.getcreate('omegac', 2 * np.pi * t.fc)
    t.getcreate('gain', t.test.gain)

    test_idx = 0

    ttype = gen_test_type(t.test)
    if ttype is None: ttype = 'highpass'

    if ttype == 'highpass' or ttype == 'lowpass' or ttype == 'bandpass':
        tests = [{'gain_adjust_hp': 9.99e5, 'gain_adjust_lp': 3.49e7, 'samples': samples}]

    if ttype == 'highpass':
        t.set('gain_adjust', tests[test_idx]['gain_adjust_hp'])

    if ttype == 'lowpass':
        t.set('gain_adjust', tests[test_idx]['gain_adjust_lp'])

    if ttype == 'bandpass':
        gain_adjust_twin = 4.982265e-01
        t.set('gain_adjust', tests[test_idx]['gain_adjust_lp'] *
                             tests[test_idx]['gain_adjust_hp'] *
                             gain_adjust_twin)

    if ttype == 'butter_lp':
        tests = [{'gain_adjust': 2.6e23, 'samples': samples}]
        t.set('gain_adjust', tests[test_idx]['gain_adjust'])

    if ttype == 'notch':
        tests = [{'gain_adjust': 1.0, 'samples': samples}]
        t.set('gain_adjust', tests[test_idx]['gain_adjust'])

    if ttype == 'resonance':
        tests = [{'gain_adjust': 7.0e22, 'samples': samples}]
        t.set('gain_adjust', tests[test_idx]['gain_adjust'])

def get_create_gain_adjust(t):
    t.getcreate('fc_idx', 0)
    t.getcreate('gain_adjust_idx', 1)

    # to iterate over several fc values
    t.test.getcreate('fc_count', 3*5+1)
    t.test.getcreate('fc_start', 1e4)
    t.test.getcreate('fc_stop', 1e7)
    fc = generate_f(t.test.fc_start, t.test.fc_stop, t.test.fc_count)

    ttype = gen_test_type(t.test)
    if ttype is None: ttype = 'highpass'

    gain_adjust = np.ones(fc.shape, dtype = float)
    if ttype == 'lowpass':
        gain_adjust *= 6.6e7

    if ttype == 'highpass':
        gain_adjust *= 9.7e5

    if ttype == 'bandpass':
        gain_adjust *= 3.25e13

    if ttype == 'butter_lp':
        gain_adjust *= 2.6e23

    if ttype == 'resonance':
        gain_adjust *= 7.0e22

    test_vectors = np.ones((fc.shape[0], 2), dtype = float)
    test_vectors[:, t.fc_idx] = fc
    test_vectors[:, t.gain_adjust_idx] = gain_adjust
    if not t.test.calibrate:
        test_vectors = load_calibration(t, test_vectors)

    t.set('gain', t.test.gain)
    t.set('iter_count', 0)
    t.set('test_vectors', test_vectors)

def generate_highpass_poles_zeros(t):
    t.zeros = np.append(t.zeros, complex( -0.000001, 0 ))
    t.poles = np.append(t.poles, complex( -1.0, 0 ))

def generate_lowpass_poles_zeros(t):
    t.poles = np.append(t.poles, complex( -1.0, 0 ))

def generate_bandpass_poles_zeros(t):
    t.zeros = np.append(t.zeros, complex( -0.000001, 0 ))

    lmulti = 1.1 ; hmulti = 0.9
    for pole in [ complex( -1.0 * lmulti, 0 ), complex( -1.0 * hmulti, 0 ) ]:
        t.poles = np.append(t.poles, pole)

def generate_butter_lp_poles_zeros(t):
    t.test.getcreate('butterworth_order', default = 3)
    for k in range(int(np.ceil(t.test.butterworth_order/2))):
        theta = (2 * k + 1) * np.pi / (2 * t.test.butterworth_order)
        t.poles = np.append(t.poles, complex(-np.sin(theta), omegac_cos(t.omegac, theta*180/np.pi)))

def generate_notch_poles_zeros(t):
    angles = [90]
    for idx_z, angle in enumerate(angles):
        angle = angles[idx_z]
        zero = complex( omegac_cosz(t.omegac, angle),
                        omegac_sin(t.omegac, angle) )
        t.zeros = np.append(t.zeros, zero)

    angles = [179.99]
    for idx_p, angle in enumerate(angles):
        angle = angles[idx_p]
        pole = complex( omegac_cosz(t.omegac, angle),
                        omegac_sin(t.omegac, angle) )
        t.poles = np.append(t.poles, pole)

def generate_resonance_poles_zeros(t):
    zero = -1e-8 / t.fc
    t.zeros = np.append(t.zeros, zero)

    angle = 91
    pole = complex( omegac_cosz(t.omegac, angle),
                    omegac_sin(t.omegac, angle) )
    t.poles = np.append(t.poles, pole)

def generate_poles_zeros(t):
    t.set('zeros', np.array([ ], dtype = np.complex128))
    t.set('poles', np.array([ ], dtype = np.complex128))
    ttype = gen_test_type(t.test)
    if ttype is None: ttype = 'highpass'

    t.run(['generate_{}_poles_zeros'.format(ttype)])

def iirf_init(t):
    t.getcreate('iirf', t.getcreate('iir'))

def try_attrib(module, **kargs):
    # work round for iir partial init bug
    # attributes have the call_setup=True applied so if there is a
    # change in poles and zeros that results in an invalid combination
    # for example old gain and new zeros, initial attribute assignment fails
    #
    # alternative work round is to use a different configuration file name or
    # delete the configuration .yml file for each attempt
    #
    # alternative is to disable by setting configuration file ''

    for arg in kargs.keys():
        try:
            if hasattr(module, arg):
                setattr(module, arg, kargs[arg])
        except ValueError:
            pass

def iirf_setup(t):
    t.set('iirf_gain', t.gain / t.gain_adjust)
    zeros=t.fc * t.zeros
    poles=t.fc * t.poles

    # print('iirf_setup poles: {} zeros: {} iirf_gain: {}'.format(t.poles, t.zeros, t.iirf_gain))
    for i in range(1):
        # work round for iir partial init bug
        try:
            t.iirf.setup(gain=t.iirf_gain,
                         poles=poles, zeros=zeros,
                         loops=t.test.loops,
                         output_direct = 'off')
            break
        except ValueError:
            pass

        try_attrib(t.iirf,
                   gain = t.iirf_gain,
                   zeros = zeros,
                   poles = poles)

        log_msg(t, 'last attempt to setup iir')
        t.iirf.setup(gain=t.iirf_gain,
                     poles=poles, zeros=zeros,
                     loops=t.test.loops,
                     output_direct = 'off')

    t.set('loops', t.iirf.iirfilter.loops)
    t.set('dt', t.iirf.iirfilter.dt)

def gain_adjust_correction(t, gain_adjust, tf_max):
    return gain_adjust * tf_max / t.gain

def iirf_gain_adjust_correction(t):
    freq = []
    for i in range(0, 3, -1):
        freq.append(t.fc / (i+1))
    for i in range(3):
        freq.append(t.fc * (i+1))

    designdata = t.iirf.transfer_function([t.fc/4, t.fc/3, t.fc/2, t.fc, t.fc*2, t.fc*3, t.fc*4])
    tf_abs = np.abs(designdata)

    return gain_adjust_correction(t, t.gain_adjust, tf_abs.max())

def network_analyser_init(t):
    # setup the network analyzer to measure tf of iir filter
    t.getcreate('na', t.get('networkanalyzer'))
    t.na.setup(start_freq = t.test.start_freq, stop_freq = t.test.stop_freq, rbw = t.test.rbw,
             points = t.test.points,
             average_per_point = 1,
             amplitude = 0.1,
             input = 'iir', output_direct = 'off',
             logscale=True)

    t.iirf.input = t.na.iq

def run_test(t):
    # check if the filter is ok
    msg = 'IIR overflows before: {}'.format(t.iirf.overflow)
    print(msg); log_msg(t, msg)

    t.set('tf', t.na.single())
    t.set('frequencies', t.na.frequencies)

    # check if the filter is still ok after measuring the transfer function
    msg = 'IIR overflows after: {}'.format(t.iirf.overflow)
    print(msg); log_msg(t, msg)

def gen_test_type(test):
    ttype = 'highpass' if hasattr(test, 'highpass') and test.highpass else None
    ttype = 'lowpass' if hasattr(test, 'lowpass') and test.lowpass else ttype
    ttype = 'bandpass' if hasattr(test, 'highpass') and hasattr(test, 'lowpass') and test.highpass and test.lowpass else ttype
    ttype = 'butter_lp' if hasattr(test, 'butter_lp') and test.butter_lp else ttype
    ttype = 'notch' if hasattr(test, 'notch') and test.notch else ttype
    ttype = 'resonance' if hasattr(test, 'resonance') and test.resonance else ttype

    return ttype

def get_run_conf(**kargs):
    today = '{}'.format(''.join(datetime.date.today().isoformat().split('-')))
    progname = 'filters.filters'
    parser = argparse.ArgumentParser(
        prog = progname,
        description = 'Calculate gain adjustment')
    parser.add_argument('--hp', action = argparse.BooleanOptionalAction, required = False)
    parser.add_argument('--lp', action = argparse.BooleanOptionalAction, required = False)
    parser.add_argument('--blp', action = argparse.BooleanOptionalAction, required = False)
    parser.add_argument('--notch', action = argparse.BooleanOptionalAction, required = False)
    parser.add_argument('--resonance', action = argparse.BooleanOptionalAction, required = False)
    parser.add_argument('--cal', action = 'store_true', required = False)
    parser.add_argument('--tag', type = str, required = False)
    parser.add_argument('--plot', action = argparse.BooleanOptionalAction, required = False)
    args = parser.parse_args()
    # print('test args: {}'.format(args))

    # command line arguments override kargs overrides config file
    testname = kargs['testname'] if 'testname' in kargs else 'test'

    # select different configuration depending on the type of filter
    highpass = False if args.hp is None else args.hp
    lowpass = False if args.lp is None else args.lp
    butter_lp = False if args.blp is None else args.blp
    notch = False if args.notch is None else args.notch
    resonance = False if args.resonance is None else args.resonance

    tag = 'highpass' if highpass else today
    tag = 'lowpass' if lowpass else tag
    tag = 'bandpass' if highpass and lowpass else tag
    tag = 'butter_lp' if butter_lp else tag
    tag = 'notch' if notch else tag
    tag = 'resonance' if resonance else tag

    # override the tag if it is provided on the command line
    tag = kargs['tag'] if 'tag' in kargs else tag
    tag = tag if args.tag is None else args.tag

    test = MetaTest(testname, tag)
    test.load()
    # print('test config after load: {}'.format([i for i in test.dump()]))

    dataname = test.getcreate('dataname', default = 'data')
    test.set('dataname', kargs['dataname'] if 'dataname' in kargs else dataname)
    rpconf = test.getcreate('rpconf', default = progname)
    test.set('rpconf', kargs['rpconf'] if 'rpconf' in kargs else rpconf)

    test.set('calibrate', True if args.cal else False)

    plot = test.getcreate('plot', default = True)
    test.set('plot', plot if args.plot is None else args.plot)

    if not args.hp is None: test.set('highpass', args.hp)
    if not args.lp is None: test.set('lowpass', args.lp)
    if not args.blp is None: test.set('butter_lp', args.blp)
    if not args.notch is None: test.set('notch', args.notch)
    if not args.resonance is None: test.set('resonance', args.resonance)

    # Now set the new tag
    tag = gen_test_type(test)

    if tag is None:
        # configure defaults
        tag = today
        test.set('highpass', True) # default test

    # set the new tag
    tag = kargs['tag'] if 'tag' in kargs else tag
    tag = tag if args.tag is None else args.tag
    test.set('tag', tag)

    test.getcreate('start_freq', default = 1e3)
    test.getcreate('stop_freq', default = 4e7)
    test.getcreate('rbw', default = 1000)
    test.getcreate('points', default = 1001)
    if hasattr(test, 'butter_lp') and test.butter_lp: test.getcreate('gain', default = 0.1)
    else: test.getcreate('gain', default = 1.0)
    test.getcreate('loops', default = 4)
    test.getcreate('fc', 8e4)

    if hasattr(test, 'butter_lp') and test.butter_lp:
        test.getcreate('butterworth_order', default = 3)
    else: test.pop('butterworth_order')

    print('test config: {}'.format([(k, v) for k,v in test.dump()]))

    return test

def test_sequence(t):
    for t.iter_count in range(t.test_vectors.shape[0]):
        # clear old results
        if hasattr(t, 'iirf_gain'): t.pop('iirf_gain')
        tf_characterisation_clear(t)
        for item in t.plot_items:
            t.run(['{}_clear'.format(item)])

        loops_samples = round(125e6 / t.test_vectors[t.iter_count, t.fc_idx])
        samples = loops_samples / t.test.loops

        t.set('fc', 125e6 / (t.test.loops * samples))
        t.set('omegac', 2 * np.pi * t.fc)

        generate_poles_zeros(t)

        t.set('gain_adjust', t.test_vectors[t.iter_count, t.gain_adjust_idx])

        if DEBUG:
            log_gain_adjust(t, t.gain, t.gain_adjust, 'gain and gain adjust')
            log_zp(t, t.zeros, t.poles, 's-plane zeros and poles')

        # configure iir with the poles, zeros and gain
        iirf_setup(t)
        if t.test.calibrate:
            gain_adjust = iirf_gain_adjust_correction(t)
            if (hasattr(t.test, 'butter_lp') and t.test.butter_lp) or (hasattr(t.test, 'lowpass') and t.test.lowpass) or (hasattr(t.test, 'highpass') and t.test.highpass) :
                factor = np.abs(t.gain_adjust - gain_adjust) / (2 * t.gain_adjust)
                t.gain_adjust = gain_adjust * factor if t.gain_adjust < gain_adjust else gain_adjust / factor
            else:
                factor = 1.0
                t.gain_adjust = gain_adjust

            log_msg(t, 'factor: {} updated gain_adjust: {:0.6e}'.format(factor, t.gain_adjust))
            iirf_setup(t)

        log_basics(t)

        run_test(t)

        # measure the transfer function
        tf_characterisation_init(t)

        if hasattr(t, 'iter_count'):
            # save calibration result - t.fc_idx, t.gain_adjust_idx
            t.getcreate('results', np.empty((0, 2), dtype = float))

            result = np.array([[t.fc, t.tf_measurement.gain_adjust_correction]], dtype = float)
            t.results = np.concatenate((t.results, result))

        if hasattr(t.test, 'plot') and t.test.plot and hasattr(t, 'iter_count') and t.iter_count % 5 == 0:
            plot_results(t)
            log_extended(t)

if __name__ == '__main__':
    RPCONFIG = 'filters.filters'
    test = get_run_conf(testname = 'filters tests', rpconf = RPCONFIG, dataname = 'data')

    t = TestBench(test)

    iirf_init(t)
    network_analyser_init(t)
    #t.set('tf_items', ['desgn', 'meas', 'pz'])
    #t.set('tf_items', ['desgn', 'meas', 'partial'])
    t.set('tf_items', ['desgn', 'meas', 'coef_trad', 'coef_comp'])
    t.set('plot_items', ['s_plane_proper', 'z_plane_pz', 'tf_dbs', 'tf_phases']) # s_plane_pz
    t.set('plot_sizes', {'s_plane': {'ncols': 1}, 'z_plane': {'ncols': 1}, 'tf': {'nrows': 1}})

    t.set('svg', True)

    # 'fpga_coef_init' generates the values saved to the fpga

    # run single item
    # t.run(['configure_gain', 'generate_poles_zeros', 'iirf_setup', 'log_basics', 'run_test', 'tf_characterisation_init', 'plot_results', 'log_extended'])
    # run sequence
    t.run(['get_create_gain_adjust', 'test_sequence'])

    # finish saving test results
    log_msg(t, '========= END TEST ============')
    if hasattr(t, 'test_vectors'):
        print('test_vectors\n', t.test_vectors)
        log_msg(t, 'test_vectors: {}'.format(t.test_vectors))
    if hasattr(t, 'results_cal'):
        print('test calibration\n', t.results_cal)
        log_msg(t, 'test calibration: {}'.format(t.results_cal))
    if hasattr(t, 'results'):
        print('test results\n', t.results)
        log_msg(t, 'test results: {}'.format(t.results))

    test.save(test.create_filename(test.tag))
    cal = '_cal' if t.test.calibrate else ''
    filename = '{}_{}{}.npz'.format(t.test.dataname, t.test.tag, cal)
    if hasattr(t, 'test_vectors') and hasattr(t, 'results'):
        np.savez_compressed(filename, test_vectors = t.test_vectors, results = t.results)
    elif hasattr(t, 'test_vectors'):
        np.savez_compressed(filename, test_vectors = t.test_vectors)
