#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def append_config(f, config, items, oper, delay = 0x0):
    for name in items:
        item = 0; axis = 1; lterm = ' // {}\n'.format(name)
        value = np.array([[0x0],[config[name]['addr']],[config[name]['value']],
                          [oper],[delay]], dtype = np.uint32)
        append_file(f, value, item, axis, lterm, 16, 16, 32, 4, 16)

def append_configs(fn, loops):
    on = 0x1
    off = 0x0
    shortcut_on = 0x03
    config = {'LOOP': {'addr': 0x100, 'value': loops},
              'SHORTCUT': {'addr': 0x104, 'value': 0x0},
              'OVERFLOW': {'addr': 0x108, 'value': 0},
              'SET_FILTER': {'addr': 0x120, 'value': 0},
              'IIRBITS': {'addr': 0x200, 'value': 0},
              'IIRSHIFT': {'addr': 0x204, 'value': 0},
              'IIRSTAGES': {'addr': 0x208, 'value': 0},
              'FILTERSTAGES': {'addr': 0x220, 'value': 0},
              'FILTERSHIFTBITS': {'addr': 0x224, 'value': 0},
              'FILTERMINBW': {'addr': 0x228, 'value': 0}
    }

    with open(fn, 'a') as f:
        write = ['LOOP', 'SET_FILTER']
        append_config(f, config, write, 0x1)

        read = ['LOOP', 'SHORTCUT', 'OVERFLOW', 'SET_FILTER', 'IIRBITS', 'IIRSHIFT', 'IIRSTAGES',
                'FILTERSTAGES', 'FILTERSHIFTBITS', 'FILTERMINBW']
        append_config(f, config, read, 0x0)

        config['SHORTCUT']['value'] = on
        write = ['SHORTCUT']
        append_config(f, config, write, 0x1, delay = 0x0100)

        read = ['OVERFLOW', 'OVERFLOW', 'OVERFLOW']
        append_config(f, config, read, 0x0, delay = 0x100)

        config['SHORTCUT']['value'] = off

        # comment out to see steady state at end of trace
        #write = ['SHORTCUT']
        #append_config(f, config, write, 0x1)

        read = ['OVERFLOW', 'OVERFLOW', 'OVERFLOW']
        append_config(f, config, read, 0x0)

def append_coefficients(fn, loops, coefs):
    cnames = ['b0', 'b1', 'a1', 'a2']
    cdef = {'b0': 0, 'b1': 0, 'a1': 1, 'a2': 1}
    with open(fn, 'a') as f:
        for l in range(loops):
            addr = 0x8000 + 4 * (l * 2) * 4
            for idx, cn in enumerate(cnames):
                cvalue = coefs[idx, l] if l < coefs.shape[1] else cdef[cn]
                value = np.array([[0], [addr], [cvalue], [0x1], [0x0]], dtype = int)
                item = 0; axis = 1; lterm = ' // {}:{}\n'.format(l, cn)
                append_file(f, value, item, axis, lterm, 16, 16, 32, 4, 16)
                value = np.array([[0], [addr+4], [0x0], [0x1], [0x0]], dtype = int)
                lterm = '\n'.format(cn)
                append_file(f, value, item, axis, lterm, 16, 16, 32, 4, 16)
                addr = addr + 8


def read_coefficients(fn, loops):
    cnames = ['b0', 'b1', 'a1', 'a2']
    with open(fn, 'a') as f:
        for l in range(loops):
            addr = 0x8000 + 4 * (l * 2) * 4
            for idx, cn in enumerate(cnames):
                cvalue = 0
                value = np.array([[0], [addr], [cvalue], [0x0], [0x0]], dtype = int)
                item = 0; axis = 1; lterm = ' // read {}:{}\n'.format(l, cn)
                append_file(f, value, item, axis, lterm, 16, 16, 32, 4, 16)
                value = np.array([[0], [addr+4], [cvalue], [0x0], [0x0]], dtype = int)
                lterm = '\n'.format(cn)
                append_file(f, value, item, axis, lterm, 16, 16, 32, 4, 16)
                addr = addr + 8

def calc_times(**kargs):
    fs = kargs['fs'] if 'fs' in kargs else 125e6
    n = kargs['n'] if 'n' in kargs else 50

    return np.linspace(0, (n - 1) / fs, num = n)

def sin(a, f, **kargs):
    fs = kargs['fs'] if 'fs' in kargs else 2 * f
    if not 'fs' in kargs:
        kargs['fs'] = fs

    times =  kargs['times'] if 'times' in kargs else calc_times(**kargs)

    y = a * np.sin(2 * np.pi * f * times)

    return times, y, 'sin'

def encode(value, bits):
    digits = int(np.ceil(bits / 4))
    if value < 0:
        value = decode('7' + 'F'*15) + value + 1

    for d in range(digits):
        if d < 1:
            svalue = '{:x}'.format(value)
        svalue = '0{}'.format(svalue)

    slen = len(svalue)
    start = slen - digits

    return svalue[start:slen]

def decode(value):
    return int(value, 16)

def append_file(f, values, item, axis, lterm, *arg):
    entry = np.take(values, item, axis = axis)
    for idx, bits in enumerate(arg):
        if idx:
            line = '{}_{}'.format(line, encode(entry[idx], bits))
        else:
            line = '{}'.format(encode(entry[idx], bits))

    f.write('{}{}'.format(line, lterm))

def save(fn, values, *arg, smode = 'w', lterm = '\n'):
    axis = len(values.shape) - 1
    with open(fn, smode) as f:
        for item in range(values.shape[axis]):
            append_file(f, values, item, axis, lterm, *arg)

def load(fn):
    empty = []
    values = empty
    with open(fn, 'r') as f:
        for line in f:
            items = line.split(' ')[0].split('_')
            if len(empty) < 1:
                for i in range(len(items)):
                    empty.append([])

                values = np.array(empty, dtype = np.uint32)

            nitems = []
            for item in items:
                nitems.append([decode(item)])

            values = np.append(values, nitems, axis = 1)

    return values

def test_save_load():
    values = np.array([[],[],[],[],[]], dtype = np.uint32)
    i = 1
    for i in range(0xf):
        values = np.append(values, [[i], [0xffff - i], [i*i], [0], [-1]], axis=1)

    save('test_save_load.emf', values, 16, 16, 32, 4, 16, smode = 'w', lterm = ' // sample\n')
    results = load('test_save_load.emf')

    print(results)

if __name__ == '__main__':
    # c array is column major
    # armadillo is column major
    # python is row major

    fn = 'test_0.emf'

    values = np.array([[],[],[],[],[]], dtype = np.uint32)

    fs = 125e6
    freq = 12e6
    n = round(fs / freq)
    freq = fs / n

    # more than once clk per sample
    nclk = 4
    times = calc_times(fs = fs * nclk, n = n * nclk)
    times, y, fnname = sin(0x7FF, freq, times = times)

    for i in range(times.shape[0]):
        values = np.append(values, [[int(y[i]+0x800)], [0x0], [0x0], [0x8], [0x0]], axis = 1)

    save(fn, values, 16, 16, 32, 4, 16, smode = 'w', lterm = ' // {} sample\n'.format(fnname))

    loops = 6
    coefs = np.array([[],[],[],[]], dtype = np.uint32)
    #cnames = ['b0', 'b1', 'a1', 'a2']
    for i in range(loops - 1):
        coefs = np.append(coefs, [[0x10000A+i],[0x10100A+i],[0x10200A+i],[0x10300A+i]], axis = 1)

    coefs = np.append(coefs, [[0x100000F],[0x100100F],[0x100200F],[0x100300F]], axis = 1)
    append_coefficients(fn, loops, coefs)

    append_configs(fn, loops)
