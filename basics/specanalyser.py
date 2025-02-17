#!/usr/bin/env python3

import json
import numpy as np
import pyrpl

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

def prepare_to_show_plot(p):
    import matplotlib
    for i in range(1):
        try:
            matplotlib.use('TkAgg')
            break
        except:
            #print('{} qt is still running!'.format(10-i))
            pass

def characterise_spectrum(data, frequencies, condition):
    idxs = np.where(condition)
    fminidx = idxs[0].min()
    fmaxidx = idxs[0].max()
    amaxidx = fminidx + np.argmax(data[fminidx:fmaxidx])
    amaxidx_lo = amaxidx - 1 if 0 < amaxidx else 0
    amaxidx_hi = amaxidx + 1 if amaxidx + 1 < len(data) else len(data) - 1

    return amaxidx_lo, amaxidx, amaxidx_hi

if __name__ == '__main__':
    HOSTNAME = 'rp-f0bd75'
    CONFIG = 'basic.specanalyser'
    p = pyrpl.Pyrpl(config = CONFIG, hostname = HOSTNAME, gui = False, reloadfpga = True) # False)

    dataname = 'data'
    c = CalData()
    c.load()
    c.save(dataname)

    probe_attenuation = 1
    input_attenuation = 20 * probe_attenuation
    sa_input = 'in1' # 'off'

    # let's use a module manager for the asg
    title = ''
    label = ''
    if sa_input == 'off':
        input_attenuation = 1

    with p.asgs.pop('user') as asg:
        asg_raw_a = 1.0
        asg_raw_o = 0.0
        asg_output = 'out1' if sa_input == 'in1' else 'off'
        asg_o, asg_a = c.cal(asg_output, asg_raw_o, ag = asg_raw_a)
        asg.setup(frequency = 20000, waveform = 'square', trigger_source = 'immediately',
                  offset = asg_o, amplitude = asg_a,
                  output_direct = asg_output)

        print('{} waveform: {} f: {} offset: {} amplitude: {} output: {}'.format(
            asg.name, asg.waveform, asg.frequency, asg.offset, asg.amplitude, asg.output_direct))
        if asg.waveform == 'noise':
            title = '{} spectrum'.format(asg.waveform)
        else:
            title = '{} fs {:0.0f}'.format(asg.waveform, asg.frequency)

        # setup the spectrumanalyzer in baseband mode
        sa_input = sa_input if asg.output_direct == 'out1' else asg.name
        sa = p.spectrumanalyzer
        sa.setup(
            input1_baseband = sa_input,
            display_unit = 'Vpk',
            span = 125e6,
            window = 'blackman'
            )

        label = '{}'.format(sa.display_unit)

        # set baseband after centre
        sa.baseband = True

        print('{} input: {} center: {} window: {} span: {} acbandwidth: {} baseband: {} sa.decimation: {}'.format(
            sa.name, sa.input1_baseband, sa.center, sa.window, sa.span, sa.acbandwidth, sa.baseband, sa.decimation))

        # the return format is (spectrum for channel 1, spectrum for channel 2,
        # real part of cross spectrum, imaginary part of cross spectrum):
        res = p.spectrumanalyzer.single()

        frequencies = p.spectrumanalyzer.frequencies
        f_low = 1.25e8 / ((1.25e8 / asg.frequency) + 1000)

    to_units = lambda x:p.spectrumanalyzer.data_to_display_unit(x, p.spectrumanalyzer.rbw)
    spec_ch1 = to_units(res[0,:])
    spec_ch1 = c.ac(sa_input, spec_ch1)
    spec_ch1 = spec_ch1 * input_attenuation
    amaxidx_lo, amaxidx, amaxidx_hi = characterise_spectrum(spec_ch1, frequencies, f_low < frequencies)
    print('{} DC: {:0.4e} @ {:0.0f}'.format(label, spec_ch1[0], frequencies[0]))
    print('{} peak: {:0.4e} @ {:0.0f}'.format(label, spec_ch1[amaxidx], frequencies[amaxidx]))

    svg = False
    if svg:
        import matplotlib

        # enable svg plotting
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    # plot the spectrum
    plt.semilogx(frequencies, spec_ch1, label = label)
    plt.legend(loc = 'center left', bbox_to_anchor = (1.04, 0.5))
    plt.title(title)
    plt.tight_layout()
    if svg:
        plt.savefig('specanalyser.svg')
    else:
        plt.show()
