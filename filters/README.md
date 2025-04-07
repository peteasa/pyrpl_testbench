This folder contains a selection of simple scripts to explore the capabilities of IIR module of pyrpl.

## Preamble
There are multiple online resources that discuss Infinite Impulse Response filters for example the section [Analysing IIR filters](https://brianmcfee.net/dstbook-site/content/ch12-ztransform/PoleZero.html#zeros) in Brian McFee's online Digital Signals Theory Book.  These resources are preparation for using the pyrpl IIR filter module however there are significant pieces of additional knowledge that are needed to use the pyrpl IIR filter module that are not well described in the [official pyrpl](https://pyrpl.readthedocs.io/en/latest/api.html#module-pyrpl.hardware_modules.iir) documentation on the subject.

Notably the numerical values used for poles and zeros in many online resources normalise the numerical value to $`\omega`$, the digital equivalent to the angular frequency.  In the pyrpl software this normalisation is not performed so the numbers used are multiples of the desired cutoff frequency. The digital angular frequency is defined in the range $`\omega = 0 \rightarrow \pi`$ and is simply $`\omega = 2 \pi {f \over f_s}`$ where $`f_s`$ is the sampling frequency of the system. The maximum frequency modeled with these digital filters ($`\omega = \pi`$) corresponds to the well known Nyquist freqency $`f = {f_s \over 2}`$.  The units of $`\omega`$ are radians per sample.

This folder contains examples of filters that have been designed using the pyrpl IIR filter module. so that interested developers can quickly create working examples of the complex filters that they require.

## Note
The PyRPL IIR filter design is a novel design that reduces the number of gates needed in the FPGA implementation.  You might see in my python code an attribute for configuring biproper (not just proper) poles / zeros conjugate pairs.  To use this option you also need to modify IirFilter::coefficients() as follows:
```
        zd = np.exp(np.asarray(z, dtype=np.complex128)*self.dt*self.loops)
        pd = np.exp(np.asarray(p, dtype=np.complex128)*self.dt*self.loops)
        if self.biproper:
            logger.debug('z-plane number of zeros: {} number of poles: {}'.format(zd.shape, pd.shape))
            while len(zd) < len(pd):
                zd = np.append(zd, complex(-1, 0))
                logger.debug('ensure H[z] = 0 at highest frequency: adding z-plane zero at (-1+j0)')

        rd, cd = residues(zd, pd, k)
```
The while loop in this code adds zeros at z = -1 to make the system a biproper system.  You can find a discussion about this in Neil Robertson's blog [Design IIR Butterworth Filters](https://www.dsprelated.com/showarticle/1119.php).  The reason why I have not yet released this change is that it only has an effect at very high frequencies. At high frequencies there are other more prominant effects (notably the delay through the filter) that have to be taken into account to correctly model the IIR filter.  You can find more discussion about the detailed design in one of my [dsp stackexchange questions](https://dsp.stackexchange.com/questions/95906/how-to-debug-this-biquad-iir-filter-that-uses-polyphase-decimation).  If you would like me to release this let me know via the discussion pages and I will do the work needed to push the changes to this testbench and to my version of PyRPL.

Comments are always welcome!
