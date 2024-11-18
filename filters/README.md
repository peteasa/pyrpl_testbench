This folder contains a selection of simple scripts to explore the capabilities of IIR module of pyrpl.

## Preamble
There are multiple online resources that discuss Infinite Impulse Response filters for example the section [Analysing IIR filters](https://brianmcfee.net/dstbook-site/content/ch12-ztransform/PoleZero.html#zeros) in Brian McFee's online Digital Signals Theory Book.  These resources are preparation for using the pyrpl IIR filter module however there are significant pieces of additional knowledge that are needed to use the pyrpl IIR filter module that are not well described in the [official pyrpl](https://pyrpl.readthedocs.io/en/latest/api.html#module-pyrpl.hardware_modules.iir) documentation on the subject.

Notably the numerical values used for poles and zeros in many online resources normalise the numerical value to $`\omega`$, the digital equivalent to the angular frequency.  In the pyrpl software this normalisation is not performed so the numbers used are multiples of the desired cutoff frequency. The digital angular frequency is defined in the range $`\omega = 0 \rightarrow \pi`$ and is simply $`\omega = 2 \pi {f \over f_s}`$ where $`f_s`$ is the sampling frequency of the system. The maximum frequency modeled with these digital filters ($`\omega = \pi`$) corresponds to the well known Nyquist freqency $`f = {f_s \over 2}`$.  The units of $`\omega`$ are radians per sample.

This folder contains examples of filters that have been designed using the pyrpl IIR filter module. so that interested developers can quickly create working examples of the complex filters that they require.

## Note
At the time of writing the author is not convinced that the IIR module is free of bugs.  One example of this is that the filters included in this section seem to require gain settings that are remarkably small.  Setting `gain = 1.0` in the pyrpl IIR module rarely produces a working filter.  The only example found so far is that of a notch filter.  As an example the simple high pass filter requires `gain = 1e-6` to ensure that the pass band gain is 1.0.  This may be a bug in this implmentation or it may be a bug in the IIR module or it may simply be working as designed.

Comments are always welcome!
