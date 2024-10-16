This testbench is for use with PyRPL.

PyRPL (Python RedPitaya Lockbox) turns the RedPitaya hardware into a powerful DSP device, especially suitable as a digital lockbox and measurement device in quantum optics experiments.

The testbench allows you to create simple projects with sequences of operations on the modules provided by PyRPL.  An example use of the testbench is demonstrated with the measurement of small resistances (mohm/mohm.py).

## PyRPL
The official PyRPL website address is [http://pyrpl.readthedocs.io/](http://pyrpl.readthedocs.io).  The official PyRPL software is available at [https://github.com/RedPitaya/pyrpl](https://github.com/RedPitaya/pyrpl).  Note: at the time of writing the Python3 branch of this repository is not being actively maintained.

## Installation
The simplest way to install PyRPL on Ubuntu 24.04.1 is to clone https://github.com/peteasa/pyrpl to a suitable location and then checkout the [consolidate_python3 branch](https://github.com/peteasa/pyrpl/tree/consolidate_python3) then install the necessary python libraries
```
sudo apt install python3-ipython python3-numpy python3-pandas python3-scp python3-paramiko python3-scipy python3-ruamel.yaml python3-qtpy python3-qasync python3-pyqtgraph python3-nest-asyncio python3-matplotlib python3-yaml
```
The folder pyrpl/pyrpl contains the python library modules for PyRPL.  These can be imported into a python3 project provided the pyrpl/pyrpl folder is present in the python3 project for example:
```
cd pyrpl_testbench/mohm
ln -s ../../pyrpl/pyrpl .
```
Other installation methods may not work for Python3.

## TestBench API
One of the objectives of the TestBench is to encourage the creation of small modular methods that perform a specific task, for example the setting up of an PyRPL module.  If this is achieved in any one test design then it will be possible to reuse those modules easily in another test design.  The TestBench and associated classes are kept as simple as possible so that they can be used in multiple designs.

TestBench.run(seq) - the argument is a list with method names to run in sequence for example:
```
    t.run(['iq_setup', 'spec_setup', 'test_sequence'])
```

Each method is passed the test bench as an argument.  Attributes added to the test bench are defined for a given test iteration.  For example the attribute t.fs might represent one signal frequency in a sequence of frequencies used in a test.

The test bench has a test attribute that may for example contain the start (fs_start), end (fs_end) and number of frequencies in the sequence (count).  For example:
```
    fs = freq_log_list(t.test.fs_start, t.test.fs_end, t.test.count)
```

The important test parameters (for example fs_start, fs_end..) can be saved at the end of a test and loaded at the start of a new test sequence
```
    t.test.save(t.test.create_filename(t.test.tag))
```
This enables a subsequent invocation of the same test to be run with the same test vectors.

For example the first invocation of a test sequence could be used to provide a calibration set (t.test.calibrate).  Using the frequency example the Asg (RedPitaya signal generator) does not support an infinite number of frequencies, instead it rounds the frequency values proportional to the maximum available sample rate.  The amplitude response of the RedPitaya output and input ports for the different frequencies can be recorded during the calibration phase.  These calibration values can then be used in the test run to provide a measure of the ratio between the signal voltages and the calibration voltages over the test frequency range.  The calibration data provides both the frequencies used in the calibration steps and the amplitude of the RedPitaya output at that frequency.  So the same frequencies can be used in the test steps and the DUT response can be compared with the calibrated response.
```
    t.test.save(t.test.create_filename(t.test.tag))
    cal = '_cal' if t.test.calibrate else ''
    filename = '{}_{}{}.npz'.format(t.test.dataname, t.test.tag, cal)
    if hasattr(t, 'test_vector') and hasattr(t, 'results'):
        np.savez_compressed(filename, test_vector = t.test_vector, results = t.results)
    elif hasattr(t, 'test_vector'):
        np.savez_compressed(filename, test_vector = t.test_vector)
```
and on load
```
    with np.load(filename) as data:
        if t.test.testype == 'freq' and 'test_vector' in data and data['test_vector'].shape[0] == seq.shape[0]:
            rtn = freq_calibration(t, seq, data)
        elif 'test_vector' in data and 0 < data['test_vector'].shape[0]:
            rtn = amp_calibration(t, seq, data)
```

If command line arguments are used in a specific test then it is recommended that the command line arguments take priority over the parameters recovered from a previous test run and over parameters hard coded in the test code.  This is achieved by using the method getcreate(name, default=value).
```
    # command line arguments override kargs overrides values loaded from the test config file
    t.test.load()
    t.test.getcreate('fs_start', 1000 )
    fs_start = kargs['fs_start'] if 'fs_start' in kargs else t.test.fs_start
    fs_start = args.fs_start if fs_start in args else fs_start
    t.test.fs_start = fs_start
```

There are several methods to get or create attributes on the test or TestBench objects (get(), set(), getcreate() and set_test_ifcal()).  The latter method is used specifically when the calibration step is used to define the type of test or specify a specific test vector.  In the calibration case the design could be that the test vector provided will override whatever the current value in the test definition (ie the set() method is used to create / override the current attribute). When performing a subsequent test run the same point in the code without calibrate defined will use the existing value (using getcreate() method) rather than override the test vector.  In the case where there is no prior calibration data then the getcreate() method will create the attribute for the test vector.

In addition to the creation of attributes on the TestBench and test definition objects it is possible to clear() all the recently created attributes or pop() a given attribute from the dictionary of custom attributes added to the object.

## PyRPL calibration
One important element missing from the toolbox provided by the PyRPL code is a calibration module for the RedPitaya tools. Calibration is a complicated subject and is best documented [here](https://redpitaya.readthedocs.io/en/latest/appsFeatures/systemtool/calibration.html) and [here](https://redpitaya.readthedocs.io/en/latest/developerGuide/hardware/hw_specs/fastIO.html#analog-inputs-calibration).  The internal RedPitaya calibration coefficients are not available to the PyRPL code and if they were the more complicated tools available with the PyRPL code may not be easy to calibrate.  Having said that it is useful to have a simple calibration object when using the PyRPL code.  The TestBench provides access (t.c) to some simple calibration tools.

Also provided is a python application that defines the simple calibration protocol.  calibration/calibrate.py provide another example use of the TestBench.

## Versions tested

RedPitaya OS 2.0: Latest Beta (2.05-37) Install 2.00-37 - Branch 2024.3
Python is 3.12.3
pyrpl 1.0.0.0 using
  IPython 8.20.0
  numpy 1.26.4
  pandas 2.1.4
  paramiko 2.12.0
  scipy 1.11.4
  scp 0.14.5
  ruamel.yaml 0.17.21
  yaml 6.0.1
  qtpy 2.4.1

## Acknowledgements
Many thanks to the RedPitaya team.  Without the excellent hardware and software provided none of this would be possible.
I would also like to acknowledge Leonhard Neuhaus, the original author of the PyRPL software, a truly world class toolbox of tools to use with the RedPitaya hardware.
The mohm test software was developed in response to K.G Libbrecht et al paper DOI[10.1119/1.1579497](https://pubs.aip.org/aapt/ajp/article-abstract/71/11/1208/1029926/A-basic-lock-in-amplifier-experiment-for-the?redirectedFrom=fulltext) A basic lock-in amplifier experiment for the undergraduate laboratory

## License
Please read the license file [LICENSE](https://github.com/peteasa/pyrpl_testbench/blob/main/LICENSE) for more information.
