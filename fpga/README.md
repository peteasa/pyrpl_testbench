This fpga test harness is for use with PyRPL fpga.

## Instruction for use

Install [Icarus Verilog](https://steveicarus.github.io/iverilog/) and [gtkwave](https://github.com/gtkwave/gtkwave).
Then clone https://github.com/peteasa/pyrpl_testbench to a suitable location.  Clone the pyrpl repository to a suitable location.
Now make suitable symbolic links to the pyrpl source
```
cd pyrpl_testbench
ln -s ../../pyrpl/pyrpl .
cd fpga
ln -s ../pyrpl/fpga/rtl .
```
The folder pyrpl_testbench/fpga/scripts contains [gen_test.py](https://github.com/peteasa/pyrpl_testbench/blob/main/fpga/scripts/gen_test.py) that can be modified and run to create test vectors.
The example dut for the IIR module is included in the folder pyrpl_testbench/fpga/iir ([dut_iir.v](https://github.com/peteasa/pyrpl_testbench/blob/main/fpga/iir/dut_iir.v)).
The folder pyrpl_testbench/fpga contains [run.sh](https://github.com/peteasa/pyrpl_testbench/blob/main/fpga/run.sh) that can be run to compile with iverilog and run the module test harness.  This results in a file waveform.vcd that can be loaded and viewed in iverilog.
```
cd pyrpl_testbench/fpga
scripts/gen_test.py
./run.sh iir
gtkwave waveform.vcd &
```
If for some reason the simulation does not finish quickly type {CTRL-C} to stop the simulation and type "finish" at the iverilog prompt.  Then load the waveform.vcd into gtkwave to find out what went wrong!

There is a saved gtkwave session that loads the demo waveform.vcd
```
gtkwave demo/iir_signals.gtkw
```

## Brief Description
The [harness/dv_top.v](https://github.com/peteasa/pyrpl_testbench/blob/main/fpga/harness/dv_top.v) code expects to find a "dut" module in a file named <name>/dut_<name>.v (for example iir/dut_iir.v).  The script [run.sh](https://github.com/peteasa/pyrpl_testbench/blob/main/fpga/run.sh) compiles the specified "dut" module with [harness/dv_top.v](https://github.com/peteasa/pyrpl_testbench/blob/main/fpga/harness/dv_top.v) and the other supporting verilog code from harness/*, <name>/*.  If there are any remaining unknows modules then they are picked up from the pyrpl fpga source in the folder harness/rtl that is simply a symbolic link to pyrpl/pyrpl/fpga/rtl.

Note that the iverilog compiler looks at the various folders in order so if you need to "override" or modify an existing verilog module then simply make a copy of the module and add the copy to the "dut" test folder so that iverilog finds that version of the module ahead of the "released" version.  Make your modifications in the copy and run the simulation until you are happy to release the changes then move the release version to the pyrpl/rtl folder replacing the original "released" file.

When run the iverilog compiled binary looks for a file "test_0.emf" that contains test vectors to send to the "dut" to exercise the fpga code.  The helper script ([gen_test.py](https://github.com/peteasa/pyrpl_testbench/blob/main/fpga/scripts/gen_test.py)) demonstrates how these test vectors can be created.  Three files are created during the simulation.  waveform.vcd can be read by gtkwave, test_ctrl_0.trace contains any read data produced during the simulation and test_sample_0.trace contains sample data produced during the simulation.

## Acknowledgements
Many thanks to the [OH](https://github.com/aolofsson/oh) project that inspired this test harness.

## License
Please read the license file [LICENSE](https://github.com/peteasa/pyrpl_testbench/blob/main/LICENSE) for more information.
