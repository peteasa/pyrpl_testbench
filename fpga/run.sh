#!/bin/bash

DUT=$1/dut_$1.v

# also any overrides can be placed in ./$1 to compile in place
# of equivalent modules found in ./rtl

iverilog -g2005\
 -DTARGET_SIM=1\
 $DUT\
 -y ./$1 \
 harness/dv_top.v\
 -y ./harness \
 -y ./rtl \
 -o dut.bin\

./dut.bin

# To view the trace run
# gtkwave waveform.vcd

# or if you have previously "File; Write Save File As"
# gtkwave iir/iir_signals.gtkw
