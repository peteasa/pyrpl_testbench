/* verilator lint_off STMTDLY */
module dv_ctrl(/*AUTOARG*/
   // Outputs
   nreset, clk, start, vdd, vss,
   // Inputs
   dut_active, cmds_done, test_done
   );

   parameter CFG_CLK_PERIOD  = 10;
   parameter CFG_CLK_PHASE   = CFG_CLK_PERIOD/2;
   parameter CFG_TIMEOUT     = 5000;

   output nreset;     // async active low reset
   output clk;       // main clock
   output start;      // start test (level)
   output vdd;        // driving vdd
   output vss;        // driving vss

   input  dut_active; // reset sequence is done
   input  cmds_done;  //stimulus is done
   input  test_done;  //test is done

   //signal declarations
   reg 	     vdd;
   reg 	     vss;
   reg 	     nreset;
   reg 	     start;
   reg 	     clk=0;
   reg [6:0] clk_phase;
   integer   seed,r;

   //#################################
   // RANDOM NUMBER GENERATOR
   // (SEED SUPPLIED EXERNALLY)
   //#################################
   initial
     begin
	r=$value$plusargs("SEED=%s", seed);
	$display("SEED=%d", seed);
`ifdef CFG_RANDOM
	clk_phase = 1 + {$random(seed)}; //generate random values
`else
	clk_phase = CFG_CLK_PHASE;
`endif
	$display("clk_phase=%d", clk_phase);
     end

   //#################################
   //CLK GENERATOR
   //#################################

   always
     #(clk_phase) clk = ~clk; //add one to avoid "DC" state

   //#################################
   //ASYNC
   //#################################

   initial
     begin
	#(1)
	nreset   = 'b0;
	vdd      = 'b0;
	vss      = 'b0;
	#(clk_phase * 10 + 100)   //ramping voltage
	vdd      = 'bx;
	#(clk_phase * 10 + 100)   //voltage is safe
	vdd      = 'b1;
	#(clk_phase * 40 + 100)   //hold reset for 20 clk cycles
	nreset   = 'b1;
     end

   //#################################
   //SYNCHRONOUS STIMULUS
   //#################################
   //START TEST
   always @ (posedge clk or negedge nreset)
     if(!nreset)
       start <= 1'b0;
     else if(dut_active)
       start <= 1'b1;

   //STOP SIMULATION
   always @ (posedge clk)
     //#(CFG_TIMEOUT) $finish;
     if(cmds_done & test_done)
       #(CFG_TIMEOUT) $finish;

   //WAVEFORM DUMP
   //Better solution?
`ifndef VERILATOR
   initial
     begin
	$dumpfile("waveform.vcd");
	$dumpvars(0, dv_top);
     end
`endif

endmodule // dv_ctrl
