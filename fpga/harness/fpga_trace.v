module fpga_trace(/*AUTOARG*/
   // Inputs
   clk, nreset, dut_sample, dut_sready, dut_cready, dut_data,
   dut_addr, dut_cwait
   );

   // stimulus
   parameter SW     = 16;             // size of sample
   parameter AW     = 16;             // size of address
   parameter DW     = 32;             // size of data
   parameter CRW    = 4;              // size of data control r/w bit
   parameter MAW    = 16;             // stimulus count width
   parameter MD     = 1<<(MAW-1);     // max number of entries in the file
   parameter INDEX  = 1;
   parameter NAME   = "not_declared";

   // Inputs
   input           clk;
   input           nreset;

   input  [SW-1:0]      dut_sample;
   input                dut_sready;

   input                dut_cready;
   input  [DW-1:0]      dut_data;
   input  [AW-1:0]      dut_addr;
   input                dut_cwait;

   reg [31:0] 	    ftrace;
   reg [255:0] 	    tracefile;
   reg [31:0] 	    strace;
   reg [255:0] 	    samplefile;

   /*AUTOWIRE*/

   // Dumps into
   initial
     begin
	#10
	  $sformat(tracefile,"%0s_ctrl_%0h%s",NAME,INDEX,".trace");
	ftrace  = $fopen({tracefile}, "w");
     end

   always @ (posedge clk or negedge nreset)
     if(nreset & dut_cready & ~dut_cwait)
       begin
	  $fwrite(ftrace, "%h_%h\n", dut_addr, dut_data);
	  // $display("%h_%h\n", dut_addr, dut_dat);
       end

   // Dumps into
   initial
     begin
	#10
	  $sformat(samplefile,"%0s_sample_%0h%s",NAME,INDEX,".trace");
	strace  = $fopen({samplefile}, "w");
     end

   always @ (posedge clk or negedge nreset)
     if(nreset & dut_sready)
       begin
	  $fwrite(strace, "%h\n", dut_sample);
	  // $display("%h\n", dut_sample);
       end

endmodule // fpga_trace

// Local Variables:
// verilog-library-directories:( "." )
// End:
