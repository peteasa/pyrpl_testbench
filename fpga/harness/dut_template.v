module dut(
   // Outputs
   dut_sready, dut_cwait, dut_sample, dut_cready, dut_data, dut_addr,
   dut_active,
   // Inputs
   clk, nreset, sample_data, ctrl_ready, ctrl_data, ctrl_addr,
   ctrl_write
   /*AUTOARG*/);

   parameter  SW              = 16;
   parameter  AW              = 16;
   parameter  DW              = 32;

   localparam WC              = 3;

   input                clk;
   input                nreset;
   input  [SW-1:0]      sample_data; // assumed to be a continuous stream

   input                ctrl_ready;
   input  [DW-1:0]      ctrl_data;
   input  [AW-1:0]      ctrl_addr;
   input                ctrl_write;
   output               dut_cwait;  // optional delay to inbound ctrl data

   output               dut_sready; // generated when processed sample is ready
   output [SW-1:0]      dut_sample;

   output               dut_cready; // ready when read data is ready
   output [DW-1:0]      dut_data;
   output [AW-1:0]      dut_addr;   // optional read address - helpful in the trace file

   output               dut_active;

   // read wait cycle counter
   reg [WC-1:0]         wait_count;

   /*AUTOWIRE*/
   // Beginning of automatic wires (for undeclared instantiated-module outputs)
   wire [13:0]          dat_o;                  // From red_pitaya_module
   wire [31:0]          rdata;                  // From red_pitaya_module
   // End of automatics

   // update this as appropriate
   /* red_pitaya_block AUTO_TEMPLATE (
                                );
   */
   // red_pitaya_block red_pitaya_block(/*AUTOINST*/);

   assign clk_i  = clk;
   assign rstn_i = nreset;

   assign dat_i[13:0]        = sample_data[13:0];

   assign wen                = ctrl_write;
   assign ren                = ~ctrl_write;

   assign addr               = ctrl_addr;
   assign wdata[DW-1:0]      = ctrl_data;

   // demonstration of simple non-pipelined read
   // each read cycle delays the ctrl command with dut_cwait
   // allowing time for the module to respond with the read data
   always @ (posedge clk or negedge nreset)
     begin
        if(!nreset | data_ready)
          begin
             wait_count  <= 0; // 1; for zero wait
          end
        else if (read_cycle & ~(|wait_count) & ~data_ready)
          begin
             wait_count <= 1; // # wait clock cycles
          end
        else if (read_cycle & ~data_ready)
          begin
             wait_count <= wait_count - 1;
          end
     end

   // update this as appropriate for the module
   assign my_address = ((1 << (AW-1)) < ctrl_addr) | (0 < ctrl_addr);

   assign read_cycle         = ~ctrl_write & iir_module_addr & ctrl_ready;
   assign data_wait          = read_cycle & ~(wait_count == 1);
   assign data_ready         = (wait_count == 1);

   assign dut_data[DW-1:0]   = rdata;
   assign dut_addr[AW-1:0]   = ctrl_addr;
   assign dut_cready         = ctrl_ready & read_cycle & data_ready;
   assign dut_cwait          = read_cycle & ~data_ready;

   assign dut_active         = 1'b1;
   assign dut_sample[SW-1:0] = {{(SW-14){1'b0}},dat_o};
   assign dut_sready         = 1'b1;

endmodule // dut

// Local Variables:
// verilog-library-directories:( "." )
// End:
