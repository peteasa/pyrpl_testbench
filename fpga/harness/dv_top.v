/*
 * Generic Top level, each modules implement a dut driven by this
 * Stimulus is provided by a file named test_0.emf
 *
 *
 */
`timescale 1ns/1ps
module dv_top(/*AUTOARG*/);

   parameter OUTWIDTH  = 8;
   parameter SW        = 16;
   parameter AW        = 16;
   parameter DW        = 32;
   parameter MAW       = 16;

   localparam NAME     = "test";
   localparam INDEX    = 0;

   dv_ctrl dv_ctrl (.test_done		(1'b1),   //optimize later
		            /*AUTOINST*/
                    // Outputs
                    .nreset             (nreset),
                    .clk                (clk),
                    .start              (start),
                    .vdd                (vdd),
                    .vss                (vss),
                    // Inputs
                    .dut_active         (dut_active),
                    .cmds_done          (cmds_done));

   /*AUTOWIRE*/
   // Beginning of automatic wires (for undeclared instantiated-module outputs)
   wire                 clk;                    // From dv_ctrl of dv_ctrl.v
   wire                 cmds_done;              // From dv_read_cmds of dv_read_cmds.v
   wire [AW-1:0]        ctrl_addr;              // From dv_read_cmds of dv_read_cmds.v
   wire [DW-1:0]        ctrl_data;              // From dv_read_cmds of dv_read_cmds.v
   wire                 ctrl_ready;             // From dv_read_cmds of dv_read_cmds.v
   wire                 ctrl_wait;              // From dv_read_cmds of dv_read_cmds.v
   wire                 ctrl_write;             // From dv_read_cmds of dv_read_cmds.v
   wire                 dut_active;             // From dut of dut.v
   wire [AW-1:0]        dut_addr;               // From dut of dut.v
   wire                 dut_cready;             // From dut of dut.v
   wire                 dut_cwait;              // From dut of dut.v
   wire [DW-1:0]        dut_data;               // From dut of dut.v
   wire [SW-1:0]        dut_sample;             // From dut of dut.v
   wire                 dut_sready;             // From dut of dut.v
   wire                 nreset;                 // From dv_ctrl of dv_ctrl.v
   wire [SW-1:0]        sample_data;            // From dv_read_cmds of dv_read_cmds.v
   wire                 start;                  // From dv_ctrl of dv_ctrl.v
   wire [MAW-1:0]       stim_count;             // From dv_read_cmds of dv_read_cmds.v
   wire                 vdd;                    // From dv_ctrl of dv_ctrl.v
   wire                 vss;                    // From dv_ctrl of dv_ctrl.v
   // End of automatics

   /* dv_read_cmds AUTO_TEMPLATE (
                                );
   */
   dv_read_cmds #(.NAME(NAME),
                  .INDEX(INDEX),
                  .SW(SW),
                  .AW(AW),
                  .DW(DW),
                  .MAW(MAW)
                  )
   dv_read_cmds(/*AUTOINST*/
                // Outputs
                .ctrl_ready             (ctrl_ready),
                .sample_data            (sample_data[SW-1:0]),
                .ctrl_addr              (ctrl_addr[AW-1:0]),
                .ctrl_data              (ctrl_data[DW-1:0]),
                .ctrl_write             (ctrl_write),
                .ctrl_wait              (ctrl_wait),
                .cmds_done              (cmds_done),
                .stim_count             (stim_count[MAW-1:0]),
                // Inputs
                .clk                    (clk),
                .nreset                 (nreset),
                .start                  (start),
                .dut_cwait              (dut_cwait));

   dut #(.SW(SW),
         .AW(AW),
         .DW(DW)
         )
   dut(/*AUTOINST*/
       // Outputs
       .dut_cwait                       (dut_cwait),
       .dut_sready                      (dut_sready),
       .dut_sample                      (dut_sample[SW-1:0]),
       .dut_cready                      (dut_cready),
       .dut_data                        (dut_data[DW-1:0]),
       .dut_addr                        (dut_addr[AW-1:0]),
       .dut_active                      (dut_active),
       // Inputs
       .clk                             (clk),
       .nreset                          (nreset),
       .sample_data                     (sample_data[SW-1:0]),
       .ctrl_ready                      (ctrl_ready),
       .ctrl_data                       (ctrl_data[DW-1:0]),
       .ctrl_addr                       (ctrl_addr[AW-1:0]),
       .ctrl_write                      (ctrl_write));

   assign sready = dut_sready & ~cmds_done;
   /* fpga_trace AUTO_TEMPLATE (
                                );
   */
   fpga_trace #(.NAME(NAME),
                .INDEX(INDEX),
                .SW(SW),
                .AW(AW),
                .DW(DW)
                )
   fpga_trace(.dut_sready               (sready),
              /*AUTOINST*/
              // Inputs
              .clk                      (clk),
              .nreset                   (nreset),
              .dut_sample               (dut_sample[SW-1:0]),
              .dut_cready               (dut_cready),
              .dut_data                 (dut_data[DW-1:0]),
              .dut_addr                 (dut_addr[AW-1:0]),
              .dut_cwait                (dut_cwait));


endmodule // dv_top

// Local Variables:
// verilog-library-directories:( "." )
// End:
