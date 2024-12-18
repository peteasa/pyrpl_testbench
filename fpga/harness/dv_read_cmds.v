/*
 * Reads samples from file
 * Reads commands from file and drives ctrl_address, ctrl_data, ctrl_write
 *
 *
 */
`timescale 1ns/1ps
module dv_read_cmds (/*AUTOARG*/
   // Outputs
   ctrl_ready, sample_data, ctrl_addr, ctrl_data, ctrl_write,
   ctrl_wait, cmds_done, stim_count,
   // Inputs
   clk, nreset, start, dut_cwait
   );

   // stimulus
   parameter SW     = 16;             // size of sample
   parameter AW     = 16;             // size of address
   parameter DW     = 32;             // size of data
   parameter CRW    = 4;              // size of data control select & r/w bit
   parameter MAW    = 16;             // stimulus count width
   parameter MD     = 1<<(MAW-1);     // max number of entries in the file
   parameter INDEX  = 1;
   parameter NAME   = "not_declared";
   parameter WAIT   = 0;
   parameter SAMPLE = 1<<(CRW-1);

   localparam SS     = AW+DW+CRW+MAW;
   localparam AS     = DW+CRW+MAW;
   localparam DS     = CRW+MAW;

   // Inputs
   input           clk;
   input           nreset;
   input           start;
   input           dut_cwait;

   // outputs
   output           ctrl_ready;
   output [SW-1:0]  sample_data;
   output [AW-1:0]  ctrl_addr;
   output [DW-1:0]  ctrl_data;
   output           ctrl_write;
   output           ctrl_wait;
   output           cmds_done;

   output [MAW-1:0] stim_count;

   // variables
   reg              dut_cwait_d1;
   reg              is_sample_p1;
   reg              is_sample_d2;

   reg [MAW-1:0]    sample_start;
   reg [MAW-1:0]    sample_end;
   reg              sample_state;
   reg [SW-1:0]     sample_data;

   reg [AW-1:0]     ctrl_addr;
   reg [DW-1:0]     ctrl_data;
   reg              ctrl_write;
   reg              ctrl_ready;

   reg [SS-1:0]     mem_data_d1;
   reg              mem_ready_d1;
   reg [SW+SS-1:0]  stimarray[MD-1:0];
   reg [MAW-1:0]    stim_addr;
   reg [1:0]        state;
   reg [MAW-1:0]    sample_count;
   reg [MAW-1:0]    stim_count;
   reg [MAW-1:0]    sample_wait_counter;
   reg [MAW-1:0]    wait_counter;

   reg [AW-1:0]     mem_addr_d2;
   reg [DW-1:0]     mem_data_d2;
   reg              mem_write_d2;
   reg              mem_ready_d2;

   // Read in stimulus
   integer     i,j;

   wire [CRW-1:0]  stype_masked;

   reg [255:0]     testfile;
   integer         fd;
   reg [128*8:0]   str;
   reg [MAW-1:0]   stim_end;
   wire            stall_random;

   /*AUTOWIRE*/

   // Read Stimulus
   initial begin
      $sformat(testfile[255:0],"%0s_%0d%s",NAME,INDEX,".emf");
      fd = $fopen(testfile, "r");
      if(!fd)
        begin
           $display("could not open the file %0s\n", testfile);
           $finish;
        end
      // Read stimulus from file
      j=0;
      while ($fgets(str, fd)) begin
         if ($sscanf(str,"%h", stimarray[j]))
           begin
              // $display("%0s %0d data=%0h", testfile, j, stimarray[j]);
              j=j+1;
           end
      end
      stim_end[MAW-1:0]=j;
   end

`define IDLE  2'b00 // ready to process stimarray
`define GO    2'b01 // processing stimarray
`define DONE  2'b10 // done processing stimarray

   // Delayed signals of interest
   always @ (posedge clk or negedge nreset)
     begin
        dut_cwait_d1 <= dut_cwait;
        is_sample_d2 <= is_sample_d1;
     end

   // Use to finish simulation
   assign cmds_done = ~dut_cwait_d1 & (state[1:0]==`DONE);

   assign is_sample_d1 = is_sample_p1;

   // Decode ctrl address and data from stimulus
   always @ (posedge clk or negedge nreset)
     if(!nreset)
       begin
          state[1:0]         <= `IDLE;
          wait_counter       <= 'b0;
          mem_ready_d1       <= 1'b0;
          mem_data_d1        <= 'd0;
          stim_count         <= 0;
          stim_addr[MAW-1:0] <= 'b0;
          is_sample_p1       <= 'b0;
       end
     else
       if(start & (state[1:0]==`IDLE)) // not started
         state[1:0] <= `GO;// going
       else if(~dut_cwait)
         if( (wait_counter[MAW-1:0]==0) & (stim_count < stim_end) & (state[1:0]==`GO) ) // going
           begin
              wait_counter[MAW-1:0] <= stimarray[stim_addr]; // first 15 bits
              mem_data_d1[SS-1:0]   <= stimarray[stim_addr];
              mem_ready_d1          <= 1'b1;
              stim_addr             <= stim_addr + 1'b1;
              stim_count            <= stim_count + 1'b1;
              is_sample_p1          <= ((stimarray[stim_addr][DS-1:MAW] & SAMPLE) == SAMPLE);
           end
         else if((wait_counter[MAW-1:0]==0) & (stim_count == stim_end) & (state[1:0]==`GO)) // not waiting and done
           begin
              state[1:0]            <= `DONE;// done
              mem_ready_d1          <= 1'b0;
           end
         else if(0<wait_counter)
           begin
              mem_ready_d1          <= 1'b0;
              wait_counter[MAW-1:0] <= wait_counter[MAW-1:0] - 1'b1;
           end

   // Removing delay value
   // assign ctrl_data[DW-1:0] = mem_data_d1[DW+16-1:
   always @ (posedge clk or negedge nreset)
     begin
        if(~nreset)
          begin
             mem_addr_d2  <= 'b0;
             mem_data_d2  <= 'b0;
             mem_write_d2 <= 'b0;
             mem_ready_d2 <= 'b0;
          end
        else if(~dut_cwait & ~is_sample_d1)
          begin
             mem_ready_d2 <= mem_ready_d1;
             mem_addr_d2  <= mem_data_d1[SS-1:AS];
             mem_data_d2  <= mem_data_d1[AS-1:DS];
             mem_write_d2 <= mem_data_d1[DS-1:MAW];
          end // if (~dut_cwait & ~is_sample)
        if(~nreset)
          begin
             ctrl_addr    <= 'b0;
             ctrl_data    <= 'b0;
             ctrl_write   <= 'b0;
             ctrl_ready   <= 'b0;
          end
        else if(~dut_cwait & ~is_sample_d2)
          begin
             ctrl_addr    <= mem_addr_d2;
             ctrl_data    <= mem_data_d2;
             ctrl_write   <= mem_write_d2;
             ctrl_ready   <= mem_ready_d2;
          end // if (~dut_cwait & ~is_sample)
        else if(~dut_cwait & is_sample_d2)
          ctrl_ready   <= 'b0;
     end

   // assign ctrl_data = dut_cwait ? ctrl_data_d2 : mem_data_d1[DW+16-1:16];
   // assign ctrl_ready = dut_cwait ? ctrl_ready_d2 : mem_ready_d1;
   // assign ctrl_ready = dut_cwait ? 1'b0 : mem_ready_d1;

   // TODO: Implement
   assign ctrl_wait = stall_random;

   // Random wait generator
   generate
      if(WAIT)
        begin
           reg [15:0] stall_counter;
           always @ (posedge clk or negedge nreset)
             if(!nreset)
               stall_counter[15:0] <= 'b0;
             else
               stall_counter[15:0] <= stall_counter+1'b1;
           assign stall_random      = (|stall_counter[6:0]);//(|wait_counter[3:0]);//1'b0;
        end
      else
        begin
           assign stall_random = 1'b0;
        end // else: !if(WAIT)
   endgenerate

   // Discover start and end of first batch of SAMPLE stimulus
   always @ (posedge clk or negedge nreset)
     if(~nreset)
       begin
          sample_start[MAW-1:0] <= 0;
          sample_end[MAW-1:0]   <= 0;
       end
     else
       if( is_sample_p1 & !sample_state )
         sample_end  <= stim_count;
       else if ( (sample_end <= sample_start) & ~is_sample_p1 )
         sample_start <= stim_count;

   // Decode sample data from stimulus
   always @ (posedge clk or negedge nreset)
     if(!nreset)
       begin
          sample_data[SW-1:0]   <= 0;
          sample_state          <= 0;
          sample_count[MAW-1:0] <= 0;
          sample_wait_counter[MAW-1:0] <= 0;
       end
     else
       if( (sample_wait_counter[MAW-1:0]==0) & (sample_count < sample_end) & (state[1:0]==`GO) )
         begin
            sample_wait_counter[MAW-1:0] <= stimarray[sample_count]; // first 15 bits
            sample_data[SW-1:0]     <= stimarray[sample_count][SW+SS-1:SS];
            sample_count            <= sample_count + 1'b1;
         end
       else if( (sample_wait_counter[MAW-1:0]==0) & (sample_count == sample_end) & (0 < sample_end) & (state[1:0]==`GO) )
         begin
            sample_wait_counter[MAW-1:0] <= stimarray[sample_start]; // first 15 bits
            sample_data[SW-1:0]     <= stimarray[sample_start][SW+SS-1:SS];
            sample_count            <= sample_start + 1'b1;
            sample_state            <= 1; // samples running
         end
       else if(sample_wait_counter>0)
         sample_wait_counter[MAW-1:0] <= sample_wait_counter[MAW-1:0] - 1'b1;

endmodule // dv_read_cmds

// Local Variables:
// verilog-library-directories:( "." )
// End:
