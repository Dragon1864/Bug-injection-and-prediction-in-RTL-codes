
`timescale 1ns/1ps
module v8cpu_alu_tb;
    reg [7:0] a, b;
    reg [3:0] op;
    wire [7:0] c;
    wire [7:0] newFlags;
    v8cpu_alu uut (.a(a), .b(b), .op(op), .c(c), .flags(8'b0), .newFlags(newFlags));
    integer outfile;
    initial begin
        outfile = $fopen("simulation_output.log", "w");

        op = 4'd0;
        a = 8'd12;
        b = 8'd24;
        #10;
        $fwrite(outfile, "%d,%d,%d,%d,%d\n", op, a, b, c, newFlags);

        op = 4'd1;
        a = 8'd45;
        b = 8'd23;
        #10;
        $fwrite(outfile, "%d,%d,%d,%d,%d\n", op, a, b, c, newFlags);

        op = 4'd2;
        a = 8'd200;
        b = 8'd100;
        #10;
        $fwrite(outfile, "%d,%d,%d,%d,%d\n", op, a, b, c, newFlags);

        op = 4'd3;
        a = 8'd50;
        b = 8'd50;
        #10;
        $fwrite(outfile, "%d,%d,%d,%d,%d\n", op, a, b, c, newFlags);

        op = 4'd4;
        a = 8'd75;
        b = 8'd125;
        #10;
        $fwrite(outfile, "%d,%d,%d,%d,%d\n", op, a, b, c, newFlags);

        $fclose(outfile);
        $finish;
    end
endmodule
