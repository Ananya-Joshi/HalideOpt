// Halide tutorial lesson 2: Processing images

// This lesson demonstrates how to pass in input images and manipulate
// them.

// On linux, you can compile and run it like so:
// g++ linearize_branch.cpp -g -I ~/Halide10/include/ -I ~/Halide10/share/Halide/tools/ -L ~/Halide10/lib/ -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o linearize_branch -std=c++11
// LD_LIBRARY_PATH=~/Halide10/lib/ ./linearize_branch


// The only Halide header file you need is Halide.h. It includes all of Halide.
#include "Halide.h"

// Include some support code for loading pngs.
#include "halide_image_io.h"

using namespace Halide::Tools;

int main(int argc, char **argv) {

    Halide::Buffer<uint8_t> input = load_image("images/rgb.png");

    Halide::Func lin;

    Halide::Var x, y, c;

    Halide::Expr value = input(x, y, c);
    lin(x, y, c) = value; // just to initalize lin
    Halide::Expr ovalue = lin(x, y, c);

    Halide::Expr threshold = 0.0404482f;

    // Cast it to a floating point value.
    value = Halide::cast<float>(value);
    ovalue = Halide::cast<float>(ovalue);

    value = value / 255.0f;

    // linearize
    ovalue = select(value <= threshold, value / 12.92f, ovalue);
    ovalue = select(value > threshold, pow((value + 0.055f) / 1.055f, 2.4f), ovalue);
    

    ovalue = ovalue * 255.0f;
    ovalue = Halide::min(ovalue, 255.0f);

    ovalue = Halide::cast<uint8_t>(ovalue);

    lin(x, y, c) = ovalue;


    Halide::Buffer<uint8_t> output =
        lin.realize(input.width(), input.height(), input.channels());

    save_image(output, "linearized_branch.png");


    printf("Success!\n");
    return 0;
}
