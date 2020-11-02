// Halide tutorial lesson 2: Processing images

// This lesson demonstrates how to pass in input images and manipulate
// them.

// On linux, you can compile and run it like so:
// g++ linearize_mask.cpp -g -I ~/Halide10/include/ -I ~/Halide10/share/Halide/tools/ -L ~/Halide10/lib/ -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o linearize_mask -std=c++11
// LD_LIBRARY_PATH=~/Halide10/lib/ ./linearize_mask


// The only Halide header file you need is Halide.h. It includes all of Halide.
#include "Halide.h"

// Include some support code for loading pngs.
#include "halide_image_io.h"

using namespace Halide::Tools;

int main(int argc, char **argv) {

    Halide::Buffer<uint8_t> input = load_image("images/rgb.png");

    Halide::Func lin;
    Halide::Func less;
    Halide::Func greater;
    Halide::Func mask;

    Halide::Var x, y, c;

    Halide::Expr value = input(x, y, c);

    Halide::Expr threshold = 0.0404482f;

    // Cast it to a floating point value.
    value = Halide::cast<float>(value);

    value = value / 255.0f;

    // linearize
    mask(x, y, c) = select(value <= threshold, 0, 1);
    less(x, y, c) = value / 12.92f;
    greater(x, y, c) = pow((value + 0.055f) / 1.055f, 2.4f);

    value = mask(x, y, c) * greater(x, y, c) + (1.0f - mask(x, y, c)) * less(x, y, c);
    

    value = value * 255.0f;
    value = Halide::min(value, 255.0f);

    value = Halide::cast<uint8_t>(value);

    lin(x, y, c) = value;


    Halide::Buffer<uint8_t> output =
        lin.realize(input.width(), input.height(), input.channels());

    save_image(output, "linearized_mask.png");


    printf("Success!\n");
    return 0;
}