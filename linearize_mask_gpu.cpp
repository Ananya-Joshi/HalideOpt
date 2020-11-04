// Halide tutorial lesson 2: Processing images

// This lesson demonstrates how to pass in input images and manipulate
// them.

// On linux, you can compile and run it like so:
// g++ linearize_mask_gpu.cpp -g -I ~/Halide10/include/ -I ~/Halide10/share/Halide/tools/ -L ~/Halide10/lib/ -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o linearize_mask_gpu -std=c++11
// LD_LIBRARY_PATH=~/Halide10/lib/ ./linearize_mask_gpu


// The only Halide header file you need is Halide.h. It includes all of Halide.
#include "Halide.h"

#include <chrono>

// Include some support code for loading pngs.
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;

Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Target target = get_host_target();

    std::vector<Target::Feature> features_to_try;
    if (target.os == Target::Windows) {
        // Try D3D12 first; if that fails, try OpenCL.
        if (sizeof(void*) == 8) {
            // D3D12Compute support is only available on 64-bit systems at present.
            features_to_try.push_back(Target::D3D12Compute);
        }
        features_to_try.push_back(Target::OpenCL);
    } else if (target.os == Target::OSX) {
        // OS X doesn't update its OpenCL drivers, so they tend to be broken.
        // CUDA would also be a fine choice on machines with NVidia GPUs.
        features_to_try.push_back(Target::Metal);
    } else {
        features_to_try.push_back(Target::OpenCL);
    }
    // Uncomment the following lines to also try CUDA:
    // features_to_try.push_back(Target::CUDA);

    for (Target::Feature f : features_to_try) {
        Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }

    printf("Requested GPU(s) are not supported. (Do you have the proper hardware and/or driver installed?)\n");
    return target;
}

int main(int argc, char **argv) {

    Target target = find_gpu_target();
    if (!target.has_gpu_feature()) {
        return false;
    }
    
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


    Var x_outer, x_inner, y_outer, y_inner;
    lin.gpu_tile(x, y, x_outer, y_outer, x_inner, y_inner, 8, 8);

    lin.compile_jit(target);

    Halide::Buffer<uint8_t> output(input.width(), input.height(), input.channels());
    
    lin.realize(output);

    using namespace std::chrono;

    double best_time = 0.0;
    for (int i = 0; i < 3; i++) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        for (int i = 0; i < 1000; i++) {
            lin.realize(output);
            output.copy_to_host();
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        double elapsed = time_span.count() / 1000.0;
        if (i == 0 || elapsed < best_time) {
            best_time = elapsed;
        }
    }

    save_image(output, "linearized_mask.png");

    printf("%1.4f seconds\n", best_time);
    return 0;
}

