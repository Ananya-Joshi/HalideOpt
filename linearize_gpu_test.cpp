// g++ linearize_gpu_test.cpp -g -I ~/Halide10/include/ -I ~/Halide10/share/Halide/tools/ -L ~/Halide10/lib/ -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o linearize_gpu_test -std=c++11
// LD_LIBRARY_PATH=~/Halide10/lib/ ./linearize_gpu_test


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

class LinearizeMaskPipeline {
public:
    Func lin;
    Var x, y, c, x_outer, x_inner, y_outer, y_inner;
    Buffer<uint8_t> input;
    LinearizeMaskPipeline(Buffer<uint8_t> in) : input(in) {
        Func mask, less, greater;
        Expr value = input(x, y, c);

        Expr threshold = 0.0404482f;

        // Cast it to a floating point value.
        value = cast<float>(value);

        value = value / 255.0f;

        // linearize
        mask(x, y, c) = select(value <= threshold, 0, 1);
        less(x, y, c) = value / 12.92f;
        greater(x, y, c) = pow((value + 0.055f) / 1.055f, 2.4f);

        value = mask(x, y, c) * greater(x, y, c) + (1.0f - mask(x, y, c)) * less(x, y, c);
        

        value = value * 255.0f;
        value = min(value, 255.0f);

        value = cast<uint8_t>(value);

        lin(x, y, c) = value;
    }

    bool schedule_for_gpu() {
        Target target = find_gpu_target();
        if (!target.has_gpu_feature()) {
            return false;
        }

        Var x_outer, x_inner, y_outer, y_inner;
        lin.gpu_tile(x, y, x_outer, y_outer, x_inner, y_inner, 8, 8);

        lin.compile_jit(target);
    }
};

class LinearizeBranchPipeline {
public:
    Func lin;
    Var x, y, c, x_outer, x_inner, y_outer, y_inner;
    Buffer<uint8_t> input;
    LinearizeBranchPipeline(Buffer<uint8_t> in) : input(in) {
        
        Expr value = input(x, y, c);
        lin(x, y, c) = value; // just to initalize lin
        Expr ovalue = lin(x, y, c);

        Expr threshold = 0.0404482f;

        // Cast it to a floating point value.
        value = cast<float>(value);
        ovalue = cast<float>(ovalue);

        value = value / 255.0f;

        // linearize
        ovalue = select(value <= threshold, value / 12.92f, ovalue);
        ovalue = select(value > threshold, pow((value + 0.055f) / 1.055f, 2.4f), ovalue);
        

        ovalue = ovalue * 255.0f;
        ovalue = min(ovalue, 255.0f);

        ovalue = cast<uint8_t>(ovalue);

        lin(x, y, c) = ovalue;
    }

    bool schedule_for_gpu() {
        Target target = find_gpu_target();
        if (!target.has_gpu_feature()) {
            return false;
        }

        Var x_outer, x_inner, y_outer, y_inner;
        lin.gpu_tile(x, y, x_outer, y_outer, x_inner, y_inner, 8, 8);

        lin.compile_jit(target);
    }
};

void test_performance(Buffer<uint8_t> input, Func lin) {
    Buffer<uint8_t> output(input.width(), input.height(), input.channels());
    
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

    printf("%1.4f seconds\n", best_time);
}

int main(int argc, char **argv) {
    Buffer<uint8_t> input = load_image("images/rgb.png");

    LinearizeMaskPipeline lmp(input);
    printf("Mask pipeline:\n");
    test_performance(input, lmp.lin);

    LinearizeBranchPipeline lbp(input);
    printf("Branch pipeline:\n");
    test_performance(input, lbp.lin);
}