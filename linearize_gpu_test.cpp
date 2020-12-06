// g++ linearize_gpu_test.cpp -g -I ~/Halide10/include/ -I ~/Halide10/share/Halide/tools/ -L ~/Halide10/lib/ -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o linearize_gpu_test -std=c++11
// LD_LIBRARY_PATH=~/Halide10/lib/ ./linearize_gpu_test


#include "Halide.h"

#include <chrono>
#include <vector>

// Include some support code for loading pngs.
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;
using namespace std::chrono;

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
        features_to_try.push_back(Target::CUDA);
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
        //mask(x, y, c) = select(value <= threshold, 0, 1);
        less(x, y, c) = value / 12.92f;
        greater(x, y, c) = pow((value + 0.055f) / 1.055f, 2.4f);

        value = select(value <= threshold, less(x, y, c), greater(x, y, c));
        

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

        Expr threshold = 0.0404482f;

        // Cast it to a floating point value.
        value = cast<float>(value);

        value = value / 255.0f;

        // linearize
        Expr ovalue = select(value <= threshold, value / 12.92f, pow((value + 0.055f) / 1.055f, 2.4f));
        

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

#include <fstream>
#include <vector>
#include <string>
void test_performance(Buffer<uint8_t> input, Func lin, std::string oname) {
    Buffer<uint8_t> output(input.width(), input.height(), input.channels());
    
    int n = 1000;
    lin.realize(output);
    std::vector<double> v;
    for (int i = 0; i < n; i++) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        lin.realize(output);
        output.copy_to_host();
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        double avg_time = time_span.count();

        v.push_back(avg_time);
    }
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    printf("Mean: %1.6f seconds\n", mean);
    printf("Std dev: %1.6f seconds\n", stdev);
    printf("N: %d\n", n);

    std::ofstream outFile(oname);
    // the important part
    for (const auto &e : v) outFile << e << "\n";
}

int main(int argc, char **argv) {
    Buffer<uint8_t> input = load_image("images/rgb.png");

    printf("CPU:\n");
    LinearizeBranchPipeline cpu_lbp(input);
    printf("Branch pipeline avg runtime (3000x):\n");
    test_performance(input, cpu_lbp.lin, "cpu_branch.txt");

    LinearizeMaskPipeline cpu_lmp(input);
    printf("Branch-free pipeline avg runtime (3000x):\n");
    test_performance(input, cpu_lmp.lin, "cpu_pred.txt");

    printf("\nGPU:\n");
    LinearizeBranchPipeline gpu_lbp(input);
    gpu_lbp.schedule_for_gpu();
    printf("Branch pipeline avg runtime (3000x):\n");
    test_performance(input, gpu_lbp.lin, "gpu_branch.txt");
    
    LinearizeMaskPipeline gpu_lmp(input);
    gpu_lmp.schedule_for_gpu();
    printf("Branch-free pipeline avg runtime (3000x):\n");
    test_performance(input, gpu_lmp.lin, "gpu_pred.txt");
}