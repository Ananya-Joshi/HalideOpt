// g++ conv_test.cpp -g -I ~/arch/Halide/distrib/include/ -I ~/arch/Halide/distrib/tools/ -L ~/arch/Halide/distrib/lib/ -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o conv_test -std=c++11
// LD_LIBRARY_PATH=~/arch/Halide/distrib/lib/ ./conv_test images/rgb.png

// g++ pixel_test.cpp -g -I ~/Halide10/include/ -I ~/Halide10/share/Halide/tools/ -L ~/Halide10/lib/ -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o pixel_test -std=c++11
// LD_LIBRARY_PATH=~/Halide10/lib/ ./pixel_test images/rgb.png

#include "Halide.h"

#include <chrono>
#include <vector>

// Include some support code for loading pngs.
#include "halide_image_io.h"
// #include <filesystem>
#include <string>
#include <iostream>


// namespace fs = std::__fs::filesystem;


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

class PixelMaskPipeline {
public:
    Func lin;
    Var x, y, c, x_outer, x_inner, y_outer, y_inner;
    Buffer<uint8_t> input;
    PixelMaskPipeline(Buffer<uint8_t> in) : input(in) {
        Func mask, less, greater;
        Expr value = input(x, y, c);

        Expr threshold = 0.5f;

        // Cast it to a floating point value.
        value = cast<float>(value);

        value = value / 255.0f;

        Func inpb = BoundaryConditions::repeat_edge(input);

        mask(x, y, c) = cast<float>(value > threshold);//select(value <= threshold, 0, 1);
        less(x, y, c) = input(x, y, c) * 5.0f - 2.0f;
        greater(x, y, c) = input(x, y, c) / 5.0f - 2.0f;

        lin(x, y, c) = cast<uint8_t>(min(mask(x, y, c) * greater(x, y, c) + (1.0f - mask(x, y, c)) * less(x, y, c), 255.0f));
    }

    bool schedule_for_cpu() {
        // lin.reorder(c, x, y)
        //     .bound(c, 0, 3)
        //     .unroll(c);
        // Var x_outer, x_inner, y_outer, y_inner;
        // lin.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
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

class PixelBranchPipeline {
public:
    Func lin;
    Var x, y, c, x_outer, x_inner, y_outer, y_inner;
    Buffer<uint8_t> input;
    PixelBranchPipeline(Buffer<uint8_t> in) : input(in) {
        
        Expr value = input(x, y, c);

        Expr threshold = 0.5f;

        // Cast it to a floating point value.
        value = cast<float>(value);

        value = value / 255.0f;


        lin(x, y, c) = cast<uint8_t>(min(select(value <= threshold, input(x, y, c) * 5.0f + 2.0f, input(x, y, c) / 5.0f - 2.0f), 255.0f));
    }

    bool schedule_for_cpu() {
        // lin.reorder(c, x, y)
        //     .bound(c, 0, 3)
        //     .unroll(c);
        // Var x_outer, x_inner, y_outer, y_inner;
        // lin.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
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
    lin.realize(output);
    output.copy_to_host();
    save_image(output, oname + "_pixel.png");
    //lin.compile_to_lowered_stmt(oname + "_pixel.html", lin.infer_arguments(), HTML);
    
    // warmup
    int n = 1000;
    for (int i = 0; i < n; i++) {
        lin.realize(output);
        output.copy_to_host();
    }
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

    if (oname != "") {
        std::ofstream outFile(oname + "_pixel.txt");
        // the important part
        for (const auto &e : v) outFile << e << "\n";
    }
}

void test_performance(Buffer<uint8_t> input, Func lin) {
    test_performance(input, lin, "");
}


int main(int argc, char **argv) {
    if (argc > 1) {
        Buffer<uint8_t> input = load_image(argv[1]);
        printf("CPU:\n");

        PixelBranchPipeline cpu_lbp(input);
        cpu_lbp.schedule_for_cpu();
        printf("Branch pipeline avg runtime (1000x):\n");
        test_performance(input, cpu_lbp.lin, "renders/cpu_branch");

        PixelMaskPipeline cpu_lmp(input);
        cpu_lmp.schedule_for_cpu();
        printf("Branch-free pipeline avg runtime (1000x):\n");
        test_performance(input, cpu_lmp.lin, "renders/cpu_pred");

        printf("\nGPU:\n");

        PixelBranchPipeline gpu_lbp(input);
        gpu_lbp.schedule_for_gpu();
        printf("Branch pipeline avg runtime (1000x):\n");
        test_performance(input, gpu_lbp.lin, "renders/gpu_branch");

        PixelMaskPipeline gpu_lmp(input);
        gpu_lmp.schedule_for_gpu();
        printf("Branch-free pipeline avg runtime (1000x):\n");
        test_performance(input, gpu_lmp.lin, "renders/gpu_pred");
    }
}