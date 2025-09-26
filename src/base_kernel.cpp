#include "base_kernel.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

void BaseKernel::compile(const std::string& output_name)
{
    auto module = build_hlo();
    if (!module)
    {
        std::cerr << "Failed to build HLO for kernel: " << name_
                  << std::endl;
        return;
    }

    write_hlo_and_compile_neff(output_name);
}

void BaseKernel::write_hlo_and_compile_neff(const std::string& output_name)
{
    auto module = build_hlo();

    // Create output directories
    std::filesystem::create_directories("output/generic/hlo");
    std::filesystem::create_directories("output/trn1/neff");
    std::filesystem::create_directories("output/inf2/neff");

    // Write HLO file
    std::string hlo_filename = "output/generic/hlo/" + output_name + ".hlo";
    std::ofstream hlo_file(hlo_filename);
    hlo_file << module->ToString();
    hlo_file.close();

    // Compile to NEFF using neuronx-cc
    std::string neff_filename = "output/trn1/neff/" + output_name + ".neff";
    std::string compile_cmd = "neuronx-cc compile " + hlo_filename +
        " --target trn1 -o " + neff_filename;

    std::cout << "Compiling: " << compile_cmd << std::endl;
    int result_code = system(compile_cmd.c_str());

    if (result_code == 0)
    {
        std::cout << "✓ Generated: " << hlo_filename << " and "
                  << neff_filename << std::endl;
    }
    else
    {
        std::cerr << "✗ NEFF compilation failed for " << output_name
                  << std::endl;
        std::cout << "✓ HLO generated: " << hlo_filename << std::endl;
    }
}
