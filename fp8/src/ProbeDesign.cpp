#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

const int WIDTH_TYPE = 25;
const int WIDTH_RESULT = 85;

uint32_t hexStringToUint(const std::string& hexStr) {
    try {
        return std::stoul(hexStr, nullptr, 16);
    } catch (...) {
        return 0;
    }
}

float uintToFloat(uint32_t i) {
    union {
        uint32_t u;
        float f;
    } temp;
    temp.u = i;
    return temp.f;
}

std::vector<uint32_t> readFingerprint(const std::string& filepath) {
    std::vector<uint32_t> data;
    std::ifstream file(filepath);
    std::string line;
    if (!file.is_open()) return data;
    
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
            data.push_back(hexStringToUint(line));
        }
    }
    return data;
}

void printRow(const std::string& type, const std::string& result) {
    std::cout << "| " << std::left << std::setw(WIDTH_TYPE) << type 
              << "| " << std::left << std::setw(WIDTH_RESULT) << result << " |" << std::endl;
}

void printRowMultiLine(const std::string& type, const std::string& result) {
    std::stringstream ss(result);
    std::string line;
    bool first = true;
    while (std::getline(ss, line)) {
        if (first) {
            std::cout << "| " << std::left << std::setw(WIDTH_TYPE) << type 
                      << "| " << std::left << std::setw(WIDTH_RESULT) << line << " |" << std::endl;
            first = false;
        } else {
             std::cout << "| " << std::left << std::setw(WIDTH_TYPE) << " " 
                      << "| " << std::left << std::setw(WIDTH_RESULT) << line << " |" << std::endl;
        }
    }
}

void printSeparator() {
    std::cout << "+" << std::string(WIDTH_TYPE + 1, '-') 
              << "+" << std::string(WIDTH_RESULT + 2, '-') << "+" << std::endl;
}

void analyzeFile(const fs::path& targetPath) {
    std::vector<uint32_t> data = readFingerprint(targetPath.string());

    if (data.size() < 10) { 
        std::cerr << "Warning: Data insufficient in " << targetPath.filename() << std::endl;
        return;
    }

    std::string signedZero = (data[0] == 0x80000000) ? "-0" : ((data[0] == 0x00000000) ? "+0" : "Unknown");
    
    int totalWidth = WIDTH_TYPE + WIDTH_RESULT + 5;
    std::string filename = targetPath.filename().string();
    std::string title = " FP8 ANALYSIS REPORT: " + filename;
    if (title.length() > totalWidth) title = title.substr(0, totalWidth);
    
    int padding = (totalWidth - title.length()) / 2;
    if (padding < 0) padding = 0;
    
    std::cout << std::endl;
    std::cout << std::string(totalWidth, '=') << std::endl;
    std::cout << std::string(padding, ' ') << title << std::endl;
    std::cout << std::string(totalWidth, '=') << std::endl;
    
    printSeparator();
    printRow("PROBE TYPE", "RESULT FEEDBACK");
    printSeparator();
    
    printRow("Signed Zero", signedZero);
    printRow("Data Points", std::to_string(data.size()));
    printRow("Precision inferred", filename.find("e5m2") != std::string::npos ? "FP8 E5M2" : "FP8 E4M3");
    
    // Add more analysis logic here as needed
    
    printSeparator();
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::string targetDir = "../numeric_fingerprints";
    
    if (!fs::exists(targetDir)) {
        std::cerr << "Error: Directory not found: " << targetDir << std::endl;
        return 1;
    }

    bool found = false;
    for (const auto& entry : fs::directory_iterator(targetDir)) {
        if (entry.is_regular_file()) {
            std::string fname = entry.path().filename().string();
            // Process files ending with _wmma_output.txt
            if (fname.find("_wmma_output.txt") != std::string::npos) {
                analyzeFile(entry.path());
                found = true;
            }
        }
    }

    if (!found) {
        std::cerr << "No scan output files found in " << targetDir << std::endl;
        return 1;
    }

    return 0;
}
