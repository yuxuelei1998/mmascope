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

void analyzeFile(const fs::path& targetPath, const std::string& hardwareNameArg) {
    std::vector<uint32_t> data = readFingerprint(targetPath.string());

    if (data.size() < 103) {
        std::cerr << "Warning: Data insufficient in " << targetPath.filename() << std::endl;
        return;
    }

    std::string filename = targetPath.filename().string();
    std::cout << "\nAnalyzing: " << filename << std::endl;

    std::string precisionType = "FP8 (Unknown)";
    std::string precisionSuffix = "_FP8_Unknown";
    if (filename.find("e5m2") != std::string::npos) {
        precisionType = "FP8 E5M2";
        precisionSuffix = "_FP8_E5M2";
    } else if (filename.find("e4m3") != std::string::npos) {
        precisionType = "FP8 E4M3";
        precisionSuffix = "_FP8_E4M3";
    }

    std::string signedZero = (data[0] == 0x80000000) ? "-0" : ((data[0] == 0x00000000) ? "+0" : "Unknown");
    
    std::string nanInf;
    bool allSameNaN = true;
    for (int i = 1; i <= 19; ++i) if (data[i] != data[1]) allSameNaN = false;
    if (allSameNaN) {
        std::stringstream ss; ss << "Fixed NaN: 0x" << std::hex << data[1]; nanInf = ss.str();
    } else nanInf = "Propagates NaN Payload";

    bool subnormalSupported = false;
    for (int i = 20; i <= 52; ++i) if (data[i] != 0) subnormalSupported = true;
    std::string subnormal = subnormalSupported ? "Supported" : "Not Supported (Flushed to Zero)";

    std::string roundingMode = "Unknown";
    uint32_t r1 = data[53], r2 = data[54], r3 = data[103];
    if (r1 == 0x3f800001 && r2 == 0xbf800001  && r3 == 0x40000000) roundingMode = "Truncation (TC-Trunc)";
    else if (r1 == 0x3f800001 && r2 == 0xbf800001  && r3 == 0x3FFFFFFF) roundingMode = "Round to Zero (RTZ)";
    else if (r1 == 0x3f800001 && r2 == 0xbf800002) roundingMode = "Round to Negative Infinity (RTN)";
    else if (r1 == 0x3f800002 && r2 == 0xbf800001) roundingMode = "Round to Positive Infinity (RTP)";
    else if (r1 == 0x3f800002 && r2 == 0xbf800002) roundingMode = "Round to Nearest Even (RNE)";

    bool hasOrder = false;
    for (int i = 55; i <= 87; ++i) if (data[i] != data[55]) hasOrder = true;
    std::string accumOrder = hasOrder ? "Has Accumulation Order" : "No Accumulation Order";

    int groups = 1;
    std::vector<uint32_t> gVals;
    gVals.push_back(data[55]);
    uint32_t cur = data[56];
    for (int i = 56; i <= 87; ++i) {
        if (data[i] != cur) { groups++; cur = data[i]; gVals.push_back(cur); }
    }
    
    int dpWidth = 32 / groups;
    bool isSeq = true;
    for (size_t i = 0; i < gVals.size() - 1; ++i) if (gVals[i] <= gVals[i+1]) isSeq = false;
    bool isButter = (groups > 1 && groups % 2 == 0);
    if (isButter) {
        int half = groups / 2;
        for (int i = 1; i < half; ++i) {
            if (gVals[i] != gVals[i + half]) {
                isButter = false;
                break;
            }
        }
    }
    
    std::string normType = "Complex/Unknown";
    if (!hasOrder) normType = "Single Group";
    else if (isSeq) normType = "Sequential Grouping (" + std::to_string(groups) + " groups)";
    else if (isButter) normType = "Butterfly Grouping (" + std::to_string(groups) + " groups)";

    std::stringstream normSS; 
    normSS << (2 * groups - 1) << " Stages, " << normType;
    std::string normalization = normSS.str();

    int precBits = 0;
    int startIdx = (roundingMode.find("Nearest") != std::string::npos) ? 87 : 91;
    for (int i = startIdx; i < startIdx + 4; ++i) if (data[i] == 0x4e800002) precBits++;
    
    std::string monotonic = "Satisfies Monotonicity";
    for (int i = 95; i < 102; i += 2) {
        if (uintToFloat(data[i]) > uintToFloat(data[i+1])) { monotonic = "Non-Monotonic"; break; }
    }

    std::string internalStructure;
    if (groups == 4 && dpWidth == 4 && normType == "Butterfly Grouping (4 groups)") {
        internalStructure = 
            "4-Group Butterfly (Width 4)\n"
            "pd[00-01, 04-05] pd[02-03, 06-07] pd[08-09, 12-13] pd[10-11, 14-15]\n"
            "        |                |                |                |\n"
            "C --+->(+)-----------+->(+)-----------+->(+)-----------+->(+)----> D";
    } else if (groups == 2 && dpWidth == 8 && normType == "Butterfly Grouping (2 groups)") {
        internalStructure = 
            "2-Group Butterfly (Width 8)\n"
            "pd[00-03, 08-11] pd[04-07, 12-15]\n"
            "        |                |\n"
            "C --+->(+)-----------+->(+)----> D";
    } else if (groups == 8 && dpWidth == 2 && normType == "Sequential Grouping (8 groups)") {
        internalStructure = 
            "8-Group Sequential (Width 2)\n"
            "   pd[00-01] pd[02-03] pd[04-05] pd[06-07] pd[08-09] pd[10-11] pd[12-13] pd[14-15]\n"
            "        |         |         |         |         |         |         |         |\n"
            "C --+->(+)----+->(+)----+->(+)----+->(+)----+->(+)----+->(+)----+->(+)----+->(+)----> D";
    } else if (groups == 4 && dpWidth == 4 && normType == "Sequential Grouping (4 groups)") {
        internalStructure = 
            "4-Group Sequential (Width 4)\n"
            "   pd[00-03] pd[04-07] pd[08-11] pd[12-15]\n"
            "        |         |         |         |\n"
            "C --+->(+)----+->(+)----+->(+)----+->(+)----> D";
    } else if (groups == 2 && dpWidth == 8 && normType == "Sequential Grouping (2 groups)") {
        internalStructure = 
            "2-Group Sequential (Width 8)\n"
            "   pd[00-07] pd[08-15]\n"
            "        |         |\n"
            "C --+->(+)----+->(+)----> D";
    } else if (groups == 1 && dpWidth == 16 && normType == "Single Group") {
        internalStructure = 
            "Single-Step Accumulation (Width 16)\n"
            "   pd[00-15]\n"
            "        |\n"
            "C --+->(+)----> D";
    } else {
        std::stringstream structSS;
        structSS << "RM: " << roundingMode.substr(0, std::min((size_t)21, roundingMode.length()))
                 << ((roundingMode.length() > 21) ? "..." : "")
                 << " | Acc: " << (hasOrder ? "Ordered" : "No Order")
                 << " | DP Width: " << dpWidth
                 << " | Extra Bits: " << precBits;
        internalStructure = structSS.str();
    }

    // Hardware Matching Logic
    std::string matchResult = "No exact match found.";
    bool matchFound = false;
    fs::path targetDir = targetPath.parent_path();
    
    if (fs::exists(targetDir)) {
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            if (entry.is_regular_file()) {
                if (entry.path().filename() == targetPath.filename()) continue;
                if (entry.path().filename().string().find("_wmma_output") != std::string::npos) continue;
                if (entry.path().extension() != ".txt") continue;

                std::vector<uint32_t> other = readFingerprint(entry.path().string());
                if (other.size() == data.size() && std::equal(data.begin(), data.end(), other.begin())) {
                    matchResult = "Matches Hardware: " + entry.path().stem().string();
                    matchFound = true;
                    break;
                }
            }
        }
    }

    if (!matchFound && !hardwareNameArg.empty() && hardwareNameArg != "Unknown_Hardware") {
        std::string newFileName = hardwareNameArg + precisionSuffix + ".txt";
        fs::path newFilePath = targetDir / newFileName;
        
        std::ofstream outFile(newFilePath);
        if (outFile.is_open()) {
            for (const auto& val : data) {
                outFile << "0x" << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
            }
            matchResult = "New fingerprint saved: " + newFileName;
        } else {
            std::cerr << "Error: Could not save new fingerprint to " << newFilePath << std::endl;
        }
    }

    int totalWidth = WIDTH_TYPE + WIDTH_RESULT + 5;
    std::string title = " NUMERIC PROBE ANALYSIS REPORT ";
    int padding = (totalWidth - title.length()) / 2;
    std::cout << std::endl;
    std::cout << std::string(totalWidth, '=') << std::endl;
    std::cout << std::string(padding, ' ') << title << std::endl;
    std::cout << std::string(totalWidth, '=') << std::endl;
    
    printSeparator();
    printRow("PROBE TYPE", "RESULT FEEDBACK");
    printSeparator();
    
    printRow("Precision inferred", precisionType);
    printRow("Signed Zero", signedZero);
    printRow("NaN & INF", nanInf);
    printRow("Subnormal Support", subnormal);
    printRow("Rounding Mode", roundingMode);
    printRow("Accumulation Order", accumOrder);
    printRow("Dot Product Unit Width", std::to_string(dpWidth));
    printRow("Extra Precision Bits", std::to_string(precBits));
    printRow("Normalization", normalization);
    printRow("Monotonicity", monotonic);
    printRowMultiLine("Internal Data Path", internalStructure);

    printSeparator();
    printRow("HARDWARE IDENTIFICATION", matchResult);
    printSeparator();
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::string targetDir = "../numeric_fingerprints";
    
    std::string hardwareName = "Unknown_Hardware";
    if (argc > 1) {
        hardwareName = argv[1];
    }
    
    bool found = false;
    if (fs::exists(targetDir)) {
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            if (entry.is_regular_file()) {
                std::string fname = entry.path().filename().string();
                if (fname.find("_wmma_output.txt") != std::string::npos || fname.find("fp8") != std::string::npos) {
                     analyzeFile(entry.path(), hardwareName);
                     found = true;
                }
            }
        }
    }

    if (!found) {
        std::cerr << "No scan output files found in " << targetDir << std::endl;
        return 1;
    }

    return 0;
}