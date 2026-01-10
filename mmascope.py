import os
import subprocess
import sys
import platform

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

if platform.system() == "Windows":
    os.system('color')

def print_step(msg):
    print(f"\n{Colors.BLUE}{Colors.BOLD}[*] {msg}{Colors.ENDC}")

def print_success(msg):
    print(f"{Colors.GREEN}[+] {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}[!] {msg}{Colors.ENDC}")

def get_gpu_architecture(gpu_name):
    name = gpu_name.upper()
    if "RTX 50" in name or "BLACKWELL" in name or "B100" in name or "B200" in name or "GB200" in name:
        return "sm_120"
    if "H100" in name or "H800" in name or "GH200" in name or "HOPPER" in name:
        return "sm_90"
    if "RTX 40" in name or "L40" in name or "L4" in name or "ADA" in name:
        return "sm_89"
    if "RTX 30" in name or "A10" in name or "A40" in name or "A30" in name or "A16" in name:
        return "sm_86"
    if "A100" in name or "A800" in name:
        return "sm_80"
    if "RTX 20" in name or "TITAN RTX" in name or "T4" in name or "QUADRO RTX" in name:
        return "sm_75"
    if "V100" in name or "TITAN V" in name:
        return "sm_70"
    if "P100" in name: return "sm_60"
    if "GTX 10" in name: return "sm_61"
    return None

def get_amd_gpu_architecture(gpu_name):
    name = gpu_name.upper()
    if "MI300" in name: return "gfx942"
    if "MI250" in name or "MI210" in name or "MI200" in name: return "gfx90a"
    if "MI100" in name: return "gfx908"
    if "7900" in name: return "gfx1100"
    if "7800" in name: return "gfx1101"
    if "6900" in name or "6800" in name: return "gfx1030"
    return None

def detect_gpus():
    gpus = []
    # Try NVIDIA
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            encoding='utf-8'
        )
        gpus.extend([line.strip() for line in result.strip().split('\n') if line.strip()])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Try AMD
    if not gpus:
        try:
            # rocm-smi might not output simple names, but let's try a basic check
            # or check if hipconfig exists to assume AMD environment
            subprocess.check_output(["rocm-smi"], stderr=subprocess.STDOUT)
            # If rocm-smi runs, let's assume we can try to find the name or just return a generic AMD placeholder
            # Better: try to get name via rocm-smi --showproductname
            try:
                result = subprocess.check_output(
                    ["rocm-smi", "--showproductname"], 
                    encoding='utf-8'
                )
                # Parse output like "GPU[0] : Card series: Instinc MI210"
                for line in result.split('\n'):
                    if "Series:" in line or "product name" in line.lower():
                        parts = line.split(':')
                        if len(parts) > 1:
                            gpus.append(parts[-1].strip())
            except:
                gpus.append("AMD GPU (Manual Arch Required)")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
            
    return gpus

def run_project():
    print(f"{Colors.HEADER}{'='*60}")
    print(f"      MMAScope: MMA Microarchitecture Probe")
    print(f"{'='*60}{Colors.ENDC}")
    print_step("Detecting Hardware...")
    gpus = detect_gpus()
    
    selected_gpu = ""
    arch_flag = ""

    if not gpus:
        print_error("No NVIDIA GPUs detected.")
        manual_arch = input(f"{Colors.WARNING}Please enter architecture flag manually (e.g., sm_90): {Colors.ENDC}")
        arch_flag = manual_arch.strip()
        selected_gpu = "Manual Selection"
    else:
        if len(gpus) == 1:
            selected_gpu = gpus[0]
            print(f"Detected: {Colors.BOLD}{selected_gpu}{Colors.ENDC}")
        else:
            print("Multiple GPUs detected:")
            for i, gpu in enumerate(gpus):
                print(f"  [{i+1}] {gpu}")
            choice = input("Select GPU index (default 1): ")
            idx = int(choice) - 1 if choice.isdigit() else 0
            if 0 <= idx < len(gpus):
                selected_gpu = gpus[idx]
            else:
                selected_gpu = gpus[0]
        
        # Try NVIDIA detection first
        arch_flag = get_gpu_architecture(selected_gpu)
        is_amd = False
        
        # If not NVIDIA, try AMD detection
        if not arch_flag:
            amd_arch = get_amd_gpu_architecture(selected_gpu)
            if amd_arch or "AMD" in selected_gpu or "MI" in selected_gpu:
                arch_flag = amd_arch
                is_amd = True

        if not arch_flag:
            print(f"{Colors.WARNING}Could not auto-detect architecture for {selected_gpu}.{Colors.ENDC}")
            arch_flag = input("Please enter architecture flag (e.g., sm_90 or gfx90a): ").strip()
            if "gfx" in arch_flag:
                is_amd = True

    print_success(f"Target Hardware: {selected_gpu} | Arch: {arch_flag} | Vendor: {'AMD' if is_amd else 'NVIDIA'}")

    safe_gpu_name = selected_gpu.replace(" ", "_").replace("/", "-").replace(":", "").replace("\\", "")

    print_step("Select Precision")
    print("  [1] fp16")
    print("  [2] bf16")
    print("  [3] fp8")
    p_choice = input("Enter choice (1, 2 or 3): ").strip()
    
    precision = "fp16"
    if p_choice == "2" or p_choice.lower() == "bf16":
        precision = "bf16"
    elif p_choice == "3" or p_choice.lower() == "fp8":
        precision = "fp8"
    
    print_success(f"Selected Precision: {Colors.BOLD}{precision}{Colors.ENDC}")

    root_dir = os.getcwd()
    base_dir = os.path.join(root_dir, precision)
    src_dir = os.path.join(base_dir, "src")
    lib_dir = os.path.join(base_dir, "lib")
    
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
        print_success(f"Created directory: {lib_dir}")
    is_windows = platform.system() == "Windows"
    exe_ext = ".exe" if is_windows else ""
    
    cuda_src_file = f"{precision}_dp16a_wmma.cu" if not is_amd else f"{precision}_mfma.hip"
    cuda_exe_name = f"{precision}_dp16a_wmma{exe_ext}" if not is_amd else f"{precision}_mfma{exe_ext}"
    
    cpp_src_file = "ProbeDesign.cpp"
    cpp_exe_name = f"probe_analysis{exe_ext}"

    print_step(f"Compiling Kernel ({cuda_src_file})...")
    
    cuda_input = os.path.join(src_dir, cuda_src_file)
    cuda_output = os.path.join(lib_dir, cuda_exe_name)
    
    if is_amd:
        # HIP Compilation
        compiler_cmd = [
            "hipcc", 
            f"--offload-arch={arch_flag}", 
            "-std=c++17", 
            "-o", cuda_output, 
            cuda_input,
            "-I/opt/rocm/include", # Common include path
            "-Wno-deprecated-declarations" # Suppress warnings if any
        ]
    else:
        # NVCC Compilation
        compiler_cmd = [
            "nvcc", 
            f"-arch={arch_flag}", 
            "-std=c++17", 
            "-o", cuda_output, 
            cuda_input
        ]
    
    print(f"Executing: {' '.join(compiler_cmd)}")
    try:
        subprocess.run(compiler_cmd, check=True)
        print_success("Kernel compilation successful.")
    except subprocess.CalledProcessError:
        print_error("Kernel compilation failed.")
        sys.exit(1)
    except FileNotFoundError:
        print_error(f"{compiler_cmd[0]} not found. Check your PATH.")
        sys.exit(1)

    print_step(f"Compiling Probe Analysis ({cpp_src_file})...")
    
    cpp_input = os.path.join(src_dir, cpp_src_file)
    cpp_output = os.path.join(lib_dir, cpp_exe_name)
    
    gpp_cmd = [
        "g++", 
        "-o", cpp_output, 
        cpp_input, 
        "-std=c++17"
    ]
    
    print(f"Executing: {' '.join(gpp_cmd)}")
    try:
        subprocess.run(gpp_cmd, check=True)
        print_success("C++ compilation successful.")
    except subprocess.CalledProcessError:
        print_error("C++ compilation failed.")
        sys.exit(1)
    except FileNotFoundError:
        print_error("g++ not found. Check your MinGW/GCC path.")
        sys.exit(1)

    print_step("Running Analysis...")
    
    os.chdir(lib_dir)
    print(f"Changed working directory to: {os.getcwd()}")

    print(f"\n{Colors.BOLD}--- Step A: Generating Fingerprint ({'HIP' if is_amd else 'CUDA'}) ---{Colors.ENDC}")
    run_kernel_cmd = [f".{os.sep}{cuda_exe_name}"]
    try:
        subprocess.run(run_kernel_cmd, check=True)
    except subprocess.CalledProcessError:
        print_error("Execution of Kernel binary failed.")
        sys.exit(1)

    print(f"\n{Colors.BOLD}--- Step B: Analyzing Report (C++) ---{Colors.ENDC}")
    run_cpp_cmd = [f".{os.sep}{cpp_exe_name}", safe_gpu_name]
    try:
        subprocess.run(run_cpp_cmd, check=True)
    except subprocess.CalledProcessError:
        print_error("Execution of Probe Analysis failed.")
        sys.exit(1)

    print_step("Automation Complete.")

if __name__ == "__main__":
    run_project()