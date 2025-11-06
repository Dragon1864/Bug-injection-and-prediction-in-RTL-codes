import os
import subprocess
import re
import csv
import random

# Configuration
SOURCE_FILE = "v8cpu.v"
BACKUP_FILE = "v8cpu.v.bak"
TESTBENCH_FILE = "v8cpu_alu_tb.v"
MEM_FILE = "v8cpu_mem_sim.v"
DATASET_CSV = "ml_training_dataset.csv"
IVERILOG_PATH = "iverilog"
VVP_PATH = "vvp"

# Bug patterns to inject
BUG_PATTERNS = [
    {
        "name": "ALU_ADD_to_SUB",
        "pattern": r'(c\s*=\s*a\s*\+\s*b;)',
        "replacement": r'c = a - b;'
    },
    {
        "name": "ALU_SUB_to_ADD",
        "pattern": r'(c\s*=\s*a\s*-\s*b;)',
        "replacement": r'c = a + b;'
    },
    {
        "name": "ALU_AND_to_OR",
        "pattern": r'(c\s*=\s*a\s*&\s*b;)',
        "replacement": r'c = a | b;'
    },
    {
        "name": "ALU_OR_to_AND",
        "pattern": r'(c\s*=\s*a\s*\|\s*b;)',
        "replacement": r'c = a & b;'
    },
    {
        "name": "ALU_XOR_to_AND",
        "pattern": r'(c\s*=\s*a\s*\^\s*b;)',
        "replacement": r'c = a & b;'
    }
]

def generate_diverse_test_vectors(bug_name, base_tests=1000, noise_level=0.1):
    vectors = []

    edge_cases = [
        0, 1, 255, 254, 128, 127, 64, 63,
        170, 85
    ]

    def noise(x):
        if random.random() < noise_level:
            return (x + random.randint(-5, 5)) % 256
        return x

    for a in edge_cases:
        for b in edge_cases:
            if bug_name == "ALU_ADD_to_SUB":
                op = 0
                expected = (a + b) & 0xFF
            elif bug_name == "ALU_SUB_to_ADD":
                op = 1
                expected = (a - b) & 0xFF
            elif bug_name == "ALU_AND_to_OR":
                op = 2
                expected = a & b
            elif bug_name == "ALU_OR_to_AND":
                op = 3
                expected = a | b
            elif bug_name == "ALU_XOR_to_AND":
                op = 4
                expected = a ^ b
            else:
                op = random.choice([0,1,2,3,4])
                if op == 0:
                    expected = (a + b) & 0xFF
                elif op == 1:
                    expected = (a - b) & 0xFF
                elif op == 2:
                    expected = a & b
                elif op == 3:
                    expected = a | b
                else:
                    expected = a ^ b
            vectors.append((op, a, b, expected))

    for _ in range(base_tests):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        op = random.choice([0,1,2,3,4])
        a = noise(a)
        b = noise(b)

        if bug_name == "ALU_ADD_to_SUB":
            op = 0
            expected = (a + b) & 0xFF
        elif bug_name == "ALU_SUB_to_ADD":
            op = 1
            expected = (a - b) & 0xFF
        elif bug_name == "ALU_AND_to_OR":
            op = 2
            expected = a & b
        elif bug_name == "ALU_OR_to_AND":
            op = 3
            expected = a | b
        elif bug_name == "ALU_XOR_to_AND":
            op = 4
            expected = a ^ b
        else:
            if op == 0:
                expected = (a + b) & 0xFF
            elif op == 1:
                expected = (a - b) & 0xFF
            elif op == 2:
                expected = a & b
            elif op == 3:
                expected = a | b
            else:
                expected = a ^ b
        
        vectors.append((op, a, b, expected))

    return vectors

def backup_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as src:
            content = src.read()
        with open(BACKUP_FILE, 'w') as backup:
            backup.write(content)
        print(f"Backed up {filepath} to {BACKUP_FILE}")

def restore_file():
    if os.path.exists(BACKUP_FILE):
        with open(BACKUP_FILE, 'r') as backup:
            content = backup.read()
        with open(SOURCE_FILE, 'w') as src:
            src.write(content)
        print(f"Restored {SOURCE_FILE} from backup")

def inject_bug(bug_pattern, output_file):
    with open(BACKUP_FILE, 'r') as f:
        content = f.read()
    modified_content = re.sub(bug_pattern["pattern"], bug_pattern["replacement"], content, count=1)
    if modified_content == content:
        print(f"Warning: Pattern '{bug_pattern['name']}' not found!")
        return False
    with open(output_file, 'w') as f:
        f.write(modified_content)
    print(f"Injected bug: {bug_pattern['name']}")
    return True

def create_testbench(test_vectors, output_log="simulation_output.log"):
    tb_content = """
`timescale 1ns/1ps
module v8cpu_alu_tb;
    reg [7:0] a, b;
    reg [3:0] op;
    wire [7:0] c;
    wire [7:0] newFlags;
    v8cpu_alu uut (.a(a), .b(b), .op(op), .c(c), .flags(8'b0), .newFlags(newFlags));
    integer outfile;
    initial begin
        outfile = $fopen("{0}", "w");
""".format(output_log)
    for op, a, b, _ in test_vectors:
        tb_content += f"""
        op = 4'd{op};
        a = 8'd{a};
        b = 8'd{b};
        #10;
        $fwrite(outfile, "%d,%d,%d,%d,%d\\n", op, a, b, c, newFlags);
"""
    tb_content += """
        $fclose(outfile);
        $finish;
    end
endmodule
"""
    with open(TESTBENCH_FILE, "w") as f:
        f.write(tb_content)
    print(f"Created testbench with {len(test_vectors)} vectors")

def compile_and_simulate(verilog_file, testbench_file):
    compiled_file = "simulation.vvp"
    cmd = [IVERILOG_PATH, "-o", compiled_file, verilog_file, testbench_file]
    if MEM_FILE:
        cmd.insert(-1, MEM_FILE)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return None
    sim = subprocess.run([VVP_PATH, compiled_file], capture_output=True, text=True)
    if sim.returncode != 0:
        print(f"Simulation failed: {sim.stderr}")
        return None
    print("Simulation successful")
    return "simulation_output.log"

def parse_sim_output(log_file):
    results = []
    if not os.path.exists(log_file):
        return results
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) == 5:
                    op, a, b, c, flags = map(int, parts)
                    results.append((op,a,b,c,flags))
    return results

def save_dataset_header(csv_file):
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant_id","bug_type","op","a","b",
            "expected_c","actual_c","expected_flags",
            "actual_flags","mismatch_c","mismatch_flags"
        ])

def save_dataset(variant_id, bug_type, test_vectors, sim_results, csv_file):
    exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    if not exists:
        save_dataset_header(csv_file)
    with open(csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        for i, (op, a, b, expected_c) in enumerate(test_vectors):
            if i < len(sim_results):
                _, _, _, actual_c, actual_flags = sim_results[i]
                mismatch_c = 1 if expected_c != actual_c else 0
                mismatch_flags = 1 if actual_flags != 0 else 0
            else:
                actual_c = -1
                actual_flags = -1
                mismatch_c = 1
                mismatch_flags = 1
            writer.writerow([variant_id, bug_type, op, a, b, expected_c, actual_c, 0, actual_flags, mismatch_c, mismatch_flags])
    print(f"Saved {len(test_vectors)} to {csv_file}")

def main():
    print("=== Start bug injection and ML data generation ===")
    if os.path.exists(DATASET_CSV):
        os.remove(DATASET_CSV)
    backup_file(SOURCE_FILE)

    clean_vectors = generate_diverse_test_vectors("NONE")
    create_testbench(clean_vectors)
    print("Testing clean design")
    clean_log = compile_and_simulate(SOURCE_FILE, TESTBENCH_FILE)
    if clean_log is None:
        print("Clean design simulation failed")
        restore_file()
        return
    clean_results = parse_sim_output(clean_log)
    save_dataset("clean", "NONE", clean_vectors, clean_results, DATASET_CSV)

    for idx, bug in enumerate(BUG_PATTERNS):
        print(f"Injecting bug {idx}: {bug['name']}")
        buggy_file = f"v8cpu_bug_{idx:03d}.v"
        variant_id = f"bug_{idx:03d}"

        if not inject_bug(bug, buggy_file):
            continue
        vectors = generate_diverse_test_vectors(bug['name'])
        create_testbench(vectors)
        bug_log = compile_and_simulate(buggy_file, TESTBENCH_FILE)
        if bug_log:
            bug_results = parse_sim_output(bug_log)
            save_dataset(variant_id, bug['name'], vectors, bug_results, DATASET_CSV)
        else:
            print(f"Bug {bug['name']} simulation failed, saving only mismatch markers")
            save_dataset(variant_id, bug['name'], vectors, [], DATASET_CSV)

        if os.path.exists(buggy_file):
            os.remove(buggy_file)

    restore_file()
    print("=== Dataset generation complete ===")

if __name__ == "__main__":
    main()
