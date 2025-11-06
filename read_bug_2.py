import subprocess
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths and configs
DATASET_CSV = "ml_training_dataset.csv"
MODEL_FILE = "xgb_model_actual_only.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"
RTL_FILE = "v8cpu_bug2.v"
MEM_FILE = "v8cpu_mem_sim.v"
TESTBENCH_FILE = "v8cpu_alu_tb.v"
SIM_LOG_FILE = "simulation_output.log"
IVERILOG = "iverilog"
VVP = "vvp"

verilog_files = [MEM_FILE, RTL_FILE]

# Train model on actual features only
def train_model():
    df = pd.read_csv(DATASET_CSV)
    features = ['op', 'a', 'b', 'actual_c', 'actual_flags']
    X = df[features]
    y = df['bug_type']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)

    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("Model and label encoder saved.\n")

# Generate testbench
def generate_testbench(test_vectors):
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
""".format(SIM_LOG_FILE)
    for op, a, b in test_vectors:
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
    with open(TESTBENCH_FILE, 'w') as f:
        f.write(tb_content)
    print(f"Testbench {TESTBENCH_FILE} generated.")

# Run iverilog compilation
def run_compilation():
    cmd = [IVERILOG, "-o", "simulation.vvp"] + verilog_files + [TESTBENCH_FILE]
    print("Compiling with:", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:", result.stderr)
        return False
    print("Compilation succeeded.")
    return True

# Run simulation
def run_simulation():
    sim_cmd = [VVP, "simulation.vvp"]
    print("Simulating with:", ' '.join(sim_cmd))
    result = subprocess.run(sim_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Simulation failed:", result.stderr)
        return False
    print("Simulation succeeded.")
    return True

# Parse simulation output into DataFrame
def parse_sim_output():
    if not os.path.exists(SIM_LOG_FILE):
        print("Simulation output log not found")
        return pd.DataFrame()
    data = []
    with open(SIM_LOG_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 5:
                op, a, b, actual_c, actual_flags = map(int, parts)
                data.append({'op': op, 'a': a, 'b': b, 'actual_c': actual_c, 'actual_flags': actual_flags})
    return pd.DataFrame(data)

# Load model and label encoder
def load_model_and_encoder():
    model = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    print("Loaded model and label encoder.")
    return model, le

# Predict bugs and decode predictions
def predict_and_decode(df, model, label_encoder):
    features = ['op', 'a', 'b', 'actual_c', 'actual_flags']
    X = df[features]
    preds = model.predict(X)
    df['prediction'] = preds
    df['bug_label'] = label_encoder.inverse_transform(preds)
    return df

# Main script usage
if __name__ == "__main__":
    # Step 1: Train model (only run once or when dataset changes)
    train_model()

    # Step 2: Generate testbench with example vectors
    test_vectors = [
    (0, 12, 24), (0, 45, 23), (0, 200, 100), (0, 50, 50), (0, 75, 125),
    (0, 100, 150), (0, 180, 60), (0, 5, 255), (0, 30, 30), (0, 90, 10),
    (0, 10, 90), (0, 123, 234), (0, 255, 0), (0, 0, 255), (0, 77, 77),
    (0, 88, 99), (0, 15, 20), (0, 34, 56), (0, 78, 12), (0, 140, 200),
    (0, 31, 63), (0, 127, 128), (0, 22, 44), (0, 66, 99), (0, 1, 1),
    (0, 5, 5), (0, 10, 10), (0, 20, 20), (0, 40, 40), (0, 80, 80),
    (0, 160, 160), (0, 200, 200), (0, 220, 220), (0, 240, 240), (0, 250, 250),
    (0, 255, 255), (0, 100, 100), (0, 50, 25), (0, 25, 50), (0, 60, 30),
    (0, 30, 60), (0, 70, 35), (0, 35, 70), (0, 90, 45), (0, 45, 90),
    (0, 110, 55), (0, 55, 110), (0, 130, 65), (0, 65, 130), (0, 150, 75)
]
    generate_testbench(test_vectors)

    # Step 3: Compile and simulate
    if run_compilation() and run_simulation():
        # Step 4: Parse output
        df = parse_sim_output()

        # Step 5: Load model and encoder, predict and decode
        model, label_encoder = load_model_and_encoder()
        df_results = predict_and_decode(df, model, label_encoder)

        # Show results
        print(df_results[['op', 'a', 'b', 'actual_c', 'actual_flags', 'prediction', 'bug_label']])
    else:
        print("Simulation or compilation failed, exiting.")
