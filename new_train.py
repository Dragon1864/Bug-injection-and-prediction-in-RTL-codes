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
MODEL_FILE = "xgb_model_expected_actual.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"
RTL_FILE = "v8cpu_bug1.v"
MEM_FILE = "v8cpu_mem_sim.v"
TESTBENCH_FILE = "v8cpu_alu_tb.v"
SIM_LOG_FILE = "simulation_output.log"
IVERILOG = "iverilog"
VVP = "vvp"

verilog_files = [MEM_FILE, RTL_FILE]

# Train model on actual + expected features
def train_model():
    df = pd.read_csv(DATASET_CSV)
    df['mismatch_c'] = (df['expected_c'] != df['actual_c']).astype(int)
    df['mismatch_flags'] = (df['expected_flags'] != df['actual_flags']).astype(int)

    features = ['op', 'a', 'b', 'expected_c', 'actual_c', 'expected_flags', 'actual_flags', 'mismatch_c', 'mismatch_flags']
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
    tb_content = f"""
`timescale 1ns/1ps
module v8cpu_alu_tb;
    reg [7:0] a, b;
    reg [3:0] op;
    wire [7:0] c;
    wire [7:0] newFlags;
    v8cpu_alu uut (.a(a), .b(b), .op(op), .c(c), .flags(8'b0), .newFlags(newFlags));

    integer outfile;
    initial begin
        outfile = $fopen("{SIM_LOG_FILE}", "w");
"""
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
    print(f"Testbench '{TESTBENCH_FILE}' generated.")

# Compile simulation
def run_compilation():
    cmd = [IVERILOG, "-o", "simulation.vvp"] + verilog_files + [TESTBENCH_FILE]
    print("Compiling:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:", result.stderr)
        return False
    print("Compilation succeeded.")
    return True

# Run simulation
def run_simulation():
    cmd = [VVP, "simulation.vvp"]
    print("Simulating:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Simulation failed:", result.stderr)
        return False
    print("Simulation succeeded.")
    return True

# Parse simulation output
def parse_sim_output():
    if not os.path.exists(SIM_LOG_FILE):
        print("Simulation log not found.")
        return pd.DataFrame()
    rows = []
    with open(SIM_LOG_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 5:
                op, a, b, actual_c, actual_flags = map(int, parts)
                rows.append({'op': op, 'a': a, 'b': b, 'actual_c': actual_c, 'actual_flags': actual_flags})
    return pd.DataFrame(rows)

# Load model and encoder
def load_model_and_enc():
    model = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    print("Loaded model and label encoder.")
    return model, le

# Predict and decode
def predict_and_decode(df_actual, df_expected, model, label_encoder):
    df = pd.merge(df_actual, df_expected, on=['op', 'a', 'b'], suffixes=('_actual', '_expected'))
    df['mismatch_c'] = (df['expected_c'] != df['actual_c']).astype(int)
    df['mismatch_flags'] = (df['expected_flags'] != df['actual_flags']).astype(int)

    features = ['op', 'a', 'b', 'expected_c', 'actual_c', 'expected_flags', 'actual_flags', 'mismatch_c', 'mismatch_flags']
    X = df[features]

    preds = model.predict(X)
    df['prediction'] = preds
    df['bug_label'] = label_encoder.inverse_transform(preds)
    return df

if __name__ == "__main__":
    train_model()

    test_vectors = [(0, 12, 24), (1, 45, 23), (2, 200, 100), (3, 50, 50), (4, 75, 125)]
    generate_testbench(test_vectors)

    if run_compilation() and run_simulation():
        df_actual = parse_sim_output()
        df_expected = pd.read_csv(DATASET_CSV)[['op', 'a', 'b', 'expected_c', 'expected_flags']]
        model, label_encoder = load_model_and_enc()
        df_results = predict_and_decode(df_actual, df_expected, model, label_encoder)

        print(df_results[['op', 'a', 'b',
                          'expected_c', 'actual_c',
                          'expected_flags', 'actual_flags',
                          'prediction', 'bug_label']])
    else:
        print("Simulation or compilation failed, exiting.")
