import subprocess
import matplotlib.pyplot as plt

# Model scripts to be executed
model_scripts = [
    "qda.py", "ranforest.py", "svm.py",
    "gaussnb.py", "knn.py", "lda.py",
    "logreg.py", "adaboost.py", "dectree.py",
    "mlp.py"
]

# Dictionary to hold the metrics for each model
metrics = {}

# Function to run scripts and capture output
def run_model(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    output = result.stdout
    return output

# Extract metrics from output
def extract_metrics(output):
    metrics_data = {}
    for line in output.split('\n'):
        if ':' in line:
            key, value = line.split(':')
            try:
                value = float(value.strip())
                metrics_data[key.strip()] = value
            except ValueError:
                continue  # Ignora linhas que não contêm valores numéricos
    return metrics_data

# Running each model and storing metrics
for script in model_scripts:
    print(f"Running {script}...")
    output = run_model(script)
    model_metrics = extract_metrics(output)
    if model_metrics:
        metrics[script.replace('.py', '')] = model_metrics
    else:
        print(f"Failed to extract metrics for {script}")

# Plotting the metrics
colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'gray']
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
fig, ax = plt.subplots(figsize=(10, 8))

# Create bars for each metric
for i, label in enumerate(labels):
    bar_offset = (i * 0.1)  # Offset each group of bars slightly for better visibility
    model_values = [metrics.get(model, {}).get(label, 0) for model in metrics]
    ax.barh([model for model in metrics], model_values, color=colors[i % len(colors)], label=label, left=bar_offset)

ax.set_xlabel('Score')
ax.set_title('Performance Metrics of Models')
ax.legend()
plt.tight_layout()
plt.show()
