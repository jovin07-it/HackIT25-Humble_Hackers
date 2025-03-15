import paramiko
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# EC2 instance details
HOST = "34.201.33.253"
USER = "ubuntu"  # or 'ec2-user' for Amazon Linux
SSH_KEY_PATH = "/content/jj.pem"

# Command to fetch system metrics remotely
CMD = """\
    echo "CPU: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}')"; \
    echo "RAM: $(free -m | awk '/Mem:/ {print $3}')"; \
    echo "Disk I/O: $(iostat -d | awk 'NR==4 {print $2}')"
"""

# Connect to EC2 instance via SSH
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, key_filename=SSH_KEY_PATH)

# Execute command & get output
stdin, stdout, stderr = ssh.exec_command(CMD)
output = stdout.read().decode().split("\n")
ssh.close()

# Parse the fetched metrics
cpu_util = float(output[0].split(":")[1].strip())
ram_util = float(output[1].split(":")[1].strip())
disk_io = float(output[2].split(":")[1].strip())

# Store metrics in a DataFrame
df_real_time = pd.DataFrame({
    "timestamp": pd.Timestamp.now(),
    "required_cpu_cores": [cpu_util],
    "required_ram": [ram_util]
})

print(df_real_time)
# Load previous model (if saved)
import pickle

# Load ARIMA models (previously trained)
with open("cpu_model.pkl", "rb") as f:
    cpu_model_fit = pickle.load(f)

with open("ram_model.pkl", "rb") as f:
    ram_model_fit = pickle.load(f)

# Predict next 24 hours usage
cpu_pred = cpu_model_fit.forecast(steps=24)
ram_pred = ram_model_fit.forecast(steps=24)

# Load cloud provider specs
cloud_specs_path = "/content/cloud_provider_specs.csv"
df_cloud = pd.read_csv(cloud_specs_path)

# Find best instances
best_providers = {}
for cpu, ram in zip(cpu_pred, ram_pred):
    for provider in df_cloud["Cloud_Provider"].unique():
        provider_instances = df_cloud[df_cloud["Cloud_Provider"] == provider]
        suitable_instances = provider_instances[
            (provider_instances["CPU_Cores"] >= cpu) &
            (provider_instances["Memory_GB"] >= ram)
        ]
        best_instance = suitable_instances.nsmallest(1, "Cost_Per_Hour") if not suitable_instances.empty else provider_instances.nsmallest(1, "Cost_Per_Hour")
        best_providers[provider] = best_instance

df_results = pd.concat(best_providers.values())
df_results["Total_Estimated_Cost"] = df_results["Cost_Per_Hour"] * 24

print(df_results)
threshold_cpu = df_results["CPU_Cores"].min() * 0.8  # 80% of min instance
threshold_ram = df_results["Memory_GB"].min() * 0.8

if cpu_util > threshold_cpu or ram_util > threshold_ram:
    print("⚠ Scaling required! Consider upgrading instance.")
else:
    print("✅ No scaling required. Instance is sufficient.")