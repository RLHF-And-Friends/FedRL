import json
import subprocess
import argparse
import os


def run_experiments(config_path):
    print(f"Run experiment with config {os.path.basename(config_path)}")

    with open(config_path, 'r') as f:
        configs = json.load(f)

    process_ids = []

    for setup_name, setup_config in configs.items():
        command = ["python3", "-m", "federated_ppo.main"]
        for flag, value in setup_config.items():
            if isinstance(value, bool):
                value = str(value)
            command.append(f"--{flag}={value}")

        if "comm_matrix_config" in setup_config:
            command.append(f"--comm-matrix-config={setup_config['comm_matrix_config']}")

        if "comm_penalty_coeff" in setup_config:
            command.append(f"--comm-penalty-coeff={setup_config['comm_penalty_coeff']}")

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
        stdout_log = open(f"{log_dir}/{setup_name}_stdout.log", "w")
        stderr_log = open(f"{log_dir}/{setup_name}_stderr.log", "w")

        print(f"Running setup: {setup_name}")
        print("Command:", " ".join(command))

        # Start the process
        process = subprocess.Popen(command, stdout=stdout_log, stderr=stderr_log)
        process_ids.append((setup_name, process.pid, stdout_log, stderr_log))

    # Output process IDs
    print("\nLaunched processes:")
    for setup_name, pid, _, _  in process_ids:
        print(f"Experiment {os.path.basename(config_path)}, setup: {setup_name}, PID: {pid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated PPO experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file (e.g., experiments/exp_7.json)."
    )
    args = parser.parse_args()

    config_path = args.config

    # Ensure the path is valid
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    run_experiments(config_path)