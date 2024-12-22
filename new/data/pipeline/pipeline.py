import subprocess
import pandas as pd
import sys

config = pd.read_csv('../config/config.csv')

generation = int(config[config.rule == 'gen'].value.values[0])

if generation == 1:
    # List of Python files to execute
    scripts = [
        "process_TrainerDataGen1Raw.py",
        "process_wild_locations_gen_1_raw_v2.py",
        "create_variants_gen_1.py",
        "calculate_ehl_gen_1_v2.py",
        "evaluate_combinations_gen_1_v6.py"
    ]

elif generation == 2:
    scripts = ["process_TrainerDataGen2Raw.py",
               "process_wild_locations_gen_2_raw_v2.py",
               "create_variants_gen_1.py",
               "calculate_ehl_gen_1_v2.py",
               "evaluate_combinations_gen_1_v6.py"]

elif generation == 3:
    scripts = []

else:
    scripts = []

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run([sys.executable, script], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break
