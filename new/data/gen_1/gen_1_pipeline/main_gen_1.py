import subprocess

# List of Python files to execute
scripts = [
    "process_TrainerDataGen1Raw.py",
    "process_wild_locations_gen_1_raw_v2.py",
    "create_variants_gen_1.py",
    "calculate_ehl_gen_1_v2.py",
    "evaluate_combinations_gen_1_v6.py"
]

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break
