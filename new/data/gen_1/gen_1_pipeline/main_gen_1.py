import subprocess

config = pd.read_csv('../gen_1_config/gen_1_config.csv')

generation = config[config.rule == 'restrictions'].gen.values[0].lower()

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
    scripts = []

elif generation == 3:
    scripts = []

else:
    scripts = []

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break
