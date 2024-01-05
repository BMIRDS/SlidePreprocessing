import datetime


def print_intro(script_name: str):
    print(f"[INFO] Running script: {script_name}")
    print(f"[INFO] Script execution started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def print_outro(script_name: str):
    print(f"[INFO] Finished running script: {script_name}")
    print(f"[INFO] Script execution completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
