import subprocess
import json
import os
import traceback

FLAG_FILE_PATH = 'restart_sn8_flag3.txt'

def find_pm2_process_by_name(process_name):
    try:
        result = subprocess.run(['pm2', 'jlist'], capture_output=True, text=True, check=True)
        processes = json.loads(result.stdout)
        for proc in processes:
            if proc['name'] == process_name:
                return process_name  # Return the process name if found
    except subprocess.CalledProcessError as e:
        print(f"Error fetching PM2 list: {e}")
    return None  # Return None if no specific process is found

def find_pm2_process_by_script(script_name):
    try:
        result = subprocess.run(['pm2', 'jlist'], capture_output=True, text=True, check=True)
        processes = json.loads(result.stdout)
        for proc in processes:
            if proc['pm2_env']['pm_exec_path'].endswith(script_name):
                return proc['name']  # Return the process name if found
    except subprocess.CalledProcessError as e:
        print(f"Error fetching PM2 list: {e}")
    return None  # Return None if no matching process is found by script name

def restart_sn8():
    try:
        if os.path.exists(FLAG_FILE_PATH):
            print("The script has already been run. Exiting...")
            return

        # First try finding and restarting the 'sn8' process
        process_name = find_pm2_process_by_name('sn8')
        if process_name:
            print(f"Found and restarting process named: {process_name}")
        else:
            print("Process 'sn8' not found, searching by script name...")
            process_name = find_pm2_process_by_script('run.sh')
            if process_name:
                print(f"Found process by script to restart: {process_name}")
            else:
                print("No matching PM2 process found by script name either.")
                return

        # Restart the found process
        subprocess.run(['pm2', 'restart', process_name], check=True)
        with open(FLAG_FILE_PATH, 'w') as f:
            f.write("This script has run and the process was restarted.\n")
    except Exception as e:
        print(f"Error restarting the PM2 process: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    restart_sn8()