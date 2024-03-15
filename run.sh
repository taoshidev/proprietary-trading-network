#!/bin/bash

# Initialize variables
script="neurons/validator.py"
autoRunLoc=$(readlink -f "$0")
proc_name="ptn"
args=()
version_location="meta/meta.json"
version=".subnet_version"

old_args=$@

echo "$old_args"

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Checks if $1 is smaller than $2
# If $1 is smaller than or equal to $2, then true.
# else false.
version_less_than_or_equal() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

# Checks if $1 is smaller than $2
# If $1 is smaller than $2, then true.
# else false.
version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

# Returns the difference between
# two versions as a numerical value.
get_version_difference() {
    local tag1="$1"
    local tag2="$2"

    # Extract the version numbers from the tags
    local version1=$(echo "$tag1" | sed 's/v//')
    local version2=$(echo "$tag2" | sed 's/v//')

    # Split the version numbers into an array
    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    # Calculate the numerical difference
    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}

        # Compare the numbers and update the difference
        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

read_version_value() {
    jq -r $version "$version_location"
}


check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)

    if [[ "$os_name" == "Linux" ]]; then
        # Use dpkg-query to check if the package is installed
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
         if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system"
        return 0
    fi
}

check_variable_value_on_github() {
    local repo="$1"
    local file_path="$2"
    local variable_name="$3"

    local url="https://api.github.com/repos/$repo/contents/$file_path"
    local response=$(curl -s "$url")

    # Check if the response contains an error message
    if [[ $response =~ "message" ]]; then
        echo "Error: Failed to retrieve file contents from GitHub."
        return 1
    fi

    # Extract the base64 content and decode it
    json_content=$(echo "$response" | jq -r '.content' | base64 --decode)

    # Extract the "subnet_version" value using jq
    subnet_version=$(echo "$json_content" | jq -r '.subnet_version')

    # Print the value
    echo "$subnet_version"
}

strip_quotes() {
    local input="$1"

    # Remove leading and trailing quotes using parameter expansion
    local stripped="${input#\"}"
    stripped="${stripped%\"}"

    echo "$stripped"
}

# Loop through all command line arguments
while [[ $# -gt 0 ]]; do
  arg="$1"

  # Check if the argument starts with a hyphen (flag)
  if [[ "$arg" == -* ]]; then
    # Check if the argument has a value
    if [[ $# -gt 1 && "$2" != -* ]]; then
          if [[ "$arg" == "--script" ]]; then
            script="$2";
            shift 2
        else
            # Add '=' sign between flag and value
            args+=("'$arg'");
            args+=("'$2'");
            shift 2
        fi
    else
      # Add '=True' for flags with no value
      args+=("'$arg'");
      shift
    fi
  else
    # Argument is not a flag, add it as it is
    args+=("'$arg '");
    shift
  fi
done

# Check if script argument was provided
if [[ -z "$script" ]]; then
    echo "The --script argument is required."
    exit 1
fi

branch=$(git branch --show-current)            # get current branch.
echo watching branch: $branch
echo pm2 process name: $proc_name

# Get the current version locally.
current_version=$(read_version_value)

# Check if script is already running with pm2
if pm2 status | grep -q $proc_name; then
    echo "The script is already running with pm2. Stopping and restarting..."
    pm2 delete $proc_name
fi

# Run the Python script with the arguments using pm2
echo "Running $script with the following pm2 config:"

# Join the arguments with commas using printf
joined_args=$(printf "%s," "${args[@]}")

# Remove the trailing comma
joined_args=${joined_args%,}

# Create the pm2 config file
echo "module.exports = {
  apps : [{
    name   : '$proc_name',
    script : '$script',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: [$joined_args]
  }]
}" > app.config.js

# Print configuration to be used
cat app.config.js

pm2 start app.config.js


# Check if packages are installed.
check_package_installed "jq"
if [ "$?" -eq 1 ]; then
    while true; do
        # First ensure that this is a git installation
        if [ -d "./.git" ]; then
            # check value on github remotely
            latest_version=$(check_variable_value_on_github "taoshidev/prop-net" $version_location $version)

            # Wait until the variable is not empty
            while [ -z "$latest_version" ]; do
                echo "Waiting for latest version to be set..."
                sleep 1  # You can adjust the sleep duration as needed
            done

            echo "latest version value" $latest_version

            latest_version="${latest_version#"${latest_version%%[![:space:]]*}"}"
            current_version="${current_version#"${current_version%%[![:space:]]*}"}"

            if [ -n "$latest_version" ] && ! echo "$latest_version" | grep -q "Error"; then
                # If the file has been updated
                if [ "$latest_version" != "$current_version" ]; then
                    echo "---- updating because of version mismatch ----"
                    echo "current validator version:" "$current_version"
                    echo "latest validator version:" "$latest_version"

                    # Pull latest changes
                    # Failed git pull will return a non-zero output
                    if git pull origin $branch; then
                        # latest_version is newer than current_version, should download and reinstall.
                        echo "New version published. Updating the local copy."

                        # Install latest changes just in case.
                        pip install -e .

                        # # Run the Python script with the arguments using pm2
                        # TODO (shib): Remove this pm2 del in the next spec version update.
                        pm2 del tsps
                        echo "Restarting PM2 process"
                        pm2 restart $proc_name

                        # Update current version:
                        current_version=$(read_version_value)
                        echo ""

                        # Restart autorun script
                        echo "Restarting script..."
                        ./$(basename $0) $old_args && exit
                    else
                        echo "**Will not update**"
                        echo "It appears you have made changes on your local copy. Please stash your changes using git stash."
                    fi
                else
                    echo "**Skipping update **"
                    echo "$current_version is the same as or more than $latest_version. You are likely running locally."
                fi
            fi
        else
            echo "The installation does not appear to be done through Git. Please install from source at https://github.com/taoshidev/prop-net and rerun this script."
        fi
        # Check if the process is running if something went sideways
        if pm2 list | grep -q "$proc_name"; then
            echo "Process $proc_name is already running."
        else
            echo "Process $proc_name is not running. Starting it..."
            pm2 start $proc_name
        fi
        # Wait about 30 minutes
        # This should be plenty of time for validators to catch up
        # and should prevent any rate limitations by GitHub.
        sleep 1800
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi

