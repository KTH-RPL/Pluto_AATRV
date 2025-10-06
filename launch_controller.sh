#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. DEFINE FILE AND PACKAGE NAMES ---
PACKAGE_NAME="robot_controller"
LAUNCH_FILE_NAME="test_controller.launch"
CONFIG_FILE_NAME="preview_controller_config.yaml"

# --- 2. GENERATE TIMESTAMPED LOG FILENAME ---
# Create a log file name in the /tmp directory based on the current date and time.
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="/tmp/${PACKAGE_NAME}_${TIMESTAMP}.log"

# --- 3. LOCATE ROS PACKAGE AND CONFIG FILE ---
echo "Searching for ROS package: $PACKAGE_NAME"
PKG_PATH=$(rospack find $PACKAGE_NAME)

# Check if the package was found
if [ -z "$PKG_PATH" ]; then
    echo "Error: Could not find package '$PACKAGE_NAME'. Make sure your workspace is sourced."
    exit 1
fi

YAML_FILE="${PKG_PATH}/config/${CONFIG_FILE_NAME}"

# Check if the YAML file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: Configuration file not found at: $YAML_FILE"
    exit 1
fi

echo "ROS package found at: $PKG_PATH"
echo "Logging configuration and all output to: $LOG_FILE"
echo # Blank line for readability

# --- 4. WRITE THE YAML CONFIG TO THE LOG FILE ---
# This creates the log file and adds the header and YAML content.
# The 'tee' command writes this initial header to both the screen and the file.
{
    echo "--- CONFIGURATION PARAMETERS from ${CONFIG_FILE_NAME} ---"
    echo ""
    cat "$YAML_FILE"
    echo ""
    echo "--- END OF CONFIGURATION ---"
    echo ""
    echo "--- STARTING ROS LAUNCH LOGS ---"
    echo ""
} | tee "$LOG_FILE"

# --- 5. EXECUTE ROSLAUNCH AND APPEND ALL OUTPUT ---
# Execute the roslaunch command.
# '2>&1' redirects stderr (2) to stdout (1), so both streams are combined.
# The pipe '|' sends the combined stream to 'tee'.
# 'tee -a' appends its input to the specified file while also printing it to the screen.
roslaunch ${PACKAGE_NAME} ${LAUNCH_FILE_NAME} 2>&1 | tee -a "$LOG_FILE"