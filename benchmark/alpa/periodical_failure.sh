#!/bin/bash

# Periodically kill random GPU processes
# Usage: ./random_gpu_process_killer.sh [min_interval_seconds] [max_interval_seconds]

MIN_INTERVAL=${1:-300}  # Default: 5 minutes
MAX_INTERVAL=${2:-900}  # Default: 15 minutes

echo "Starting GPU process random termination script"
echo "Will trigger terminations at random intervals between $MIN_INTERVAL and $MAX_INTERVAL seconds"

# Get random time interval
get_random_interval() {
    echo $(( RANDOM % (MAX_INTERVAL - MIN_INTERVAL + 1) + MIN_INTERVAL ))
}

# Get random GPU ID
get_random_gpu() {
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [ $gpu_count -eq 0 ]; then
        echo "No GPU devices detected"
        return 1
    fi
    echo $(( RANDOM % gpu_count ))
}

# Kill random process on specified GPU
kill_random_gpu_process() {
    local gpu_id=$1
    echo "[$(date)] Finding processes on GPU $gpu_id..."
    
    # Get process IDs running on the specified GPU
    local pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i $gpu_id)
    
    if [ -z "$pids" ]; then
        echo "No running processes found on GPU $gpu_id"
        return 1
    fi
    
    # Convert PID list to array
    local pid_array=()
    while read -r pid; do
        if [ ! -z "$pid" ]; then
            pid_array+=($pid)
        fi
    done <<< "$pids"
    
    if [ ${#pid_array[@]} -eq 0 ]; then
        echo "No valid processes found on GPU $gpu_id"
        return 1
    fi
    
    # Select random process to kill
    local rand_index=$(( RANDOM % ${#pid_array[@]} ))
    local target_pid=${pid_array[$rand_index]}
    
    echo "[$(date)] Terminating process PID=$target_pid on GPU $gpu_id"
    
    # Log process info
    local process_info=$(ps -p $target_pid -o cmd= 2>/dev/null)
    if [ ! -z "$process_info" ]; then
        echo "Process being terminated: $process_info"
    fi
    
    # Send SIGTERM signal
    kill -15 $target_pid
    sleep 2
    
    # Force kill if still running
    if ps -p $target_pid > /dev/null 2>&1; then
        echo "Process not responding to SIGTERM, sending SIGKILL"
        kill -9 $target_pid
    fi
    
    echo "[$(date)] Terminated PID=$target_pid on GPU $gpu_id"
    echo "[$(date)] Terminated GPU $gpu_id process PID=$target_pid: $process_info" >> gpu_process_killer.log
}

# Main loop
while true; do
    interval=$(get_random_interval)
    echo "Waiting $interval seconds until next termination..."
    sleep $interval
    
    gpu_id=$(get_random_gpu)
    if [ $? -ne 0 ]; then
        echo "Failed to get GPU device, waiting for next cycle"
        continue
    fi
    
    kill_random_gpu_process $gpu_id
    if [ $? -ne 0 ]; then
        echo "No processes terminated this cycle, waiting for next"
    fi
done