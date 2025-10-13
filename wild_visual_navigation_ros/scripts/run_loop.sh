#!/bin/bash

# === RViz one time ===
echo "Starting RViz (view.launch)..."
roslaunch wild_visual_navigation_ros view.launch &
sleep 5  # waiting time

# === 100 loop ===
for i in $(seq 1 200); do
    echo "=============================="
    echo " Run $i / 200 "
    echo "=============================="

    # wild_visual_navigation.launch
    echo "[Launch] wild_visual_navigation.launch"
    roslaunch wild_visual_navigation_ros wild_visual_navigation.launch &
    LAUNCH_PID=$!

    # waitinig & bag
    sleep 3
    echo "[Bag] Playing slag_heap_0.bag"
    rosbag play ~/catkin_ws/src/wild_visual_navigation/dataset_2/slag_heap_2023-09-14-12-23-05/slag_heap_0.bag --quiet --clock
    BAG_STATUS=$?

    # if bag finish, launch stop
    echo "[Stop] Killing nodes..."
    rosnode kill -a
    sleep 3

    echo "✅ Run $i completed."
    echo ""
done

echo "🎉 All 200 runs finished!"
