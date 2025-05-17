#!/usr/bin/env python3
import rosbag
import sys
import os
from tqdm import tqdm # Optional: for progress bar (pip install tqdm)

def merge_bags(output_bag_path, input_bag_paths):
    """
    Merges multiple ROS bags chronologically based on message timestamps.

    Args:
        output_bag_path (str): Path for the merged output ROS bag file.
        input_bag_paths (list): A list of paths to the input ROS bag files.
    """
    if not input_bag_paths:
        print("Error: No input bags provided.")
        return

    print("Input bags:")
    for bag_path in input_bag_paths:
        if not os.path.exists(bag_path):
            print(f"Error: Input bag not found: {bag_path}")
            return
        print(f"- {bag_path}")

    print(f"\nOutput bag: {output_bag_path}")

    input_bags = [rosbag.Bag(bag_path, 'r') for bag_path in input_bag_paths]
    message_generators = [bag.read_messages(raw=False) for bag in input_bags] # raw=False gives deserialized messages

    # Get the first message from each bag to initialize
    current_messages = []
    active_generators = []
    total_messages = 0
    for i, gen in enumerate(message_generators):
        try:
            topic, msg, t = next(gen)
            current_messages.append({'topic': topic, 'msg': msg, 't': t, 'gen_idx': i})
            active_generators.append(gen)
            total_messages += input_bags[i].get_message_count() # Count total for progress bar
            print(f"Bag {i+1} start time: {t.to_sec()}")
        except StopIteration:
            print(f"Warning: Input bag {input_bag_paths[i]} is empty.")
            input_bags[i].close() # Close empty bag

    if not active_generators:
        print("Error: All input bags are empty or could not be read.")
        # Ensure all bags attempted to open are closed
        for bag in input_bags:
            try:
                bag.close()
            except: # Might already be closed if empty
                pass
        return

    print(f"\nStarting merge... Total messages approx: {total_messages}")

    with rosbag.Bag(output_bag_path, 'w') as outbag:
        # Use tqdm for progress bar if available
        pbar = None
        if 'tqdm' in sys.modules:
            pbar = tqdm(total=total_messages, unit='msgs', smoothing=0.1)

        while current_messages:
            # Find the message with the earliest timestamp
            current_messages.sort(key=lambda x: x['t'])
            earliest = current_messages.pop(0) # Get and remove the earliest

            # Write the earliest message to the output bag
            outbag.write(earliest['topic'], earliest['msg'], earliest['t'])
            if pbar: pbar.update(1)

            # Get the next message from the generator where the earliest message came from
            gen_idx = earliest['gen_idx']
            try:
                topic, msg, t = next(message_generators[gen_idx])
                # Add the new message back into our list to be sorted
                current_messages.append({'topic': topic, 'msg': msg, 't': t, 'gen_idx': gen_idx})
            except StopIteration:
                # This generator is exhausted, close its corresponding bag
                print(f"\nFinished reading bag: {input_bag_paths[gen_idx]} (Index {gen_idx})")
                input_bags[gen_idx].close()
                # Don't add anything back to current_messages for this generator

        if pbar: pbar.close()

    print(f"\nMerge complete. Output saved to: {output_bag_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_bags.py <output_bag.bag> <input_bag1.bag> <input_bag2.bag> [input_bag3.bag ...]")
        sys.exit(1)

    output_path = sys.argv[1]
    input_paths = sys.argv[2:]

    merge_bags(output_path, input_paths)