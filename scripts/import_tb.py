from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
import wandb

# 1. Initialize a new W&B run
wandb.init(project="LLama32", name="llama32_20260926_1527_v2")

# 2. Specify the directory containing your .tfevents file
log_dir = "runs/llama32_20260926_1527"

print(f"Loading TensorBoard logs from {log_dir}...")
ea = EventAccumulator(log_dir)
ea.Reload()

# 3. Extract and group scalar metrics by step
if "scalars" in ea.Tags():
    scalars = ea.Tags()["scalars"]
    step_data = defaultdict(dict)

    for tag in scalars:
        for event in ea.Scalars(tag):
            step_data[event.step][tag] = event.value

    print(f"Found {len(scalars)} scalar metrics across {len(step_data)} steps.")
    print("Uploading to Weights & Biases...")

    # 4. Log to W&B step by step
    for step in sorted(step_data.keys()):
        wandb.log(step_data[step], step=step)

# 5. Finalize the run
wandb.finish()
print("Successfully synced TensorBoard logs to W&B!")
