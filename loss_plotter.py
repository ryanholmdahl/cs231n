import csv
from matplotlib import pyplot as plt
import os


def read_csv(path, max_step, l, step_count):
    step = []
    loss = []
    prev_step = -1
    with open(path, "rt") as infile:
        csvreader = csv.DictReader(infile)
        for row in csvreader:
            cur_step = int(row["Step"])
            if cur_step > max_step:
                break
            if cur_step - prev_step > step_count:
                step.append(cur_step)
                loss.append(l * float(row["Value"]))
                prev_step = cur_step
    return step,loss

input_path = "mnist_losses"

fig, ax1 = plt.subplots()
plt.title("Generator Losses by Source")
step, loss = read_csv("summaries/lfw0/run_.,tag_generator_image_discriminator_cost.csv", 200000, 0.001, 5000)
ax1.plot(step, loss, label="Output Dis.", color='b')
ax1.set_ylabel("Output Dis. Cost", color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
step, loss = read_csv("summaries/lfw0/run_.,tag_generator_reconstruction_cost.csv", 200000, 1, 5000)
ax2.plot(step, loss, label="Reconstruction", color='r')
ax1.set_xlabel("Training Step")
ax2.set_ylabel("Reconstruction Cost", color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()

plt.savefig(os.path.join(input_path, "LFW_genlosses.png"))