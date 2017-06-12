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

plt.title("Generator Losses by Source")
step, loss = read_csv("summaries/mnist_csv/run_.,tag_generator_image_discriminator_cost.csv", 800000, 0.002, 1000)
plt.plot(step, loss, label="Output Dis.")
step, loss = read_csv("summaries/mnist_csv/run_.,tag_generator_reconstruction_cost.csv", 800000, 1, 1000)
plt.plot(step, loss, label="Reconstruction")
plt.xlabel("Training Step")
plt.ylabel("Cost")
plt.legend()

plt.savefig(os.path.join(input_path, "MNIST_genlosses.png"))