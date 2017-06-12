import csv
from matplotlib import pyplot as plt
import os


def read_csv(path, max_step, l):
    step = []
    loss = []
    with open(path, "rt") as infile:
        csvreader = csv.DictReader(infile)
        for row in csvreader:
            if int(row["Step"]) > max_step:
                break
            step.append(row["Step"])
            loss.append(l * float(row["Value"]))
    return step,loss

input_path = "mnist_losses"

plt.title("maaGMA Style Samples")
step, loss = read_csv("summaries/mnist_csv/run_.,tag_generator_Gaussian_discriminator_cost.csv", 800000, 0.01)
plt.plot(step, loss)
step, loss = read_csv("summaries/mnist_csv/run_.,tag_generator_image_discriminator_cost.csv", 800000, 0.002)
plt.plot(step, loss)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.savefig(os.path.join(input_path, "whatever.png"))