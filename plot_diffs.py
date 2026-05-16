import csv
import matplotlib.pyplot as plt

input_path = "data/events_indices.csv"

diffs = [[] for _ in range(4)]

with open(input_path, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        vals = [int(row[i]) for i in range(5)]
        for i in range(4):
            diffs[i].append(vals[i + 1] - vals[i])
        if len(diffs[0]) == 50:
            break

sums = [sum(diffs[i][j] for i in range(4)) for j in range(len(diffs[0]))]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

labels = ["2nd-1st", "3rd-2nd", "4th-3rd", "5th-4th"]
for i, values in enumerate(diffs):
    ax1.plot(values, label=labels[i])
ax1.set_xlabel("Row")
ax1.set_ylabel("Difference")
ax1.set_title("Consecutive element differences per row")
ax1.legend()

ax2.plot(sums)
ax2.set_xlabel("Row")
ax2.set_ylabel("Sum of differences")
ax2.set_title("Sum of all differences per row")

plt.tight_layout()
plt.show()
