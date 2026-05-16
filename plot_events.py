import csv
import matplotlib.pyplot as plt

input_path = "data/events_indices.csv"

columns = [[] for _ in range(5)]

with open(input_path, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        for i in range(5):
            columns[i].append(int(row[i]))
        if len(columns[0]) == 50:
            break

fig, ax = plt.subplots(figsize=(12, 5))
for i, values in enumerate(columns):
    ax.plot(values, label=f"Element {i + 1}")

ax.set_xlabel("Row")
ax.set_ylabel("Column index")
ax.set_title("1st–5th elements per row")
ax.legend()
plt.tight_layout()
plt.show()
