import csv

input_path = "data/events.csv"
output_path = "data/events_indices.csv"

result = []
with open(input_path, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        indices = [i + 1 for i, val in enumerate(row) if val.strip() == "1"]
        result.append(indices)

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(result)

for indices in result:
    print(indices)
