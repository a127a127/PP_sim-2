import csv

new_arr = []
with open('PE_utilization.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    idx = 0
    scale = 100
    for row in csv_reader:
        if idx % scale == 0:
            new_arr.append(row)
        idx += 1

with open('New_PE_utilization.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in new_arr:
        writer.writerow(row)
