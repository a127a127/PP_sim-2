import csv

path = './statistics/Cifar10/Same_Column_First_Mapping/Non_pipeline/'
new_arr = []
with open(path+'PE_utilization.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    idx = 0
    scale = 100
    for row in csv_reader:
        #if idx % scale == 0:
        if idx > 1800000:
            new_arr.append(row)
        idx += 1

with open(path+'New_PE_utilization.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in new_arr:
        writer.writerow(row)
