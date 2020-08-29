import csv
from tqdm import tqdm

model = "DeepID"
mapping = "Same_Column_First_Mapping"
scheduling = "Pipeline"

path = './statistics/'+ model + '/' + mapping + '/' + scheduling + '/'
new_arr = []
with open(path+'PE_utilization.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    idx = 0
    scale = 2
    for row in tqdm(csv_reader):
        #if idx % scale == 0:
        if idx > 600000:
            new_arr.append(row)
        idx += 1

with open(path+'Scale_PE_utilization.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in tqdm(new_arr):
        writer.writerow(row)
