
import csv
#import java.util.*;

def avg(lst):
    return sum(lst)/len(lst)
with open('RANet_train_energy2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst=[]
        row=row[0:len(row)-1]
        for r in row:
            lst.append(float(r))
        lst.sort()
        if(lst[0]<0.75*lst[4]):
            lst=lst[1:4]
        info=[]
        info.append(avg(lst))
        f = open("RANet_train_energy_avg.csv", "a")
        writer = csv.writer(f)
        # print(energy)
        writer.writerow(info)
        f.close()
        #lst.append(float(row[2]))
