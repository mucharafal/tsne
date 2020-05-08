import random
import sys

if(len(sys.argv) < 3):
    print("Too few args")

points_number = int(sys.argv[1])
dimensions = int(sys.argv[2])

with open("test.csv", "w+") as file_handle:
    for i in range(points_number):
        file_handle.write(str(random.random()))
        for j in range(dimensions - 1):
            file_handle.write(", ")
            file_handle.write(str(random.random()))
    
        file_handle.write("\n")
