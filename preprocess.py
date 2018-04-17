import numpy as np
import csv

with open('pc1.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print row["loc"]
