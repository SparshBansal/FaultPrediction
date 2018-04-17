import numpy as np
import csv

def read_and_preprocess():
    data=[]

    x_data=[]
    y_data=[]

    label_map = {'true' : 1, 'false' : 0}

    with open('pc1.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # create the x_data and y_data lists 
            x_dc = [row[val] for val in row.keys()]
            data.append(x_dc)


    # separate x and y vector
    data_np = np.array(data)

    x_data = data_np[:,0:3]
    x_data = np.append(x_data,data_np[:,4:],axis=1)

    y_data = data_np[:,3]
    y_data = np.array([label_map[val] for val in y_data])

    # convert strings to float32
    x_data = x_data.astype(np.float32)

    return shuffle(x_data, y_data)

def shuffle(X,Y):
    index = np.arange(0, X.shape[0])
    np.random.shuffle(index)

    shuf_x = np.array(X)
    shuf_y = np.array(Y)

    for x in range(X.shape[0]):
        shuf_x[x] = X[index[x]]
        shuf_y[x] = Y[index[x]]
    
    return shuf_x, shuf_y

x, y = read_and_preprocess()
