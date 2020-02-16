#!/usr/bin/python3

# AUTHOR:  Yixin Zhang
# NetID:   yzh223
# csugID:  yzh223

import numpy as np
# TODO: understand that you should not need any other imports other than those
# already in this file; if you import something that is not installed by default
# on the csug machines, your code will crash and you will lose points

# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1] + [1], dtype=np.float64)
    y = np.float64(tokens[-1])
    #print(x)
    #print(y)
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        (xs,ys) = (np.array([v[0] for v in vals]),np.array([v[1] for v in vals]))
        print("---------------------- parse data ---------------------")
        print(xs)
        print(ys)
        return xs, ys


#For each dataset you must identify whether it is definitely linearly separable or not.
#you should report the iteration at which your algorithm converges.
#also report the maximum vector norm over the dataset (i.e. the value R from the convergence proof discussed in class)
#using the relation k < R2/δ2 report an upper bound on the value for the separting margin delta. 
# also produce a plot indicating accuracy per iteration (i.e., x-axis is number of iterations, y-axis is overall accuracy on the training data). 

#For the non-separable datasets, you should report the maximum number of iterations you explored, the overall highest accuracy obtained
#also include a short explanation of why you believe you have tested the data “enough” 

# Do learning.
def perceptron(train_xs, train_ys, iterations, rate):
    w = np.zeros(train_xs.shape[1])
    accuracy = []
    print("---------------------- training ---------------------")
    print("size of data: ", train_ys.shape[0])
    print("iterations", iterations)
    ck = 0
    for k in range (iterations):
        #print("iter", k)
        #print("w: ", w)
        correct = 0
        wrong = 0
        w_prev = w
        for i in range (train_ys.shape[0]):
            y = predict(train_xs[i], w)
            e = train_ys[i] - y
            #print("e ",e )
            if e == 0.0:
                correct+=1
            else:
                wrong+=1
            w = w + rate * e * train_xs[i]
            #print("w", w)
        accuracy.append(correct/(correct+wrong))
        if np.array_equal(w, w_prev):
            print()
            print("Converged at: {}th iterations".format(k))
            ck = k
            break
        else: 
            if k == iterations-1:
                print()
                print("Did not converge at current iterations")
    return w, accuracy, ck


def predict(xi, w):
    z = xi.dot(w)
    return 1 if z >= 0 else -1


# Return the accuracy over the data using current weights.
def test_accuracy(weights, test_xs, test_ys):
    correct = 0
    wrong = 0
    for i in range(test_ys.shape[0]):
        e = test_ys[i] - predict(test_xs[i], weights)
        if e == 0.0:
            correct+=1
        else:
            wrong+=1
    return correct/(correct+wrong)


def convergance_bound(train_xs, train_ys, weights):
    maxNorm = norm(train_xs[0])
    delta = train_xs[0].dot(weights)*train_ys[0]

    for i in range(train_ys.shape[0]):
        if train_xs[0].dot(weights) < delta:
            delta = train_xs[0].dot(weights)*train_ys[i]
        if norm(train_xs[i] > maxNorm):
            maxNorm = norm(train_xs[i])
    return maxNorm, delta

def main():
    import argparse
    import os
    from numpy.linalg import norm
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default=None, help='Training data file.')
    parser.add_argument('--train_rate', type=float, default=1, help='Training rate.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """
    train_xs, train_ys = parse_data(args.train_file)

    weights, accuracy_arr, ck = perceptron(train_xs, train_ys, args.iterations, args.train_rate)
    accuracy = test_accuracy(weights, train_xs, train_ys)

    #print(accuracy_arr)
    plt.plot(accuracy_arr)
    plt.ylabel('accuracy')


    plt.xlabel('iteration')
    plt.show()
    
    print("---------------------- result ---------------------")
    print('Train accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

    #Maximun vector norm calculation
    maxNorm = norm(train_xs[0])
    delta = train_xs[0].dot(weights)*train_ys[0]
    norm_weights = weights/norm(weights)
    for i in range(train_ys.shape[0]):
        if train_xs[i].dot(norm_weights) < delta:
            delta = train_xs[i].dot(norm_weights)*train_ys[i]
        if norm(train_xs[i] > maxNorm):
            maxNorm = norm(train_xs[i])

    print("R (maximun vector norm): ", maxNorm)
    print("δ^2 (minimun seperation): ", delta**2)
    print("(R^2)/k =", maxNorm**2/ck)




if __name__ == '__main__':
    main()
