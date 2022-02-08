import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import array
r=620


def error(x, y, x2, y2,r):
    learning_rate = 0.1
    # give appropriate number to the dimensions
    M, Din, H, H2, Dout = x.shape[1], x.shape[0], r, r, y.shape[0]
    M2 = x2.shape[1]
    # w1, w2 = np.random.rand(H,Din)*np.sqrt(2/(H+Din)), np.random.rand(Dout,H)*np.sqrt(2/(Dout+H))
    R1 = np.sqrt(6 / (H + Din))
    w1 = np.random.uniform(-R1, R1, size=(H, Din))
    R2 = np.sqrt(6 / (H + H2))
    w2 = np.random.uniform(-R2, R2, size=(H2, H))
    R3 = np.sqrt(6 / (Dout + H2))
    w3 = np.random.uniform(-R3, R3, size=(Dout, H2))
    p = 0
    training_iter = 300
    for i in range(training_iter):
        # forward prop.
        Z1 = w1.dot(x)
        A1 = np.where(Z1 <= 0, 0, Z1)
        relu1 = w2.dot(A1)
        A2 = np.where(relu1 <= 0, 0, relu1)
        relu2 = w3.dot(A2)

        softmaxAnswer = (softmax(relu2))

        AccsoftmaxAnswer = (softmax(relu2)).T
        y_hat = (AccsoftmaxAnswer == AccsoftmaxAnswer.max(axis=1)[:, None]).astype(int)
        accuracy = (np.sum(np.multiply(y_hat.T, y)) / M) * 100
        # accuracy = (y_hat.T == y).mean()
        print('TrainAccuracy', accuracy, "%")

        Cross = cross_entropy(relu2, y)
        # forward prop.for test data
        Z1T = w1.dot(x2)
        A1T = np.where(Z1T <= 0, 0, Z1T)
        relu1T = w2.dot(A1T)
        A2T = np.where(relu1T <= 0, 0, relu1T)
        relu2T = w3.dot(A2T)

        softmaxAnswer2 = softmax(relu2T)

        AccsoftmaxAnswer2 = (softmax(relu2T)).T
        y_hat2 = (AccsoftmaxAnswer2 == AccsoftmaxAnswer2.max(axis=1)[:, None]).astype(int)
        accuracy2 = (np.sum(np.multiply(y_hat2.T, y2)) / M2) * 100
        # accuracy2 = (y_hat2.T == y2).mean()
        print('TestAccuracy', accuracy2, "%")

        TestCross = cross_entropy(relu2T, y2)

        # back prop.
        dY_hat = (softmaxAnswer - y) / M
        dw3 = dY_hat.dot(A2.T)
        dA2 = (w3.T).dot(dY_hat)  # this is the backpropgation through the layer
        dZ2 = dA2 * (relu1 > 0).astype(np.int)  # or (A1>0)
        dw2 = dZ2.dot(A1.T)
        dA1 = (w2.T).dot(dZ2)
        dZ1 = dA1 * (Z1 > 0).astype(np.int)  # or (A1>0)
        dw1 = dZ1.dot(x.T)
        dX = (w1.T).dot(dZ1)

        # gd
        w1 -= (learning_rate * dw1)
        w2 -= (learning_rate * dw2)
        w3 -= (learning_rate * dw3)

        # print("w1", w1)
        # print("w2", w2)
        print("TrainError", Cross)

        # print("Lastw1", w1)
        # print("Lastw2", w2)
        print("TestError", TestCross)

        trainloss_array[p] = Cross
        testloss_array[p] = TestCross


        traincost_array[p] = accuracy
        testcost_array[p] = accuracy2
        p += 1
        print("iteration", p)
    train_errorrate_array.append(accuracy)
    test_errorrate_array.append(accuracy2)


def softmax(x):

    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0, keepdims=True)


def rfunction(x, y, x2, y2, r):
    r_array = [i for i in range(10, r, 150)]

    for i in range(10, r, 150):
        print("R neuron= ", i)
        error(x, y, x2, y2, i)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.plot(iteration_array, traincost_array, label="TrainAccuracy")
        plt.plot(iteration_array, testcost_array, label="TestAccuracy")
        plt.legend(loc="upper left")
        plt.title("R neurons " + str(i))
        plt.show()

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.plot(iteration_array, trainloss_array, label="TrainLoss")
        plt.plot(iteration_array, testloss_array, label="TestLoss")
        plt.legend(loc="upper left")
        plt.title("R neurons " + str(i))
        plt.show()

    # plt.plot(r_array, train_errorrate_array, r_array,test_errorrate_array)
    plt.xlabel('r neuron')
    plt.ylabel('accuracy')
    plt.plot(r_array, train_errorrate_array, label="Train")
    plt.plot(r_array, test_errorrate_array, label="Test")
    plt.legend(loc="upper left")
    plt.show()


def cross_entropy(X,y):

    m = y.shape[1]
    p = softmax(X)

    # loss = -np.mean(y * np.log(p + 1e-8))
    product = -(y * np.log(p + 1e-8))
    loss = np.sum(product) / m

    return loss

train_data_list = pd.read_csv('mnist_train.csv')
test_data_list = pd.read_csv('mnist_test.csv')
train_data = np.array(train_data_list)
test_data = np.array(test_data_list)



# give appropriate number to the dimensions
C = train_data
B = test_data

train_labels = train_data
test_labels = test_data
C = np.delete(C, 0, 1)
B = np.delete(B, 0, 1)
optimized_train_data=C/255
optimized_test_data=B/255
train_labels = np.delete(train_labels, np.s_[1:], 1)
test_labels = np.delete(test_labels, np.s_[1:], 1)
lr = np.arange(10)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

# # we don't want zeroes and ones in the labels neither:
# train_labels_one_hot[train_labels_one_hot==0] = 0.01
# train_labels_one_hot[train_labels_one_hot==1] = 0.99
# test_labels_one_hot[test_labels_one_hot==0] = 0.01
# test_labels_one_hot[test_labels_one_hot==1] = 0.99
x1, y1 = optimized_train_data.T , train_labels_one_hot.T
x2, y2 = optimized_test_data.T , test_labels_one_hot.T
N=300
traincost_array = np.zeros(N)
testcost_array = np.zeros(N)
trainloss_array = np.zeros(N)
testloss_array = np.zeros(N)
train_errorrate_array = array.array('d', [])
test_errorrate_array = array.array('d', [])
iteration_array = [i for i in range(N)]




# error(x1,y1,x2,y2,r)
#
# # plt.plot(iteration_array,traincost_array,iteration_array,testcost_array )
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# plt.plot(iteration_array, traincost_array, label="TrainAccuracy")
# plt.plot(iteration_array, testcost_array, label="TestAccuracy")
# plt.legend(loc="upper left")
# plt.show()
#
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.plot(iteration_array, trainloss_array, label="TrainLoss")
# plt.plot(iteration_array, testloss_array, label="TestLoss")
# plt.legend(loc="upper left")
# plt.show()
rfunction(x1, y1, x2, y2, r)