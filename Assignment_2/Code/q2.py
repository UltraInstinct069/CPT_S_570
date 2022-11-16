import numpy as np
import math
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt

train_file = "data/fashion-mnist_train.csv"
train_data = np.loadtxt(train_file, skiprows=1, delimiter=',')

#testing data loading
test_file = "data/fashion-mnist_test.csv"
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')

X=train_data[:,1:]
X=X/255

X_train=X[:48000,:]
y_train=train_data[:48000,0]
y_train=y_train.astype(int)

X_val=X[48000:,:]
y_val=train_data[48000:,0]
y_val=y_val.astype(int)

X_test=test_data[:,1:]
X_test=X_test/255.0
y_test=test_data[:,0]
y_test=y_test.astype(int)
print('Train Data: ',X_train.shape, y_train.shape)
print('Test Data: ',X_test.shape, y_test.shape)

def kernalized_perceptron():
    num_feature=X_train.shape[1]#
    num_data=X_train.shape[0] #
    num_classes=10
    max_itr=5
    degree=[2,3,4]
    # kernal_train=(1+np.matmul(X_train,np.transpose(X_train)))**2
    alpha_value=np.zeros((10,num_data),dtype=float)
    res_a_dot_kernal=[0.0]*num_classes
    # print('Kernal Shape: ',kernal_train.shape, 'Alpha Vactor: ',alpha_value.shape)
    for deg_ind in range(len(degree)):
        print('Degree: ',degree[deg_ind])
        kernal_train=(1+np.matmul(X_train,np.transpose(X_train)))**degree[deg_ind]
        alpha_value=np.zeros((10,num_data),dtype=float)
        training_acc=[0.0]*5
        validation_acc=[0.0]*5
        testing_acc=[0.0]*5
        for index in range(max_itr):
            print('Iteration : ',index)
            mistake=0
            for i in range(num_data):
                for k in range(num_classes):
                    res_a_dot_kernal[k]=np.sum(alpha_value[k]*kernal_train[i])
                
                y_pred=np.argmax(res_a_dot_kernal)
                if y_pred != y_train[i]:
                    alpha_value[y_pred][i]=alpha_value[y_pred][i]-1
                    alpha_value[y_train[i]][i]=alpha_value[y_train[i]][i]+1
                    mistake+=1
        # test_kernalized_perceptron(X_test,y_test,alpha_value)
            training_acc[index]=test_kernalized_perceptron(X_train,y_train,alpha_value,degree[deg_ind])
            validation_acc[index]=test_kernalized_perceptron(X_val,y_val,alpha_value,degree[deg_ind])
            testing_acc[index]=test_kernalized_perceptron(X_test,y_test,alpha_value,degree[deg_ind])
            print("Training Accuracy: ", (len(X_train)-training_acc[index])/len(X_train) )
            print("Validation Accuracy: ",(len(X_val)-validation_acc[index])/len(X_val) )
            print("Testing Accuracy: ",(len(X_test)-testing_acc[index])/len(X_test))
        
        print("Training Mistakes: ",training_acc )
        print("Validation Mistakes: ",validation_acc )
        print("Testing Mistakes: ",testing_acc)

def test_kernalized_perceptron(data,label,alpha,deg):
    num_data=data.shape[0] #
    num_classes=10
    kernal_test=(1+np.matmul(data,np.transpose(X_train)))**deg
    res_a_dot_kernal=[0.0]*num_classes
    mistake=0
    for i in range(num_data):
            for k in range(num_classes):
                res_a_dot_kernal[k]=np.sum(alpha[k]*kernal_test[i])
            y_pred=np.argmax(res_a_dot_kernal)
            if y_pred != label[i]:
                mistake+=1
    return mistake

kernalized_perceptron()