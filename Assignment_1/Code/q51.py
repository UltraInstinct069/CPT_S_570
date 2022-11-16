import numpy as np
import math
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
from regex import D
from sympy import lerchphi

# loading training data
train_file = "data/fashion-mnist_train.csv"
train_data = np.loadtxt(train_file, skiprows=1, delimiter=',')
# preapring feature data
X=train_data[:,1:]
X=X/255
y=train_data[:,0]
#odd labels as -1 and even labels as 1
y[y%2!=0]=-1
y[y%2==0]=1
y=y.astype(int)
#testing data loading
test_file = "data/fashion-mnist_test.csv"
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')
X_test=test_data[:,1:]
X_test=X_test/255.0
y_test=test_data[:,0]
#odd labels as -1 and even labels as 1
y_test[y_test%2!=0]=-1
y_test[y_test%2==0]=1
y_test=y_test.astype(int)

def test_perceptron(weight):
    len_test_data=len(test_data)
    mistake=0
    for i in range(len_test_data):
        pred_y= -1 if np.sum(weight*X_test[i])<=0 else 1
        if pred_y!= y_test[i]:
            mistake+=1
    acc=(len_test_data-mistake)/len_test_data
    return acc

def train_acc_perceptron(weight):
    len_train_data=len(train_data)
    mistake=0
    for i in range(len_train_data):
        pred_y= -1 if np.sum(weight*X[i])<=0 else 1
        if pred_y!= y[i]:
            mistake+=1
    acc=(len_train_data-mistake)/len_train_data
    return acc

def figure_5_1_a(perc_mistakes, PA_mistakes):
    x_value = list(range(1, 51))
    plt.figure(1)
    plt.plot(x_value, perc_mistakes,label='Number of Mistakes of Perceptron')
    plt.plot(x_value, PA_mistakes, label='Number of Mistakes of Passive Aggresive')
    plt.xlabel('Iterations')
    plt.ylabel('Mistakes')
    plt.legend()
    #plt.show()
    plt.savefig('5_1_A.png')
    
def figure_5_1_b(acc_train_perc,acc_train_PA,acc_test_perc, acc_test_PA):
    x_value = list(range(1, 21))
    plt.figure(2)
    plt.plot(x_value, acc_train_perc,label='Accuracy of Perceptron (train)')
    plt.plot(x_value, acc_test_perc, label='Accuracy of Perceptron (test)')
    plt.plot(x_value, acc_train_PA,label='Accuracy of PA (train)')
    plt.plot(x_value, acc_test_PA, label='Accuracy of PA (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig('5_1_B.png')

def figure_5_1_c(acc_train_perc,acc_train_avg_pec,acc_test_perc, acc_test_avg_pec):
    x_value = list(range(1, 21))
    plt.figure(3)
    plt.plot(x_value, np.array(acc_train_perc),label='Accuracy of Perceptron (train)')
    plt.plot(x_value, np.array(acc_test_perc), label='Accuracy of Perceptron (test)')
    plt.plot(x_value, np.array(acc_train_avg_pec),label='Accuracy of Avg. Perceptron (train)')
    plt.plot(x_value, np.array(acc_test_avg_pec), label='Accuracy of Avg. Perceptron (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig('5_1_C.png')

def figure_5_1_d(acc_test_perc, acc_test_PA):
    x_value = np.arange(start=0, stop=60100, step=100)
    plt.figure(4)
    plt.plot(x_value, acc_test_perc,label='Accuracy of Perceptron (test)')
    plt.plot(x_value, acc_test_PA, label='Accuracy of PA (test)')
    plt.xlabel('Training Data')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig('5_1_D.png')

def binary_classification():
    num_feature=X.shape[1]
    num_data=X.shape[0]
    max_itr=50

    count_mistake_perc=0
    count_mistake_PA=0
    counter_avg_perc=1

    lr_perceptron=1
    lr_PA=1

    w_perceptron=[0.0]*num_feature
    w_PA=[0.0]*num_feature
    avg_perc_cache=[0.0]*num_feature
    # initializing variable for tracking mistakes
    mis_per_itr_perc=[0 for i in range(50)] 
    mis_per_itr_PA=[0 for i in range(50)]

    avg_perc_weight_per_itr=[0 for i in range(50)]
    # initializing variable for tracking accuracy
    acc_per_itr_perc_test=[0 for i in range(20)]
    acc_per_itr_PA_test=[0 for i in range(20)]
    acc_per_itr_perc_train=[0 for i in range(20)]
    acc_per_itr_PA_train=[0 for i in range(20)]
    acc_per_itr_avg_perc_train=[0 for i in range(20)]
    acc_per_itr_avg_perc_test=[0 for i in range(20)]
    for index in range(max_itr):
        print('Iteration : ',index)
        count_mistake_perc=0
        count_mistake_PA=0
        for i in range(num_data):
            sum_w_x_perc=np.sum(w_perceptron*X[i]) #calucluating wx 
            pred_y_perc= x = -1 if sum_w_x_perc<=0 else 1 # predicting based on wx
            sum_w_x_PA=np.sum(w_PA*X[i])
            pred_y_PA=x = -1 if sum_w_x_PA<=0 else 1
            if pred_y_perc != y[i]:
                count_mistake_perc=count_mistake_perc+1
                w_perceptron= w_perceptron + lr_perceptron*y[i]*X[i]    #perceptron weight update
                avg_perc_cache=avg_perc_cache + y[i]*counter_avg_perc * X[i] # avg perc weight caching
            if pred_y_PA != y[i]:
                count_mistake_PA=count_mistake_PA+1
                lr_PA=(1-(y[i]*sum_w_x_PA))/(norm(X[i])**2) # learning rate for PA
                w_PA=w_PA + lr_PA*y[i]*X[i] #PA weight update
            counter_avg_perc=counter_avg_perc+1
        #for 5.1.2 and 5.1.3
        if index<20:
            acc_per_itr_perc_test[index]=test_perceptron(w_perceptron)
            acc_per_itr_PA_test[index]= test_perceptron(w_PA)
            avg_perc_weight_per_itr= w_perceptron-(1/counter_avg_perc)*(avg_perc_cache)
            acc_per_itr_avg_perc_test[index]= test_perceptron(avg_perc_weight_per_itr)
            acc_per_itr_perc_train[index]=train_acc_perceptron(w_perceptron)
            acc_per_itr_PA_train[index]=train_acc_perceptron(w_PA)
            acc_per_itr_avg_perc_train[index]=train_acc_perceptron(avg_perc_weight_per_itr)
        #print('Mistakes: ',count_mistake)
        mis_per_itr_perc[index]=count_mistake_perc
        mis_per_itr_PA[index]=count_mistake_PA
        
    print('Mistakes(perc): ',mis_per_itr_perc)
    print('Mistakes(PA): ',mis_per_itr_PA)
    print('ACC(perc) test: ',acc_per_itr_perc_test)
    print('ACC(PA) train: ',acc_per_itr_PA_test)
    print('ACC(avg_perc) test: ',acc_per_itr_avg_perc_test)
    print('ACC(perc) train: ',acc_per_itr_perc_train)
    print('ACC(PA) train: ',acc_per_itr_PA_train)
    figure_5_1_a(mis_per_itr_perc,mis_per_itr_PA)
    figure_5_1_b(acc_per_itr_perc_train,acc_per_itr_PA_train,acc_per_itr_perc_test,acc_per_itr_PA_test)
    figure_5_1_c(acc_per_itr_perc_train,acc_per_itr_avg_perc_train,acc_per_itr_perc_test,acc_per_itr_avg_perc_test)

def learning_curve_5_1_d():
    num_feature=X.shape[1]
    num_data=0
    max_itr=20

    count_mistake_perc=0
    count_mistake_PA=0
    counter_avg_perc=1

    lr_perceptron=1
    lr_PA=1

    w_perceptron=[0.0]*num_feature
    w_PA=[0.0]*num_feature
    avg_perc_cache=[0.0]*num_feature

    mis_per_itr_perc=[0 for i in range(50)] 
    mis_per_itr_PA=[0 for i in range(50)]

    avg_perc_weight_per_itr=[0 for i in range(50)]
    data_val=np.arange(start=0, stop=60100, step=100) # dividing data by 100 increments
    acc_per_itr_perc=[0 for i in range(len(data_val))]
    acc_per_itr_PA=[0 for i in range(len(data_val))]

    for d_val in range(len(data_val)):
        print('Step : ',d_val)
        for index in range(max_itr):
            
            count_mistake_perc=0
            count_mistake_PA=0
            num_data=data_val[d_val]
            for i in range(num_data):
                sum_w_x_perc=np.sum(w_perceptron*X[i])
                pred_y_perc= x = -1 if sum_w_x_perc<=0 else 1
                sum_w_x_PA=np.sum(w_PA*X[i])
                pred_y_PA=x = -1 if sum_w_x_PA<=0 else 1
                if pred_y_perc != y[i]:
                    count_mistake_perc=count_mistake_perc+1
                    w_perceptron= w_perceptron + lr_perceptron*y[i]*X[i]
                    avg_perc_cache=avg_perc_cache + y[i]*counter_avg_perc * X[i]
                if pred_y_PA != y[i]:
                    count_mistake_PA=count_mistake_PA+1
                    lr_PA=(1-(y[i]*sum_w_x_PA))/(norm(X[i])**2)
                    w_PA=w_PA + lr_PA*y[i]*X[i]
                
                counter_avg_perc=counter_avg_perc+1
        acc_per_itr_perc[d_val]=test_perceptron(w_perceptron)
        acc_per_itr_PA[d_val]= test_perceptron(w_PA)


          
    print(acc_per_itr_perc)
    print(acc_per_itr_PA)
    figure_5_1_d(acc_per_itr_perc,acc_per_itr_PA)

if __name__=="__main__":
  binary_classification()
  learning_curve_5_1_d()
  