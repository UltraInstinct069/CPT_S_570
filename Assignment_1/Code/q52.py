import numpy as np
import math
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt

# loading training data
train_file = "data/fashion-mnist_train.csv"
train_data = np.loadtxt(train_file, skiprows=1, delimiter=',')

# preapring feature data
X=train_data[:,1:]
X=X/255 # normalizing

# preapring target data
y=train_data[:,0]

y=y.astype(int)

#testing data loading
test_file = "data/fashion-mnist_test.csv"
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')
X_test=test_data[:,1:]
X_test=X_test/255.0
y_test=test_data[:,0]

y_test=y_test.astype(int)

def test_perceptron(weight):
    len_test_data=len(test_data)
    num_feature=X.shape[1]
    num_classes=10
    mistake=0
    F_x_y=np.zeros((10,num_feature),dtype=float)
    res_w_dot_f=[0.0]*num_classes
    
    for i in range(len_test_data):
        # taking score for each class then take the maximum
        for k in range(num_classes):
            F_x_y[k]=X_test[i]
            value=np.sum(weight[k]*F_x_y)
            res_w_dot_f[k]=value
            F_x_y=np.zeros((10,num_feature),dtype=float) 
        
        pred_y= np.argmax(res_w_dot_f)
        if pred_y!= y_test[i]:
            mistake+=1
    acc=(len_test_data-mistake)/len_test_data
    return acc

def train_acc_perceptron(weight):
    len_train_data=len(train_data)
    num_classes=10
    num_feature=X.shape[1]
    mistake=0
    F_x_y=np.zeros((10,num_feature),dtype=float)
    res_w_dot_f=[0.0]*num_classes
    for i in range(len_train_data):
        for k in range(num_classes):
            F_x_y[k]=X[i]
            value=np.sum(weight[k]*F_x_y)
            res_w_dot_f[k]=value
            F_x_y=np.zeros((10,num_feature),dtype=float) 
            
        pred_y= np.argmax(res_w_dot_f)
        if pred_y!= y[i]:
            mistake+=1
    acc=(len_train_data-mistake)/len_train_data
    return acc

def figure_5_2_a(perc_mistakes, PA_mistakes):
    x_value = list(range(1, 51))
    plt.close()
    plt.figure(1)
    plt.plot(x_value, perc_mistakes,label='Number of Mistakes of Perceptron')
    plt.plot(x_value, PA_mistakes, label='Number of Mistakes of Passive Aggresive')
    plt.xlabel('Iterations')
    plt.ylabel('Mistakes')
    plt.legend()
    plt.savefig('5_2_A.png')
    
def figure_5_2_b(acc_train_perc,acc_train_PA,acc_test_perc, acc_test_PA):
    x_value = list(range(1, 21))
    plt.close()
    plt.figure(2)
    plt.plot(x_value, acc_train_perc,label='Accuracy of Perceptron (train)')
    plt.plot(x_value, acc_test_perc, label='Accuracy of Perceptron (test)')
    plt.plot(x_value, acc_train_PA,label='Accuracy of PA (train)')
    plt.plot(x_value, acc_test_PA, label='Accuracy of PA (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('5_2_B.png')

def figure_5_2_c(acc_train_perc,acc_train_avg_pec,acc_test_perc, acc_test_avg_pec):
    x_value = list(range(1, 21))
    plt.close()
    plt.figure(3)
    plt.plot(x_value, np.array(acc_train_perc),label='Accuracy of Perceptron (train)')
    plt.plot(x_value, np.array(acc_test_perc), label='Accuracy of Perceptron (test)')
    plt.plot(x_value, np.array(acc_train_avg_pec),label='Accuracy of Avg. Perceptron (train)')
    plt.plot(x_value, np.array(acc_test_avg_pec), label='Accuracy of Avg. Perceptron (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('5_2_C.png')

def figure_5_2_d(acc_test_perc):
    x_value = np.arange(start=0, stop=60100, step=100)
    plt.figure(4)
    plt.plot(x_value, acc_test_perc,label='Accuracy of Perceptron (test)')
    plt.xlabel('Training Data')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('5_2_D.png')

def multiclass_perceptron():
    num_feature=X.shape[1]
    num_data=X.shape[0]
    num_classes=10
    max_itr=50

    count_mistake_perc=0
    count_mistake_PA=0
    counter_avg_perc=1

    lr_perceptron=1
    lr_PA=1

    w_perceptron=np.zeros((10,num_feature),dtype=float) # 2d matrix for perceptron weight fector
    w_PA=np.zeros((10,num_feature),dtype=float)  # 2d matrix for PA weight fector
    avg_perc_cache=np.zeros((10,num_feature),dtype=float)  # 2d matrix for caching avg perc weight fector
    F_x_y=np.zeros((10,num_feature),dtype=float) # 2d matrix for placing feature
    res_w_dot_f_perc=[0.0]*num_classes # score storing per class for perc
    res_w_dot_f_PA=[0.0]*num_classes # score storing per class for PA

    # initializing variable for tracking mistakes
    mis_per_itr_perc=[0 for i in range(50)] 
    mis_per_itr_PA=[0 for i in range(50)]
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
            # calulating score for each class (perc)
            for k in range(num_classes):
                F_x_y[k]=X[i]
                value=np.sum(w_perceptron[k]*F_x_y)
                res_w_dot_f_perc[k]=value
                F_x_y=np.zeros((10,num_feature),dtype=float)
            # calulating score for each class (PA)
            for k in range(num_classes):
                F_x_y[k]=X[i]
                value=np.sum(w_PA[k]*F_x_y)
                res_w_dot_f_PA[k]=value
                F_x_y=np.zeros((10,num_feature),dtype=float)
                
            y_pred_perc=np.argmax(res_w_dot_f_perc) # predicting class label using argmax (perc)
            y_pred_PA=np.argmax(res_w_dot_f_PA) # predicting class label using argmax (PA)
            # In case of mistake update weight vetor (perc)
            if y[i] != y_pred_perc:
                # placing value F(x,y)
                F_x_y_cor=np.zeros((10,num_feature),dtype=float)
                F_x_y_cor[y[i]]=X[i] 
                S_F_x_y_cor=np.sum(F_x_y_cor*w_perceptron[y[i]])
                # placing value F(x,y')
                F_x_y_pred=np.zeros((10,num_feature),dtype=float)
                F_x_y_pred[y_pred_perc]=X[i]
                S_F_x_y_pred=np.sum(F_x_y_pred*w_perceptron[y_pred_perc])

                w_perceptron= w_perceptron + lr_perceptron*(F_x_y_cor-F_x_y_pred) # updating weight (perc)
                count_mistake_perc+=1
                avg_perc_cache=avg_perc_cache + counter_avg_perc*(F_x_y_cor-F_x_y_pred) # for average perceptron
            
            # In case of mistake update weight vetor (PA)
            if y[i] != y_pred_PA:
                # placing value F(x,y)
                F_x_y_cor=np.zeros((10,num_feature),dtype=float)
                F_x_y_cor[y[i]]=X[i]
                S_F_x_y_cor=np.sum(F_x_y_cor*w_PA[y[i]])
                # placing value F(x,y')
                F_x_y_pred=np.zeros((10,num_feature),dtype=float)
                F_x_y_pred[y_pred_PA]=X[i]
                S_F_x_y_pred=np.sum(F_x_y_pred*w_PA[y_pred_PA])

                sub_cor_pred=F_x_y_cor-F_x_y_pred
                # weight update rule: (1- (w.F(x,y)-w.F(x,y')))/ || F(x,y) - F(x,y')||^2
                lr_PA= (1-(S_F_x_y_cor-S_F_x_y_pred))/(norm(sub_cor_pred)**2) 
                w_PA= w_PA + lr_PA*sub_cor_pred
                count_mistake_PA+=1
            
            counter_avg_perc=counter_avg_perc+1
        
        
        # for ques b and c calculating accuracy and storing them
        if index<20:
            acc_per_itr_perc_test[index]=test_perceptron(w_perceptron)
            acc_per_itr_PA_test[index]= test_perceptron(w_PA)
            avg_perc_weight_per_itr= w_perceptron-(1/counter_avg_perc)*(avg_perc_cache)
            #print(avg_perc_weight_per_itr)  
            acc_per_itr_avg_perc_test[index]= test_perceptron(avg_perc_weight_per_itr)
            acc_per_itr_perc_train[index]=train_acc_perceptron(w_perceptron)
            acc_per_itr_PA_train[index]=train_acc_perceptron(w_PA)
            acc_per_itr_avg_perc_train[index]=train_acc_perceptron(avg_perc_weight_per_itr)  
        
        print('Perc: ',count_mistake_perc)
        print('PA: ',count_mistake_PA)
        mis_per_itr_perc[index]=count_mistake_perc
        mis_per_itr_PA[index]=count_mistake_PA

    print('Perc: ',mis_per_itr_perc)
    print('PA: ',mis_per_itr_PA)
    print('ACC(perc) test: ',acc_per_itr_perc_test)
    print('ACC(PA) test: ',acc_per_itr_PA_test)
    print('ACC(avg_perc) test: ',acc_per_itr_avg_perc_test)
    print('ACC(perc) train: ',acc_per_itr_perc_train)
    print('ACC(PA) train: ',acc_per_itr_PA_train)
    figure_5_2_a(mis_per_itr_perc,mis_per_itr_PA)
    figure_5_2_b(acc_per_itr_perc_train,acc_per_itr_PA_train,acc_per_itr_perc_test,acc_per_itr_PA_test)
    figure_5_2_c(acc_per_itr_perc_train,acc_per_itr_avg_perc_train,acc_per_itr_perc_test,acc_per_itr_avg_perc_test)

def learning_curve_5_2_d():
    num_feature=X.shape[1]
    num_data=X.shape[0]
    num_classes=10
    max_itr=20

    count_mistake_perc=0
    count_mistake_PA=0
    counter_avg_perc=1

    lr_perceptron=1
    lr_PA=1

    w_perceptron=np.zeros((10,num_feature),dtype=float) # 2d matrix for perceptron weight fector
    w_PA=np.zeros((10,num_feature),dtype=float)  # 2d matrix for PA weight fector
    F_x_y=np.zeros((10,num_feature),dtype=float) # 2d matrix for placing feature
    res_w_dot_f_perc=[0.0]*num_classes # score storing per class for perc
    res_w_dot_f_PA=[0.0]*num_classes # score storing per class for PA

    data_val=np.arange(start=0, stop=60100, step=100)
    acc_per_itr_perc=[0 for i in range(len(data_val))]
    acc_per_itr_PA=[0 for i in range(len(data_val))]

    for d_val in range(len(data_val)):
        w_perceptron=np.zeros((10,num_feature),dtype=float)
        print('Step : ',d_val)
        for index in range(max_itr):
            #print('Iteration : ',index)
            count_mistake_perc=0
            count_mistake_PA=0
            num_data=data_val[d_val]
            for i in range(num_data):
                # calulating score for each class (perc)
                for k in range(num_classes):
                    F_x_y[k]=X[i]
                    value=np.sum(w_perceptron[k]*F_x_y)
                    res_w_dot_f_perc[k]=value
                    F_x_y=np.zeros((10,num_feature),dtype=float)
                # calulating score for each class (PA)
                # for k in range(num_classes):
                #     F_x_y[k]=X[i]
                #     value=np.sum(w_PA[k]*F_x_y)
                #     res_w_dot_f_PA[k]=value
                #     F_x_y=np.zeros((10,num_feature),dtype=float)
                    
                y_pred_perc=np.argmax(res_w_dot_f_perc) # predicting class label using argmax (perc)
                #y_pred_PA=np.argmax(res_w_dot_f_PA) # predicting class label using argmax (PA)
                # In case of mistake update weight vetor (perc)
                if y[i] != y_pred_perc:
                    # placing value F(x,y)
                    F_x_y_cor=np.zeros((10,num_feature),dtype=float)
                    F_x_y_cor[y[i]]=X[i] 
                    S_F_x_y_cor=np.sum(F_x_y_cor*w_perceptron[y[i]])
                    # placing value F(x,y')
                    F_x_y_pred=np.zeros((10,num_feature),dtype=float)
                    F_x_y_pred[y_pred_perc]=X[i]
                    S_F_x_y_pred=np.sum(F_x_y_pred*w_perceptron[y_pred_perc])

                    w_perceptron= w_perceptron + lr_perceptron*(F_x_y_cor-F_x_y_pred) # updating weight (perc)
                    count_mistake_perc+=1
                
                # In case of mistake update weight vetor (PA)
                # if y[i] != y_pred_PA:
                #     # placing value F(x,y)
                #     F_x_y_cor=np.zeros((10,num_feature),dtype=float)
                #     F_x_y_cor[y[i]]=X[i]
                #     S_F_x_y_cor=np.sum(F_x_y_cor*w_PA[y[i]])
                #     # placing value F(x,y')
                #     F_x_y_pred=np.zeros((10,num_feature),dtype=float)
                #     F_x_y_pred[y_pred_PA]=X[i]
                #     S_F_x_y_pred=np.sum(F_x_y_pred*w_PA[y_pred_PA])

                #     sub_cor_pred=F_x_y_cor-F_x_y_pred
                #     w_PA= w_PA + lr_PA*sub_cor_pred
                #     # weight update rule: (1- (w.F(x,y)-w.F(x,y')))/ || F(x,y) - F(x,y')||^2
                #     lr_PA= (1-(S_F_x_y_cor-S_F_x_y_pred))/(norm(sub_cor_pred)**2) 
                #     count_mistake_PA+=1
                
                #counter_avg_perc=counter_avg_perc+1
        
        
        #acc_per_itr_perc[d_val]=test_perceptron(w_perceptron)
        #acc_per_itr_PA[d_val]= test_perceptron(w_PA)
        #print(acc_per_itr_perc)
        #print(acc_per_itr_PA)
          
    print(acc_per_itr_perc)
    figure_5_2_d(acc_per_itr_perc)

if __name__=="__main__":
  multiclass_perceptron()
  learning_curve_5_2_d()