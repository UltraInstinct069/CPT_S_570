import numpy as np
import math
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
from regex import D
from sympy import lerchphi
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# loading training data
train_file = "data/fashion-mnist_train.csv"
train_data = np.loadtxt(train_file, skiprows=1, delimiter=',')
# preapring feature data
X=train_data[:,1:]
X=X/255
x_limit=48000
x_train=X[:x_limit,:]
x_val=X[x_limit:,:]
#preparing target data
y=train_data[:,0]

y=y.astype(int)
y_train=y[:x_limit]
y_val=y[x_limit:]
#testing data loading
test_file = "data/fashion-mnist_test.csv"
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')
x_test=test_data[:,1:]
x_test=x_test/255.0
y_test=test_data[:,0]

y_test=y_test.astype(int)

def svm_linear(c_param):
    test_accu = [0] * len(c_param)
    val_acc = [0] * len(c_param)
    training_acc = [0] * len(c_param)
    for c in range(len(c_param)):
        print('C: ',c_param[c])
        clf = svm.SVC(kernel='linear', C = c_param[c])
        clf.fit(x_train,y_train)
        train_pred=clf.predict(x_train)
        training_acc[c]=accuracy_score(y_train,train_pred)
        print('Training Acc: ',training_acc[c])
        val_pred=clf.predict(x_val)
        val_acc[c]=accuracy_score(y_val,val_pred)
        print('Validation Acc: ',val_acc[c])
        test_pred=clf.predict(x_test)
        test_accu[c]=accuracy_score(y_test,test_pred)
        print('Testing Acc: ',test_accu[c])
        
        print('Support vectors: ',clf.n_support_)
        print('Number of Support vectors: ',np.sum(clf.n_support_))
    
    figure_1_1_a(training_acc,val_acc,test_accu)

def svm_linear_best(c_val):
    print('Using the best C:',c_val)
    clf = svm.SVC(kernel='linear', C = c_val)
    clf.fit(X,y)
    test_pred=clf.predict(x_test)
    print('Testing Acc: ',accuracy_score(y_test,test_pred))
    print('Number of Support vectors: ',np.sum(clf.n_support_))
    cm = confusion_matrix(y_test,test_pred)
    print(cm)
    sns.set(font_scale=1.4)
    ax = sns.heatmap(cm, annot=True, cmap='Blues',annot_kws={"size": 16},fmt='g')

    ax.set_title('Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0','1','2','3','4','5','6','7','8','9'])
    ax.yaxis.set_ticklabels(['0','1','2','3','4','5','6','7','8','9'])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig('1_1_b_cft.png')

def svm_polynomial(deg_param):
    test_accu = [0] * len(deg_param)
    Val_acc = [0] * len(deg_param)
    training_acc = [0] * len(deg_param)
    for c in range(len(deg_param)):
        print('deg: ',deg_param[c])
        clf = svm.SVC(kernel='poly', C = 0.1,degree=deg_param[c])
        clf.fit(x_train,y_train)
        train_pred=clf.predict(x_train)
        print('Training Acc: ',accuracy_score(y_train,train_pred))
        val_pred=clf.predict(x_val)
        print('Validation Acc: ',accuracy_score(y_val,val_pred))
        test_pred=clf.predict(x_test)
        print('Testing Acc: ',accuracy_score(y_test,test_pred))
        
        print('Support vectors: ',clf.n_support_)
        print('Number of Support vectors: ',np.sum(clf.n_support_))

def figure_1_1_a(train_acc,val_acc,test_acc):
    iterations_1 = np.array(['0.0001','0.001','0.01','0.1','1','10','100'])
    plt.figure(1)
    plt.plot(iterations_1, train_acc,label='Training Accuracy')
    plt.plot(iterations_1, val_acc, label='Validation Accuracy')
    plt.plot(iterations_1, test_acc, label='Testing Accuracy')
    plt.xlabel('C-Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Q1_1_at.png')

def figure_1_1_c(train_acc,val_acc,test_acc):
    iterations_1 = np.array(['0.0001','0.001','0.01','0.1','1','10','100'])
    plt.figure(1)
    plt.plot(iterations_1, train_acc,label='Training Accuracy')
    plt.plot(iterations_1, val_acc, label='Validation Accuracy')
    plt.plot(iterations_1, test_acc, label='Testing Accuracy')
    plt.xlabel('C-Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Q1_1_at.png')

svm_linear([0.0001,0.001,0.01,0.1,1,10,100])
svm_linear_best(0.1)
svm_polynomial([2,3,4])