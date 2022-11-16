import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Split the dataset based on a continuos value
def split_tree_cont(col, value, dataset):
    left=list()
    right = list()
    for row in dataset:
        if row[col] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# entropy calculation
def calc_entropy(groups, classes):
    # counting samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    ent = 0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
        if p > 0:
            score = (p * math.log(p, 2))
        # weight the group score by its relative size i.e Entrpy gain
        ent -= (score * (size / n_instances))
    return ent


# calculating the split
def make_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    col_id, sp_value, entrop_at_split, split_groups = 999, 999, 1, None
    features = {}
    for index in range(len(dataset[0])-1):
        features[index] = set([example[index] for example in dataset]) 
        my_list = list(sorted(features[index]))
        split_val= set()
        # calculatint the split threshold (used median)
        for val in range(len(my_list)):
            if val==len(my_list)-1:
                break
            split_val.add(my_list[val]+(my_list[val+1]-my_list[val])/2)
        features[index]=split_val    
        for row in features[index]:
            groups = split_tree_cont(index, row, dataset)
            ent = calc_entropy(groups, class_values)
            if ent < entrop_at_split:
                col_id, sp_value, entrop_at_split, split_groups = index, row, ent, groups
    return {'index': col_id, 'value': sp_value, 'groups': split_groups}


# calculatin the majority class of a split
def majority_class(group):
    maj_count = [row[-1] for row in group]
    return max(set(maj_count), key=maj_count.count)


# building the tree recursively
def tree_recursive(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = majority_class(left + right)
        return
    # check for max depth
    if depth == max_depth:
        node['left'], node['right'] = majority_class(left), majority_class(right)
        return
    # left child processing. In case doesnot have many leaves select majority class otherwise keep spliting
    if len(left) <= min_size:
        node['left'] = majority_class(left)
    else:
        node['left'] = make_split(left)
        tree_recursive(node['left'], max_depth, min_size, depth + 1)
    # right child processing. In case doesnot have many leaves select majority class otherwise keep spliting
    if len(right) <= min_size:
        node['right'] = majority_class(right)
    else:
        node['right'] = make_split(right)
        tree_recursive(node['right'], max_depth, min_size, depth + 1)

# constructing the ID3
def tree_build(train, max_depth, min_size):
    root_node = make_split(train)
    tree_recursive(root_node, max_depth, min_size, 1)
    return root_node

# printing the tree node by node
def show_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[ATTRIBUTE[%s] <= %.50s]' % ((depth * '\t', (node['index']), node['value'])))
        show_tree(node['left'], depth + 1)
        show_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


# predicting from the tree
def pred_data(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return pred_data(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return pred_data(node['right'], row)
        else:
            return node['right']


def cal_accuracy(data):
    mistake=0
    for i in range(len(data)):
        if pred_data(tree,data[i]) != data[i][30]:
            mistake=mistake+1
    acc=(len(data)-mistake)/len(data)
    return acc

def figure_3_1_d(val_acc,test_acc,dep):
    iterations_1 = list(range(1, dep+1))
    plt.close()
    plt.figure(1)
    plt.plot(iterations_1, val_acc,label='Validation Accuracy')
    plt.plot(iterations_1, test_acc, label='Testing Accuracy')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('Q3d.png')

if __name__=="__main__":
    br_cancer_data=pd.read_csv('data/data.csv',delimiter=',')
    print(br_cancer_data.shape)
    #encoding malignant as 1 and benign as 0
    encode_colums = {"diagnosis":    {"M": 1, "B": 0}}
    br_cancer_data=br_cancer_data.replace(encode_colums)
    # removing id and dummy column
    br_cancer_data=br_cancer_data.iloc[:,1:-1]
    #transfering diagnosis column to last
    diagnosis_col=br_cancer_data.pop('diagnosis')
    br_cancer_data.insert(br_cancer_data.shape[1],'diagnosis',diagnosis_col)
    x_train=br_cancer_data.iloc[:399,:].values.tolist()
    x_val=br_cancer_data.iloc[399:456,:].values.tolist()
    x_test=br_cancer_data.iloc[456:,:].values.tolist()
    tree = tree_build(x_train, -1, 1)
    print('Without Pruning:')
    print('Validation Accuracy:',cal_accuracy(x_val))
    print('Testing Accuracy: ',cal_accuracy(x_test))
    depth_range=10
    print('With Pruning:')
    val_acc=[0.0]*depth_range
    test_acc=[0.0]*depth_range
    for d in range(depth_range):
        tree = tree_build(x_train, d+1, 1)
        val_acc[d]=cal_accuracy(x_val)
        test_acc[d]=cal_accuracy(x_test)
        print('Depth: ',d+1,' Validation Accuracy: ',val_acc[d],' Testing Accuracy:',test_acc[d])

    figure_3_1_d(val_acc,test_acc,depth_range)
    best_depth=5
    print('Building tree with best depth: ',best_depth)
    tree = tree_build(x_train, best_depth, 1)
    print('Validation Accuracy:',cal_accuracy(x_val))
    print('Testing Accuracy: ',cal_accuracy(x_test))

