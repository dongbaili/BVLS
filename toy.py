from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from algorithms import WeightedLasso
import numpy as np
model = LogisticRegression()
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

mean_0 = 0
mean_1 = 1
std_dev = 0.3
def gen_spurious_feature(X_train, X_test, y_train, y_test, shift_degree):
    spu_col_tr = []
    g_train = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            if np.random.random() < shift_degree:
                sample = np.random.normal(mean_0, std_dev)  
                g = 0
            else :
                sample = np.random.normal(mean_1, std_dev)
                g = 1
        else:
            if np.random.random() < shift_degree:
                sample = np.random.normal(mean_1, std_dev) 
                g = 3
            else:
                sample = np.random.normal(mean_0, std_dev)
                g = 2
        g_train.append(g)
        spu_col_tr.append(sample)

    spu_col_te = []
    g_test = []
    for i in range(len(y_test)):
        # 0.5概率为 以1为均值的正态分布
        # 0.5概率为 以0为均值的正态分布
        if np.random.random() < 0.5:
            sample = np.random.normal(mean_0, std_dev) 
            if y_test[i] == 0:
                g = 0
            else:
                g = 2
        else:
            sample = np.random.normal(mean_1, std_dev)
            if y_test[i] == 0:
                g = 1
            else:
                g = 3

        g_test.append(g)
        spu_col_te.append(sample)

    noisy_feature = []
    for i in range(len(y_train)):
        if np.random.random() < 0.5:
            sample = np.random.normal(0, std_dev) 
        else:
            sample = np.random.normal(1, std_dev)
        noisy_feature.append(sample)

    X_train_s = np.hstack([X_train, np.array(spu_col_tr).reshape(-1, 1), np.array(noisy_feature[:len(y_train)]).reshape(-1, 1)])
    X_test_s = np.hstack([X_test, np.array(spu_col_te).reshape(-1, 1), np.array(noisy_feature[:len(y_test)]).reshape(-1, 1)])
    # X_train_s = np.hstack([X_train, np.array(spu_col_tr).reshape(-1, 1)])
    # X_test_s = np.hstack([X_test, np.array(spu_col_te).reshape(-1, 1)])

    return X_train_s, X_test_s, g_train, g_test

def check_weight_param(shift_degree):
    X_train_s, X_test_s, g_train, g_test = gen_spurious_feature(X_train, X_test, y_train, y_test, shift_degree)

    repeat_times = 100
    for i in range(repeat_times):
        random_indices = np.random.choice(len(X_train_s), len(X_train_s), replace=True)
        X_train_boot = X_train_s[random_indices]
        y_train_boot = y_train[random_indices]
        model.fit(X_train_boot, y_train_boot)
        if i == 0:
            total_coef = model.coef_[0]
        else:
            total_coef = np.vstack([total_coef, model.coef_[0]])
    
    weight = np.var(total_coef, axis=0)
    weight = weight**2 / np.sum(weight**2)

    model1 = WeightedLasso(input_dim=X_train_s.shape[1], output_dim=1, alpha=0)
    model1.fit(X_train_s, y_train)
    model2 = WeightedLasso(input_dim=X_train_s.shape[1], output_dim=1, alpha=0.05, weight=weight)
    model2.fit(X_train_s, y_train)

    return weight, model1.coef_()[-2], model2.coef_()[-2]

def main(shift_degree):
    X_train_s, X_test_s, g_train, g_test = gen_spurious_feature(X_train, X_test, y_train, y_test, shift_degree)

    repeat_times = 100
    for i in range(repeat_times):
        random_indices = np.random.choice(len(X_train_s), len(X_train_s), replace=True)
        X_train_boot = X_train_s[random_indices]
        y_train_boot = y_train[random_indices]
        model.fit(X_train_boot, y_train_boot)
        if i == 0:
            total_coef = model.coef_[0]
        else:
            total_coef = np.vstack([total_coef, model.coef_[0]])
    
    # Test worst accuracy on ERM
    model1 = WeightedLasso(input_dim=X_train_s.shape[1], output_dim=1, alpha=0)
    model1.fit(X_train_s, y_train)
    g_test = np.array(g_test)
    worst_acc = 1
    for group in [0, 1, 2, 3]:
        yp = model1.predict(X_test_s[g_test == group])
        if accuracy_score(yp, y_test[g_test==group]) < worst_acc:
            worst_acc = accuracy_score(yp, y_test[g_test==group])

    # Test worst accuracy on weighted lasso
    weight = np.var(total_coef, axis=0)
    weight = weight**2 / np.sum(weight**2)

    model2 = WeightedLasso(input_dim=X_train_s.shape[1], output_dim=1, alpha=0.05, weight=weight)
    model2.fit(X_train_s, y_train)
    worst_acc_lasso = 1
    for group in [0, 1, 2, 3]:
        yp = model2.predict(X_test_s[g_test == group])
        if accuracy_score(yp, y_test[g_test==group]) < worst_acc_lasso:
            worst_acc_lasso = accuracy_score(yp, y_test[g_test==group])
    
    print(shift_degree, "ERM:", worst_acc, "Weighted Lasso:", worst_acc_lasso)
    var_frac = np.var(total_coef, axis=0)[-2] / np.sum(np.var(total_coef, axis=0))

    return var_frac, worst_acc, worst_acc_lasso

def test1():
    repeat = 10
    var_list = []
    wa_list = []
    wa_l_list = []
    # degrees = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    degrees = [0.5, 0.6, 0.7, 0.8, 0.9]
    for degree in degrees:
        var_list_tmp = []
        wa_list_tmp = []
        wa_l_list_tmp = []
        for i in range(repeat):
            var, worst_acc, worst_acc_lasso = main(degree) 
            var_list_tmp.append(var)
            wa_list_tmp.append(worst_acc)
            wa_l_list_tmp.append(worst_acc_lasso)
        var = np.mean(var_list_tmp)
        worst_acc = np.mean(wa_list_tmp)
        worst_acc_lasso = np.mean(wa_l_list_tmp)
        var_list.append(var)
        wa_list.append(worst_acc)
        wa_l_list.append(worst_acc_lasso)
    
    # show them in one plot
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=240)
    
    ax1.plot(degrees, var_list, label='Variance of the Spurious Feature / Sum of Variance')
    ax1.legend()

    ax2.plot(degrees, wa_list, label='ERM')
    ax2.plot(degrees, wa_l_list, label='weighted lasso')
    ax2.set_xlabel('Shift extent')
    ax2.legend()

    plt.savefig('toy_Weight_Lasso.png')

def test2():
    repeat = 10
    # degrees = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    degrees = [0.5, 0.6, 0.7, 0.8, 0.9]
    es_list = []
    ls_list = []
    for degree in degrees:
        weight_list_tmp = []
        es_list_tmp = []
        ls_list_tmp = []
        for i in range(repeat):
            weight, Es, Ls = check_weight_param(degree) 
            weight_list_tmp.append(weight)
            es_list_tmp.append(Es)
            ls_list_tmp.append(Ls)
        weight = np.mean(weight_list_tmp, axis=0)
        plt.figure(dpi = 240)
        plt.bar(range(len(weight)), weight)
        plt.xlabel('Feature index')
        plt.ylabel('Weight')
        plt.savefig(f'toy_weight_{degree}.png')
        Es = np.mean(es_list_tmp) 
        Ls = np.mean(ls_list_tmp)
        es_list.append(Es)
        ls_list.append(Ls)

    plt.figure(dpi = 240)
    plt.plot(degrees, es_list, label='ERM spurious feature coefficient')
    plt.plot(degrees, ls_list, label='Weighted Lasso spurious feature coefficient')
    plt.legend()
    plt.xlabel('Shift extent')

    plt.savefig('toy_weight_param.png')

if __name__ == '__main__':
    np.random.seed(10)
    test2()