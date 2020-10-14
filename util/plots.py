import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_time(time, name, title, hp_name, hyper_parameters, path=None):
    fig = plt.figure()
    plt.plot(hyper_parameters, time, label='time for one epoch')
    plt.ylabel('time per epoch')
    plt.xlabel(hp_name)
    plt.title(title)
    plt.legend()

    fig.savefig(path+name) if path else fig.savefig('./figures/'+name)

def plot_acc(history, name, title, with_val=True, path=None):
    fig = plt.figure()
    plt.plot(history['accuracy'], label='train accuracy')
    if with_val:
        plt.plot(history['val_accuracy'], label='validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title(title)
    plt.legend(loc="lower right")

    if path:
        fig.savefig(path + name)
    else:
        fig.savefig('./figures/' + name)

def plot_accs(accs, name, title, hp_name, hyper_parameters, train=True, path=None,):
    fig = plt.figure()
    for i, acc in enumerate(accs):
        hyper_parameter = hyper_parameters[i]
        plt.plot(acc, label=hp_name+" = "+str(hyper_parameter))
    plt.ylabel('train_accuracy') if train else plt.ylabel('val_accuracy')
    plt.xlabel('epochs')
    plt.title(title)
    plt.legend(loc="lower right")

    fig.savefig(path+name) if path else fig.savefig('./figures/'+name)
        

def plot_loss(history, name, loss, title, with_val=True , path=None):
    fig = plt.figure()
    plt.plot(history[loss], label='train loss')
    if with_val:
        plt.plot(history['val_'+loss], label='val loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title(title)
    plt.legend(loc='lower right')

    if path:
        fig.savefig(path + name)
    else:
        fig.savefig('./figures/' + name)

def compare_models(history_1, history_2, which_compare, model_1_name, model_2_name, plot_name, title, with_test=True, path=None):
    fig = plt.figure()
    plt.plot(history_1[which_compare], label=model_1_name+'_train_'+which_compare)
    plt.plot(history_2[which_compare], label=model_2_name+'_train_'+which_compare)
    # reasoning behind naming is because model comparison done on test
    if with_test:
        plt.plot(history_1['val_'+which_compare], label=model_1_name+'_test_'+which_compare)
        plt.plot(history_2['val_'+which_compare], label=model_2_name+'_test_'+which_compare)
    plt.ylabel(which_compare)
    plt.xlabel('epochs')
    plt.title(title)
    plt.legend(loc='best')
    
    fig.savefig(path+plot_name) if path else fig.savefig('./figures/'+plot_name)

def plot_predictions(model, X_test, y_test, num_obs, plot_name, title, path=None):
    idx = np.arange(X_test.shape[0])
    np.random.shuffle(idx)
    X_test = X_test[idx]
    y_test = y_test[idx]
    # take a subset of data
    X_test = X_test[0:num_obs, :]
    y_test = y_test[0:num_obs]
    predictions = model.predict(X_test)

    fig = plt.figure()
    plt.scatter(np.arange(num_obs), predictions, label='predictions')
    plt.scatter(np.arange(num_obs), y_test, label='truth_values')
    plt.xlabel('$i^{th}$ data point')
    plt.ylabel('$y$ values')
    plt.title(title)
    plt.legend()

    fig.savefig(path+plot_name) if path else fig.savefig('./figures/'+plot_name)