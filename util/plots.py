import matplotlib.pyplot as plt
import seaborn as sns

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