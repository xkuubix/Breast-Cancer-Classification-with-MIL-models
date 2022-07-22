import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats


def my_show_image(dcm, with_marks=False, prediction=None,
                  class_names=None, format_type=None):

    if format_type == 'svg':
        set_matplotlib_formats(format_type)
        fig, ax = plt.subplots(figsize=(35.18, 28), dpi=100)
        fs = 50
    else:
        fig, ax = plt.subplots()
        fs = 10

    ax.imshow(dcm[0].permute(1, 2, 0), cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if with_marks:
        view_str = 'View: ' + dcm[1]["view"]
        class_str = 'Class: ' + dcm[1]["class"]
        if prediction is not None:
            pred_str = '\nclassified as: ' + str(class_names[prediction])
            ax.set_title(view_str + '   ' + class_str + pred_str, fontsize=fs)
        else:
            ax.set_title(view_str + '   ' + class_str, fontsize=fs)
    return


def my_show_training_results(accuracy_stats: dict, loss_stats: dict):

    set_matplotlib_formats('svg')

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(accuracy_stats["train"])
    ax1.plot(accuracy_stats["val"])
    ax1.set_title('Accuracy stats')
    ax1.legend(list(accuracy_stats.keys()))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')

    ax2.plot(loss_stats["train"])
    ax2.plot(loss_stats["val"])
    ax2.set_title('Loss stats')
    ax2.legend(list(loss_stats.keys()))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    return
