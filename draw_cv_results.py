# %%
import neptune.new as neptune
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
sns.set()
# %%
data = neptune.init_project(name='ProjektMMG/5-fold-CV-Cancers')

markers = ['r.-', 'go-', 'bv-', 'kx-', 'y*-', 'cD-', 'mH-']
marker = 0
tags = [
        # 'nr 12',
        # 'nr 2',
        'nr 19',
        'nr 20',
        # 'a1',
        # 'nr 21',
        # 'nr 8',
        # 'nr 18', 'nr 9', 'nr 11', 'nr 17', 'nr 14',
        # 'nr 2', 'nr 3',
        # 'nr 5',
        # 'nr 4', 'nr 13',
        # 'nr 5', 'nr 15', 'nr 10', 'nr 6',
        # 'nr 2', 'nr 5', 'nr 19', 'nr 20', 'nr 22'
        # 'nr 19', 'nr 20'
        # 'n1',
        # 'n2',
        # 'n4'
        # '1n1',
        # '2n2',
        # '3n3',
        # '4n4'
        # 'a6',
        # 'a5',
        # 'a2',
        # 'a3',
        'a4',
        'a10',
        # 'nr 5'
        # 'a7',
        # 'a8',
        # 'a9',
        # 'a10'
        # 'nr 5'
        ]

f1s = []
accs = []
aucs = []
precs = []
recs = []

for tag in tags:
    marker += 1
    runs_table_df = data.fetch_runs_table(owner=["rbuler", 'jakub-buler'],
                                          tag=tag).to_pandas()
    # print(runs_table_df['sys/monitoring_time'])
    mean_time = runs_table_df['sys/monitoring_time'].mean()
    hours = mean_time // 3600
    minutes = (mean_time % 3600) // 60
    print(int(hours), ':', int(minutes))
    # %%

    k = max(runs_table_df['fold-k']) + 1
    metrics = {'f1': [], 'acc': [], 'auc': [],
               'precision': [], 'recall': []}
    for i in range(k):
        if i == 7:
            continue
        fold = runs_table_df['fold-k'][runs_table_df['fold-k'] == i].to_string(
            index=False)
        bacc = runs_table_df['test/BACC/auc_roc'][
            runs_table_df['fold-k'] == i].values[0]
        bloss = runs_table_df['test/auc_roc'][
            runs_table_df['fold-k'] == i].values[0]
        if bloss >= bacc:
            s = str(runs_table_df[['test/BL/metrics']][
                runs_table_df['fold-k'] == i].values)
            auc = bloss
        else:
            s = str(runs_table_df[['test/BACC/metrics']][
                runs_table_df['fold-k'] == i].values)
            auc = bacc
        values = re.findall(
            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)

        # negative_precision = values[1]
        # negative_recall = values[2]
        # negative_f1 = values[3]
        # negative_suppot = values[4]
        # positive_precision = values[5]
        # positive_recall = values[6]
        # positive_f1 = values[7]
        # positive_support = values[8]
        if len(values) > 2:
            f1_weighted = values[-2]
            recall = values[-3]
            precision = values[-4]
            accuracy = values[9]
            # accuracy_support = values[10]
            metrics['acc'].append(float(accuracy))
            metrics['auc'].append(auc.__round__(2))
            metrics['f1'].append(float(f1_weighted))
            metrics['precision'].append(float(precision))
            metrics['recall'].append(float(recall))

    aucs.append(metrics['auc'])
    accs.append(metrics['acc'])
    f1s.append(metrics['f1'])
    precs.append(metrics['precision'])
    recs.append(metrics['recall'])

    # print(runs_table_df.columns)
    # print(runs_table_df[['fold-k', 'test/BACC/auc_roc', 'test/auc_roc']])
    plt.plot(runs_table_df['fold-k'], runs_table_df['test/auc_roc'],
             'x-')
    plt.xlabel('Fold number')
    plt.ylabel('AUC ROC')
    # plt.legend('ABCDEF', ncol=2, loc='upper left')
    # print(metrics)
    # Creating axes instance

# %%
def bp(data, labels_x, label_y, fig, sp):
    # fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(sp)
    bplot = ax.boxplot(data, patch_artist=True,
                       notch=None, vert=1)

    colors = [
        '#212121', '#424242', '#757575', '#455A64', '#546E7A', '#78909C',
        '#212121', '#424242', '#757575', '#455A64', '#546E7A', '#78909C']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='red', linewidth=3)

    # ax.set_xticklabels(labels_x)
    title = '5-fold CV'
    if sp == 232:
        plt.title(title)
    plt.ylabel(label_y)
    # ax.get_xaxis().tick_bottom()
    if sp == 235:
        # import matplotlib.patches as mpatches
        ax.get_yaxis().tick_left()
        plt.xticks(rotation=0)
        # p1 = mpatches.Patch(color=colors[0], label='')
        # p2 = mpatches.Patch(color=colors[1], label='')
        # p3 = mpatches.Patch(color=colors[2],  label='')
        # p4 = mpatches.Patch(color=colors[3],  label='')
        # p5 = mpatches.Patch(color=colors[4],  label='')
        # p6 = mpatches.Patch(color=colors[5],  label='')
        # p6 = mpatches.Patch(color=colors[0],  label='')
        # p7 = mpatches.Patch(color='#eaeaf200',  label='')
        # p8 = mpatches.Patch(color='#eaeaf200',  label='')
        # plt.legend(handles=[p1, p2, p3])
        # plt.legend(handles=[p1, p2, p3, p4,
        #                     p5, p6], loc='center left',
        #            bbox_to_anchor=(1.25, 0.5),
        #            fontsize='large')
        plt.show()


tags = [
        # 'CC',
        # 'CC\nresize',
        # 'MLO',
        # 'MLO\nresize',
        # 'CC\nMLO',
        # 'CC\nMLO\nresize',
        # 'ov 0.\nps 224',
        # '\n\nov 0.25\nps 224',
        # 'ov 0.5\nps 224',
        # '\n\nov 0.5\nps 128',
        # 'ov 0.5\nps 448',
        # '\n\nov 0\nps 224',
        # 'ov 0.25\nps 224',
        # '\n\nov 0.5\nps 224',
        # 'ov 0.5\nps 128',
        # '\n\nov 0.5\nps 448',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6'
        # '7',
        # '8',
        # '9',
        # '10',
        # '5'
        # 'ResNet\n18',
        # 'ResNet\n34',
        # 'ResNet\n50',
        # 'EfficientNet-\nB0',
        # 'pusty bez pasków',
        # 'pusty z paskami'
        ]

fig = plt.figure(figsize=(16.5, 12))
bp(aucs, tags, 'AUC-ROC', fig=fig, sp=231)
bp(f1s, tags, 'F1-Score', fig=fig, sp=232)
bp(accs, tags, 'Accuracy', fig=fig, sp=233)
bp(precs, tags, 'Precision', fig=fig, sp=234)
bp(recs, tags, 'Recall', fig=fig, sp=235)

data.stop()
# %%
np_aucs = np.array(aucs)
np_accs = np.array(accs)
np_f1s = np.array(f1s)
np_precs = np.array(precs)
np_recs = np.array(recs)
for i in range(len(tags)):
    print()
    print(tags[i], end=' ')
    print("{0:.3f}".format(np_aucs.mean(axis=1)[i]), end=' ')
    print("{0:.3f}".format(np_aucs.std(axis=1)[i]), end=' ')

    print("{0:.3f}".format(np_f1s.mean(axis=1)[i]), end=' ')
    print("{0:.3f}".format(np_f1s.std(axis=1)[i]), end=' ')

    print("{0:.3f}".format(np_accs.mean(axis=1)[i]), end=' ')
    print("{0:.3f}".format(np_accs.std(axis=1)[i]), end=' ')

    print("{0:.3f}".format(np_precs.mean(axis=1)[i]), end=' ')
    print("{0:.3f}".format(np_precs.std(axis=1)[i]), end=' ')

    print("{0:.3f}".format(np_recs.mean(axis=1)[i]), end=' ')
    print("{0:.3f}".format(np_recs.std(axis=1)[i]), end=' ')

# %%
