import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydicom import dcmread
import os
import torch


def make_df(dir: str, from_file=True,
            save_to_file=False, file_dir=None) -> pd.DataFrame:
    '''
    Return pd.DataFrame for dicoms in given dir,
    df contains id, age, (laterality+view), class, dmc filename.
    '''
    if from_file is True and save_to_file is False:
        return pd.read_pickle(file_dir)

    empty_dict = {"id": None, "age": None, "view": None,
                  "class": None, "birads": None, "filename": None}
    patient_info = []
    ln = 0
    os.chdir(dir)
    for folder in os.listdir(os.getcwd()):
        if folder in ['Normal', 'Benign', 'Malignant', 'Lymph_nodes']:
            os.chdir(os.path.join(dir, folder))
        else:
            continue
        for file in os.listdir(os.getcwd()):
            if file.startswith('.'):
                continue
            new_patient = empty_dict.copy()
            dcm = dcmread(file)
            new_patient["id"] = str(dcm.PatientID)
            age = str(dcm.PatientAge)
            idx_end = age.find('Y')
            new_patient["age"] = int(age[idx_end-3:idx_end])
            new_patient["view"] = str(dcm.ImageLaterality
                                      + dcm.ViewPosition)
            # Firstly check DICOM data, then file name, and eventually discard
            if (dcm.ViewPosition != 'CC') and (dcm.ViewPosition != 'MLO'):
                if (str(file).__contains__('MLO')) or (dcm.ViewPosition
                                                       == 'ML'):
                    new_patient["view"] = str(dcm.ImageLaterality) + 'MLO'
                elif str(file).__contains__('CC'):
                    new_patient["view"] = str(dcm.ImageLaterality) + 'CC'
                else:
                    print('One dicom ommited due to lack of valid data')
                    continue

            new_patient["class"] = str(folder)
            new_patient["filename"] = str(dcm.filename)

            if str(new_patient["id"]) in ['1988128',
                                          '1281363',
                                          '1034498',
                                          '1203259',
                                          '1229710',
                                          '1299922',
                                          '1924088'
                                          ]:
                print('Manually chosen dicom ommited')
                continue
            if str(new_patient["id"]) not in ['46042', '89616', '97839', '133571', '140947', '150152', '170472', '180874', '296439', '313323', '328288', '338410', '468426', '602814', '607863', '732943', '820068', '824092', '854978', '864272', '868123', '911127', '914568', '1202672', '1206428', '1276395', '1321363', '1356072', '1361564', '1403576', '1404555', '1520161', '1547682', '1596229', '1607034', '1667721', '1674708', '1704783', '1705930', '1717708', '1720451', '1723300', '1723359', '1744566', '1768626', '1768907', '1769005', '1779605', '1783595', '1829164', '1831004', '1833492', '1847540', '1860082', '1862454', '1871642', '1879238', '1881963', '1884597', '1908990', '1927627', '1927739', '1964401', '1978727', '1994403', '2003125', '2031576', '2040257', '2050527', '2067675', '2086865', '2090209', '2110549', '2111071', '2114375', '2120222', '2135470', '2135842', '2136893', '2148719', '2160277', '2164329', '2201413', '2221645', '101642', '104138', '110685', '126065', '184441', '222056', '284992', '350189', '371276', '595160', '652848', '750013', '771090', '824031', '850528', '910636', '924138', '1016925', '1038577', '1046008', '1181189', '1197452', '1198408', '1199765', '1203375', '1281975', '1284001', '1300448', '1311687', '1336988', '1343653', '1393911', '1403083', '1426644', '1443028', '1460760', '1471738', '1476462', '1487588', '1524252', '1556953', '1607034', '1609525', '1636896', '1648834', '1672712', '1678601', '1694784', '1710382', '1715048', '1715126', '1727616', '1743909', '1759066', '1767174', '1780810', '1786151', '1816993', '1831621', '1863241', '1865848', '1879084', '1897797', '1906166', '1906479', '1924088', '1925843', '1927348', '1951970', '1977639', '1992048', '2008095', '2015088', '2023401', '2045435', '2047327', '2073074', '2085818', '2093633', '2208264']:
                # cancerous breast with no visible LN
                ln += 1
                # print('No lymph nodes visible:', ln)
                continue
            # filter image size
            if dcm.Rows == 3518 and dcm.Columns == 2800:
                patient_info.append(new_patient)
            else:
                print('One dicom ommited due to its img size:',
                      dcm.Rows, 'x', dcm.Columns, 'id:', dcm.PatientID)

    df = pd.DataFrame(patient_info)
    # print(df)
    df = df.sort_values(
        by=['view']).groupby('id').agg({"age": np.mean,
                                        "view": list,
                                        "class": list,
                                        "filename": list}).reset_index()
    df = df.sort_values(by='view', key=lambda x: x.str.len(), ascending=False)
    df = df.set_index(pd.Series([i for i in range(0, df.shape[0])]))

    if save_to_file:
        # with open(file_dir, 'a') as f:
        #     dfAsString = df.to_string(header=False, index=False)
        #     f.write(dfAsString)
        df.to_pickle(file_dir)
    return df


def make_img_stats_dict(ds):
    stats_dict = {'id': [], 'area': [],
                  'mean': [], 'std': [], 'var': [],
                  'class': []}
    for d_s in ['train', 'val', 'test']:
        for image, target in ds[d_s]:
            my_mask = target['full_image'].clone()
            my_mask[my_mask < 1e-4] = 0
            my_mask[my_mask > 1e-4] = 1
            np_image = my_mask.permute(1, 2, 0).numpy()
            # plt.imshow(np_image)
            # plt.show()
            # plt.imshow(target['full_image'].permute(1, 2, 0))
            # plt.show()
            total_pixels = np.sum(np_image)/np_image.shape[2]
            total_area = 0.085**2 * total_pixels * 0.01  # cm square
            # print('area = ', total_area, ' [cm2]')
            masked_image_mean = torch.mean(target['full_image'][my_mask == 1])
            masked_image_std = torch.std(target['full_image'][my_mask == 1])
            masked_image_var = torch.var(target['full_image'][my_mask == 1])
            # print(masked_image_mean)
            # print(masked_image_std)
            # print(masked_image_var)
            stats_dict['id'].append(target['patient_id'])
            stats_dict['area'].append(total_area)
            stats_dict['mean'].append(masked_image_mean.item())
            stats_dict['std'].append(masked_image_std.item())
            stats_dict['var'].append(masked_image_var.item())
            stats_dict['class'].append(target['class'])
    # plt.scatter(stats_dict['area'], stats_dict['mean'])
    return stats_dict


def box_plot(stats_dict, k):
    '''k = area, mean, std or var as string
       stats_dict from make_img_stats_dict'''
    normal = np.array(stats_dict[k])[np.array(stats_dict['class'])
                                     == 'Normal']
    benign = np.array(stats_dict[k])[np.array(stats_dict['class'])
                                     == 'Benign']
    malignant = np.array(stats_dict[k])[np.array(stats_dict['class'])
                                        == 'Malignant']
    lymph_nodes = np.array(stats_dict[k])[np.array(stats_dict['class'])
                                          == 'Lymph_nodes']
    data = [normal, benign, malignant, lymph_nodes]
    classes_names = ['Normal', 'Benign', 'Malignant', 'Lymph_nodes']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist=True,
                    notch='True', vert=0)

    colors = ['#0000FF', '#00FF00',
              '#FFFF00', '#FF00FF']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")
    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)
    # x-axis labels
    ax.set_yticklabels(classes_names)
    # Adding title
    if k == 'area':
        title = k + ' [cm$^2$]'
    else:
        title = k
    plt.title(title)
    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # show plot
    plt.show()


def calculate_laterality_count(dataset) -> dict:
    """
    Returns L/R projections count for given dataset
    """
    L_count = 0
    R_count = 0
    laterality_count = dict()

    for element in range(len(dataset)):
        if dataset[element][1]["laterality"] == 'R':
            R_count += 1
        elif dataset[element][1]["laterality"] == 'L':
            L_count += 1
    laterality_count["L"] = L_count
    laterality_count["R"] = R_count
    return laterality_count


def calculate_age_stats(dataset) -> dict:
    """
    Returns dict of patients' age descriptive statistics for given dataset
    """
    patients_age = np.array([])

    for element in range(len(dataset)):
        patients_age = np.append(patients_age, dataset[element][1]["age"])
    age_stats = dict()
    age_stats["mean"] = np.mean(patients_age)
    age_stats["median"] = np.median(patients_age)
    age_stats["min"] = np.min(patients_age)
    age_stats["max"] = np.max(patients_age)
    age_stats["all_ages"] = patients_age
    return age_stats


def projections_hist(classes_names: list, proj_L: list, proj_R: list):
    labels = classes_names
    X = np.arange(len(labels))
    y1 = proj_L
    y2 = proj_R
    width = 0.4
    X_axis = np.arange(len(X))

    fig, ax = plt.subplots()
    ax.bar(X_axis - 0.2, y1, width, label='L')
    ax.bar(X_axis + 0.2, y2, width, label='R')
    # plt.ylim(top=130)
    plt.grid(True)
    plt.xticks(X_axis, labels)
    plt.xlabel("Classes")
    ax.set_ylabel('Count')
    ax.set_title('Projections count among classes')
    ax.legend()
    plt.show()
    return


def age_hist(classes_names: list, all_ages: list):
    normal = all_ages[0]
    benign = all_ages[1]
    malignant = all_ages[2]
    lymph_nodes = all_ages[3]
    data = [normal, benign, malignant, lymph_nodes]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist=True,
                    notch='True', vert=0)

    colors = ['#0000FF', '#00FF00',
              '#FFFF00', '#FF00FF']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)

    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)

    # x-axis labels
    ax.set_yticklabels(classes_names)

    # Adding title and xlabel description
    plt.title("Patients' age stats")
    plt.xlabel('Age')

    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # show plot
    plt.show()
    return


def img_sizes(classes: list):
    pair_list = []
    for n_class in range(len(classes)):
        for element in range(len(classes[n_class])):
            w = classes[n_class][element][1]["img_w"]
            h = classes[n_class][element][1]["img_h"]
            pair = (w, h)
            pair_list.append(pair)
    pair_set = set(pair_list)
    F = {}
    for pair in list(pair_set):
        F[pair] = pair_list.count(pair)
    K = F.keys()
    X = [pair[0] for pair in K]
    Y = [pair[1] for pair in K]
    plt.plot(X, Y, 'ko')
    for item in K:
        plt.text(item[0], item[1]+20, str(F[item]), fontsize=16)
    plt.plot(X, Y, 'ko')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image sizes', fontsize=16)
    plt.ylim(top=3800)
    plt.xlim(right=2900)
    return


def show_stats(classes: list):
    """
    Displays statistics for given list of datasets...
    """
    classes_names = ['Normal', 'Benign', 'Malignant', 'Lymph node']
    projections_L = list()
    projections_R = list()
    all_ages = list()
    print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        'Class name', 'Total',
        'Left', 'Rigth',
        'age_min', 'age_mean',
        'age_median', 'age_max'))

    for n_class in range(len(classes)):
        d1 = calculate_laterality_count(classes[n_class])
        d2 = calculate_age_stats(classes[n_class])
        projections_L.append(d1["L"])
        projections_R.append(d1["R"])
        all_ages.append(d2["all_ages"])
        print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
            classes_names[n_class],  d1["L"] + d1["R"],
            d1["L"], d1["R"],
            d2["min"], round(d2["mean"], 1),
            d2["median"], d2["max"]))
    print('-'*8*12)
    print("{:<12} {:<12} {:<12} {:<12}".format(
          ' ',
          sum([len(item) for item in classes]),
          sum(projections_L),
          sum(projections_R)))

    projections_hist(classes_names, projections_L, projections_R)
    age_hist(classes_names, all_ages)
    img_sizes(classes)
    return


def show_statistics(df):
    class_counts = {'Normal': 0, 'Benign': 0,
                    'Malignant': 0, 'Lymph_nodes': 0}
    laterality_counts_L = {'Normal': 0, 'Benign': 0,
                           'Malignant': 0, 'Lymph_nodes': 0}
    laterality_counts_R = {'Normal': 0, 'Benign': 0,
                           'Malignant': 0, 'Lymph_nodes': 0}
    all_ages = {'Normal': list(), 'Benign': list(),
                'Malignant': list(), 'Lymph_nodes': list()}
    for i in range(len(df)):
        # for k in class_counts.keys():
        if len(df.iloc[i]['view']) == 2:
            class_counts[df.iloc[i]['class'][0]] += 1
            if df.iloc[i]['view'][0].__contains__('L'):
                laterality_counts_L[df.iloc[i]['class'][0]] += 1
            elif df.iloc[i]['view'][0].__contains__('R'):
                laterality_counts_R[df.iloc[i]['class'][0]] += 1
            all_ages[df.iloc[i]['class'][0]].append(df.iloc[i]['age'])
        elif len(df.iloc[i]['view']) == 4:
            laterality_counts_L[df.iloc[i]['class'][0]] += 1
            laterality_counts_R[df.iloc[i]['class'][0]] += 1
            # class_counts[df.iloc[i]['class'][0]] += 2
            all_ages[df.iloc[i]['class'][0]].append(df.iloc[i]['age'])
            all_ages[df.iloc[i]['class'][0]].append(df.iloc[i]['age'])

    classes_names = list(class_counts.keys())
    # print(class_counts)
    # print(laterality_counts_L)
    # print(laterality_counts_R)
    # print(all_ages)
    ages_list = [all_ages[key] for key in classes_names]
    age_hist(classes_names, ages_list)
    print('L')
    print(laterality_counts_L)
    print('R')
    print(laterality_counts_R)

    i = 0
    for item in ages_list:
        print('\nClass name', classes_names[i])
        print('mean', end=' ')
        print(np.array(item).mean())
        print('std', end=' ')
        print(np.array(item).std())
        print('max', end=' ')
        print(np.array(item).max())
        print('min', end=' ')
        print(np.array(item).min())
        print('median', end=' ')
        print(np.median(np.array(item)))
        i += 1
    full_ds_list = [item for sublist in ages_list for item in sublist]
    print('\nWhole dataset')
    print('mean', end=' ')
    print(np.array(full_ds_list).mean())
    print('std', end=' ')
    print(np.array(full_ds_list).std())
    print('max', end=' ')
    print(np.array(full_ds_list).max())
    print('min', end=' ')
    print(np.array(full_ds_list).min())
    print('median', end=' ')
    print(np.median(np.array(full_ds_list)))

    full_ds_list = [item for sublist in ages_list[0:2] for item in sublist]
    print('\nNon-cancer')
    print('mean', end=' ')
    print(np.array(full_ds_list).mean())
    print('std', end=' ')
    print(np.array(full_ds_list).std())
    print('max', end=' ')
    print(np.array(full_ds_list).max())
    print('min', end=' ')
    print(np.array(full_ds_list).min())
    print('median', end=' ')
    print(np.median(np.array(full_ds_list)))

    full_ds_list = [item for sublist in ages_list[2:] for item in sublist]
    print('\nCancer')
    print('mean', end=' ')
    print(np.array(full_ds_list).mean())
    print('std', end=' ')
    print(np.array(full_ds_list).std())
    print('max', end=' ')
    print(np.array(full_ds_list).max())
    print('min', end=' ')
    print(np.array(full_ds_list).min())
    print('median', end=' ')
    print(np.median(np.array(full_ds_list)))

    Ls = [laterality_counts_L[key] for key in classes_names]
    Rs = [laterality_counts_R[key] for key in classes_names]
    projections_hist(classes_names, Ls, Rs)
    return
# %%
# %%
