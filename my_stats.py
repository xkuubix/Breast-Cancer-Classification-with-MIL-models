import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydicom import dcmread
import os


def make_df(dir: str, from_file=True,
            save_to_file=False, file_dir=None) -> pd.DataFrame:
    '''
    Return pd.DataFrame for dicoms in given dir,
    df contains id, age, (laterality+view), class, dmc filename.
    '''
    if from_file is True and save_to_file is False:
        return pd.read_pickle(file_dir)

    empty_dict = {"id": None, "age": None, "view": None,
                  "class": None, "filename": None}
    patient_info = []
    os.chdir(dir)
    for folder in os.listdir(os.getcwd()):
        os.chdir(os.path.join(dir, folder))
        for file in os.listdir(os.getcwd()):
            new_patient = empty_dict.copy()
            dcm = dcmread(file)
            new_patient["id"] = int(dcm.PatientID)
            age = str(dcm.PatientAge)
            idx_end = age.find('Y')
            new_patient["age"] = int(age[idx_end-3:idx_end])
            new_patient["view"] = str(dcm.ImageLaterality
                                      + dcm.ViewPosition)
            # Firstly check DICOM data, then file name, and eventually discard
            if dcm.ViewPosition != 'CC' and dcm.ViewPosition != 'MLO':
                if str(file).__contains__('MLO') or dcm.ViewPosition == 'ML':
                    new_patient["view"] = str(dcm.ImageLaterality) + 'MLO'
                elif str(file).__contains__('CC'):
                    new_patient["view"] = str(dcm.ImageLaterality) + 'CC'
                else:
                    print('One dicom ommited due to lack of valid data')
                    continue

            new_patient["class"] = str(folder)
            new_patient["filename"] = str(dcm.filename)

            # filter image size
            if dcm.Rows == 3518 and dcm.Columns == 2800:
                patient_info.append(new_patient)
            else:
                print('One dicom ommited due to its img size')

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
        df.to_pickle(file_dir)

    return df


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
    plt.ylim(top=130)
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

    # Adding title
    plt.title("Patients' age stats")

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
