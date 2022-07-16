from pandas import DataFrame


def random_split_df(df: DataFrame, seed) -> tuple:

    print('\n seed = ', seed)
    train = df.sample(frac=0.8, random_state=seed)
    x = df.drop(train.index)
    val = x.sample(frac=0.5, random_state=seed)
    test = x.drop(val.index)
    datasets = [train, val, test]
    for x in datasets:
        classes_count = {'Normal': 0, 'Benign': 0,
                         'Malignant': 0, 'Lymph_nodes': 0}
        view_count = {'CC': 0, 'MLO': 0}

        for name in ['Normal', 'Benign', 'Malignant', 'Lymph_nodes']:
            temp = [0, 0, 0]
            for item in zip(x['class'], x['view']):
                temp[0] += sum([1 for s in item[0] if s == name])
                temp[1] += sum([1 for s in item[1] if s.find('CC') is not -1])
                temp[2] += sum([1 for s in item[1] if s.find('MLO') is not -1])
            classes_count[name] = temp[0]
            view_count['CC'] = temp[1]
            view_count['MLO'] = temp[2]
        print()
        print(classes_count, view_count)

    return train, val, test
