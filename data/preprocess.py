import pandas as pd


loaded = {}
FOLDER = __file__ + '/../'


def prompt_set() -> pd.DataFrame:
    while True:
        print('Available data sets:')
        print(*['  ' + key for key in list(available_sets.keys())], sep='\n')
        name = input('Please select dataset: ')
        
        dataset = get_data_set(name)

        if dataset is not None:
            return dataset


def get_data_set(name: str) -> pd.DataFrame:
    name = name.strip().lower()
    if name not in available_sets:
        print(f'{name} is not a name of an available data set.')
        return None

    print(f'Loading {name}...')
    dataset = available_sets[name]() if name not in loaded else loaded[name]
    loaded[name] = dataset
    print('Loaded!')
    return dataset


def get_zoo():
    classes = ['savec', 'pták', 'plaz', 'ryba', 'obojživelník', 'hmyz', 'korýš']
    data = pd.read_csv(FOLDER + 'zoo/zoo.csv')
    del data['animal']

    data['type'] = [classes[idx - 1] for idx in data['type']]

    for col in data.columns:
        if col in ('type', 'legs'): continue
        data[col] = ['ANO' if val == 1 else 'NE' for val in data[col]]

    return data


def get_churn1():
    # Load data
    data = pd.read_csv(FOLDER + 'churn1/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Drop unecessary and continuous attributes as binning is currently not supported
    delete = ['customer ID',  'tenure', 'Monthly Charges', 'Total Charges']
    for att in delete:
        del data[att]

    return data


available_sets = {
    'zoo': get_zoo,
    'churn1': get_churn1
}