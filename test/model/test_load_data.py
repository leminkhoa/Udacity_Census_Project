def test__load_raw_data(raw_data):
    '''Raw dataframe should not be empty and has 15 columns'''
    assert raw_data.shape[0] > 0, "Empty dataframe"
    assert raw_data.shape[1] == 15, "Number of columns is not equal to 15"


def test__load_clean_data(cleaned_data):
    '''Cleaned dataframe should not be empty and has 15 columns'''
    assert cleaned_data.shape[0] > 0, "Empty dataframe"
    assert cleaned_data.shape[1] == 15, "Number of columns is not equal to 15"


def test__col_whitespace_clean_data(cleaned_data):
    '''Test if any whitespace in column names'''
    for col in cleaned_data.columns.tolist():
        assert ' ' not in col, "Test failed because there are whitespaces in columns"


def test__col_name_clean_data(cleaned_data):
    '''Test column names'''
    cols = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education_num',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital_gain',
        'capital_loss',
        'hours_per_week',
        'native_country',
        'salary'
    ]
    assert cleaned_data.columns.tolist() == cols
