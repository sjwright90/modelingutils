from numpy import isnan


def check_Xy_for_na(X, y):
    """Checks for missing values in the target variable and features

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray

    y : pandas.Series or numpy.ndarray

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If there are missing values in the target variable

    Exception
        If there are missing values in the features and the user chooses not to continue
    """

    try:
        print("Checking for missing values in the target variable")
        assert isnan(y).sum() == 0
    except AssertionError:
        print("There are missing values in the target variable")

    try:
        print("Checking for missing values in the features")
        assert isnan(X).sum().sum() == 0
    except:
        print(
            "There are missing values in the features, if this is expected enter 'y' to continue"
        )
        inpt = input("Do you want to continue? [y/n]")
        if "y" in inpt.lower():
            print("Continuing")
            pass
        else:
            raise Exception("Missing values found user chose not to continue")
