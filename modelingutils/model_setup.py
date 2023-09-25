def check_Xy_for_na(X, y):
    try:
        print("Checking for missing values in the target variable")
        assert y.isna().sum() == 0
    except AssertionError:
        print("There are missing values in the target variable")

    try:
        print("Checking for missing values in the features")
        assert (X.isna().sum() == 0).all()
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
