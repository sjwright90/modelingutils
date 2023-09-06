from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import os


# %%
def plot_cm(clf, X_test, y_test, labels=["STPAG", "LTPAG", "NPAG"]):
    """Plot confusion matrix for a classifier
    Parameters
    ----------
    clf : sklearn classifier
        The classifier to be evaluated
    X_test : array-like
        The test set
    y_test : array-like
        The test set labels
    Returns
    -------
    cmd : ConfusionMatrixDisplay
        The confusion matrix plot, a ConfusionMatrixDisplay object"""
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
    _ = cmd.plot()
    return cmd


# %%
def side_by_side_cm(clf_a, clf_b, X_test, y_test, sup_title="", **kwargs):
    """Plot side by side confusion matrices for two classifiers
    Parameters
    ----------
    clf_a : sklearn classifier
        The first classifier to be evaluated
    clf_b : sklearn classifier
        The second classifier to be evaluated
    X_test : array-like
        The test set
    y_test : array-like
        The test set labels
    Returns
    -------
    figx : matplotlib figure
        The figure object
    axx : matplotlib axes
        The axes object
    """
    figx, axx = plt.subplots(1, 2, figsize=(15, 5))

    cmd_a = plot_cm(clf_a, X_test, y_test, **kwargs)
    cmd_b = plot_cm(clf_b, X_test, y_test, **kwargs)

    _ = cmd_a.plot(ax=axx[0])
    _ = cmd_b.plot(ax=axx[1])

    axx[0].set_title("Gradient Boosting", fontsize=17)
    axx[1].set_title("Decision Tree", fontsize=17)

    _ = axx[0].text(
        0.5,
        -0.2,
        f"F1_score: {f1_score(y_test,clf_a.predict(X_test), average='weighted'):.2f}",
        fontsize=15,
        ha="center",
        transform=axx[0].transAxes,  # relative coordinates
    )
    _ = axx[1].text(
        0.5,
        -0.2,
        f"F1_score: {f1_score(y_test,clf_b.predict(X_test), average='weighted'):.2f}",
        fontsize=15,
        ha="center",
        transform=axx[1].transAxes,  # relative coordinates
    )
    _ = figx.suptitle(sup_title, fontsize=23, y=1.05)
    return figx, axx


# %%
def make_props_df(
    clf_a, clf_b, X_test, y_test, ordered_labels=["STPAG", "LTPAG", "NPAG"]
):
    """Make a dataframe with proportions of predictions for each class
    Parameters
    ----------
    clf_a : sklearn classifier
        The first classifier to be evaluated, will be labeled GBM_Predictions
    clf_b : sklearn classifier
        The second classifier to be evaluated, will be labeled DT_Predictions
    X_test : array-like
        The test set, is independent from y_test
    y_test : array-like
        The test set labels, will be labeled Actual
    Returns
    -------
    proportions : pandas DataFrame
        The dataframe with proportions of predictions for each class
    """
    y_pred = clf_a.predict(X_test)
    y_pred_dt = clf_b.predict(X_test)
    proportions = DataFrame(
        {
            "GBM_Predictions": Series(y_pred).value_counts(normalize=True),
            "DT_Predictions": Series(y_pred_dt).value_counts(normalize=True),
            "Actual": Series(y_test).value_counts(normalize=True),
        }
    ).reindex(ordered_labels)
    return proportions


# %%
def table_to_png(df, filename, **kwargs):
    """Save a pandas dataframe as a png file
    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be saved
    filename : str
        Pathname of the file to be saved
    **kwargs : dict
        Additional arguments to be passed to the table method
    Returns
    -------
    None
        Saves the file to the specified path
    """
    # not very modular, mainly use for plotting value counts of
    # 2 or 3 classifiers side by side
    figx, axx = plt.subplots(figsize=(7, 1))
    axx.xaxis.set_visible(False)  # hide the x axis
    axx.yaxis.set_visible(False)  # hide the y axis
    axx.set_frame_on(False)  # no visible frame, uncomment if size is ok
    axx.axis("tight")
    tabla = axx.table(
        cellText=df.applymap(lambda x: f"{x:.2f}").values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
        rowLabels=df.index,
        **kwargs,
    )
    tabla.auto_set_font_size(False)  # Activate set fontsize manually
    tabla.set_fontsize(15)  # if ++fontsize is necessary ++colWidths
    tabla.scale(1, 2)  # change size table
    figx.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )
