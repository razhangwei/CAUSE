import numpy as np
import pandas as pd


def explode(df, col: str, new_col: str = None):
    """Explode an array-valued column

    Args:
        df (DataFrame):
        col (str):

    Returns:
        DataFrame: the exploded DataFrame, whose length equal to the total
          length of the `col` column.
    """
    # Flatten columns of lists
    col_flat = [x for arr in df[col] for x in arr]
    # Row numbers to repeat
    lens = df[col].apply(len)
    ilocations = np.arange(len(df)).repeat(lens)

    # Replicate rows and add flattened column of lists
    col_indices = [i for i, c in enumerate(df.columns) if c != col]
    new_df = df.iloc[ilocations, col_indices].copy()
    new_col = new_col or col
    new_df[new_col] = col_flat
    return new_df


def applyParallel(dfGrouped, func):
    """parallel apply after group

    Args:
        dfGrouped (DataFrameGroupBy): the object after calling `groupby(...)`
        func (Callable): the function to apply

    Returns:
        List: results, one for each group key.
    """
    from multiprocessing import Pool

    with Pool() as p:
        ret_list = p.map(func, (group for name, group in dfGrouped))
    return ret_list


def multiindex_pivot(df, index, columns, values):
    """pivot with index using multiple columns

    > From <https://github.com/pandas-dev/pandas/issues/23955>

    Args:
        df (DataFrame): [description]
        index (Union[str, List[str]]):
        columns (str):
        values (Union[str, List[str]]):

    Returns:
        DataFrame: [description]
    """

    tuples_index = list(map(tuple, df[index].values))
    df = df.assign(tuples_index=tuples_index)
    df = df.pivot(index="tuples_index", columns=columns, values=values)
    new_index = pd.MultiIndex.from_tuples(df.index, names=index)
    df.index = new_index

    return df
