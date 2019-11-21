import collections
import numpy as np
import pandas as pd

import multiprocessing
import multiprocessing.pool

import base
from base import Variable
from correlations import calculate_correlations
from correlations import perform_check_correlation

config = {
    "pool_size" : 0, 
    "check_correlation_pearson" : True, 
    "correlation_threshold_pearson" : 0.9,
    "check_correlation_cramers": False,
    "correlation_threshold_cramers": 0.9,
    "check_recoded": True
    }
	

def describe_numeric(series: pd.Series, series_description: dict):
    """Describe a numeric series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    distinct_count_without_nan = series.value_counts(dropna=True).count()

    stats = {
        "mean": series.mean(), 
        "std": series.std(), 
        "variance": series.var(), 
        "min": series.min(),
        "max": series.max(), 
        "kurtosis": series.kurt(), 
        "skewness": series.skew(), 
        "sum": series.sum(),
        "mad": series.mad(), 
        "distinct_count_without_nan" : distinct_count_without_nan, 
        "count_without_nan" : series.count()}

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    stats.update(
        {
            "{:.0%}".format(percentile):value for percentile, value in dict(series.quantile(quantiles)).items()
        }
        )

    
    stats["range"] = stats["max"] - stats["min"]
    stats["iqr"] = stats["75%"] - stats["25%"]
    stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else np.NaN


    return stats
	

def describe_date(series: pd.Series, series_description: dict):
    """Describe a date series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    stats = {"min": series.min(), "max": series.max()}
    stats["range"] = stats["max"] - stats["min"]

    return stats
	
	
def describe_categorical(series: pd.Series, series_description: dict) -> dict:
    """Describe a categorical series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.
    Returns:
        A dict containing calculated series description values.
    """

    # Make sure we deal with strings
    series = series[~series.isnull()].astype(str)

    value_counts = series_description["value_counts_without_nan"]
    stats = {"top": value_counts.index[0], "freq": value_counts.iloc[0]}

    return stats
	
def describe_url(series: pd.Series, series_description: dict) -> dict:
    """Describe a path series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    stats = {}

    # Make sure we deal with strings. Convert only non empty values as string
    series = series[~series.isnull()].astype(str)
    
    value_counts = series_description["value_counts_without_nan"]
    
    stats["top"] = value_counts.index[0]
    stats["freq"] = value_counts.iloc[0]

    return stats
	

def describe_path(series: pd.Series, series_description: dict) -> dict:
    """Describe a path series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    stats = {}

    # Make sure we deal with strings
    series = series[~series.isnull()].astype(str)

    # Only run if at least 1 non-missing value
    value_counts = series_description["value_counts_without_nan"]

    stats["top"] = value_counts.index[0]
    stats["freq"] = value_counts.iloc[0]

    return stats
	

def describe_boolean(series: pd.Series, series_description: dict) -> dict:
    """Describe a boolean series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.
    Returns:
        A dict containing calculated series description values.
    """

    value_counts = series_description["value_counts_without_nan"]

    stats = {"top": value_counts.index[0], "freq": value_counts.iloc[0]}

    return stats
	

def describe_constant(series: pd.Series, series_description: dict) -> dict:
    """Describe a constant series (placeholder).
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.
    Returns:
        An empty dict.
    """
    return {}
	
	
def describe_unique(series: pd.Series, series_description: dict) -> dict:
    """Describe a unique series (placeholder).
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.
    Returns:
        An empty dict.
    """

    return {}
	
	
def describe_supported(series: pd.Series, series_description: dict) -> dict:
    """Describe a supported series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.
    
    Returns:
        A dict containing calculated series description values.
    """

    # number of observations in the Series
    leng = len(series)

    # number of non-NaN observations in the Series
    count = series.count()

    # number of infinite observations in the Series
    n_infinite = count - series.count()

    distinct_count = series_description["distinct_count_with_nan"]

    stats = {
        "count": count,
        "distinct_count": distinct_count,
        "p_missing": 1 - count * 1.0 / leng,
        "n_missing": leng - count,
        "p_infinite": n_infinite * 1.0 / leng,
        "n_infinite": n_infinite,
        "is_unique": distinct_count == leng,
        "mode": series.mode().iloc[0] if count > distinct_count > 1 else series[0],
        "p_unique": distinct_count * 1.0 / leng,
        "memorysize": series.memory_usage()
    }

    return stats
	
def describe_unsupported(series: pd.Series, series_description: dict):
    """Describe an unsupported series.
    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.
    
    Returns:
        A dict containing calculated series description values.
    """

    # number of observations in the Series
    leng = len(series)

    # number of non-NaN observations in the Series
    count = series.count()

    # number of infinte observations in the Series
    n_infinite = count - series.count()

    results_data = {
        "count": count,
        "p_missing": 1 - count * 1.0 / leng,
        "n_missing": leng - count,
        "p_infinite": n_infinite * 1.0 / leng,
        "n_infinite": n_infinite,
        "memorysize": series.memory_usage()
    }

    return results_data
	

def describe_var(series: pd.Series) -> dict:
    """Describe a series (infer the variable type, then calculate type-specific values).
    Args:
        series: The Series to describe.
    
    Returns:
        A dictionary containing calculated series description values.
    """

    # Replace infinite values with NaNs to avoid issues.
    series.replace(to_replace=[np.inf, np.NINF, np.PINF], value=np.nan, inplace=True)

    # Infer variable types
    series_description = base.get_var_type(series)

    # Run type specific analysis
    if series_description["type"] == Variable.S_TYPE_UNSUPPORTED:
        series_description.update(describe_unsupported(series, series_description))
    else:
        series_description.update(describe_supported(series, series_description))

        type_to_func = {
            Variable.S_TYPE_CONST: describe_constant,
            Variable.TYPE_BOOL: describe_boolean,
            Variable.TYPE_NUM: describe_numeric,
            Variable.TYPE_DATE: describe_date,
            Variable.S_TYPE_UNIQUE: describe_unique,
            Variable.TYPE_CAT: describe_categorical,
            Variable.TYPE_URL: describe_url,
            Variable.TYPE_PATH: describe_path,
        }

        if series_description["type"] in type_to_func:
            series_description.update(
                type_to_func[series_description["type"]](series, series_description)
            )
        else:
            raise ValueError("Unexpected type")

    # Return the description obtained
    return series_description
	

def describe_table(df: pd.DataFrame, variable_stats: pd.DataFrame) -> dict:
    """General statistics for the DataFrame.
    Args:
      df: The DataFrame to describe.
      variable_stats: Previously calculated statistic on the DataFrame.
    Returns:
        A dictionary that contains the table statistics.
    """
    n = len(df)

    memory_size = df.memory_usage(index=True).sum()
    record_size = float(memory_size) / n

    table_stats = {
        "n": n,
        "nvar": len(df.columns),
        "memsize": memory_size,
        "recordsize": record_size,
        "n_cells_missing": variable_stats.loc["n_missing"].sum(),
        "n_vars_with_missing": sum((variable_stats.loc["n_missing"] > 0).astype(int)),
        "n_vars_all_missing": sum((variable_stats.loc["n_missing"] == n).astype(int)),
    }

    table_stats["p_cells_missing"] = table_stats["n_cells_missing"] / (
        table_stats["n"] * table_stats["nvar"]
    )

    supported_columns = variable_stats.transpose()[
        variable_stats.transpose().type != Variable.S_TYPE_UNSUPPORTED
    ].index.tolist()
    table_stats["n_duplicates"] = (
        sum(df.duplicated(subset=supported_columns))
        if len(supported_columns) > 0
        else 0
    )
    table_stats["p_duplicates"] = (
        (table_stats["n_duplicates"] / len(df))
        if (len(supported_columns) > 0 and len(df) > 0)
        else 0
    )

    # Variable type counts
    table_stats.update({k.value: 0 for k in Variable})
    table_stats.update(
        dict(variable_stats.loc["type"].apply(lambda x: x.value).value_counts())
    )
    table_stats[Variable.S_TYPE_REJECTED.value] = (
        table_stats[Variable.S_TYPE_CONST.value]
        + table_stats[Variable.S_TYPE_CORR.value]
        + table_stats[Variable.S_TYPE_RECODED.value]
    )
    return table_stats
	

def multiprocess_1d(column, series):
    """Wrapper to process series in parallel.
    Args:
        column: The name of the column.
        series: The series values.
    Returns:
        A tuple with column and the series description.
    """
    return column, describe_var(series)
	

def update(d: dict, u: dict) -> dict:
    """ Recursively update a dict.
    Args:
        d: Dictionary to update.
        u: Dictionary with values to use.
    Returns:
        The merged dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
	

def describe(df: pd.DataFrame) -> dict:
    """Calculate the statistics for each series in this DataFrame.
    Args:
        df: DataFrame.
    Returns:
        This function returns a dictionary containing:
            - table: overall statistics.
            - variables: descriptions per series.
            - correlations: correlation matrices.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be of type pandas.DataFrame")

    if df.empty:
        raise ValueError("df can not be empty")

    # Multiprocessing of Describe_var for each column
    pool_size = config['pool_size']

    if pool_size <= 0:
        pool_size = multiprocessing.cpu_count()

    if pool_size == 1:
        args = [(column, series) for column, series in df.iteritems()]
        series_description = {
            column: series
            for column, series in itertools.starmap(multiprocess_1d, args)
        }
    else:
        with multiprocessing.pool.ThreadPool(pool_size) as executor:
            series_description = {}
            results = executor.starmap(multiprocess_1d, df.iteritems())
            for col, description in results:
                series_description[col] = description

    # Mapping from column name to variable type
    variables = {
        column: description["type"]
        for column, description in series_description.items()
    }

    # Get correlations
    correlations = calculate_correlations(df, variables)    

    # Check correlations between numerical variables
    if config["check_correlation_pearson"] and "pearson" in correlations:
        # Overwrites the description with "CORR" series
        correlation_threshold = config["correlation_threshold_pearson"]
        update(
            series_description,
            perform_check_correlation(
                correlations["pearson"],
                lambda x: x > correlation_threshold,
                Variable.S_TYPE_CORR,
            ),
        )

    # Check correlations between categorical variables
    if config["check_correlation_cramers"] and "cramers" in correlations:

        # Overwrites the description with "CORR" series
        correlation_threshold = config["correlation_threshold_cramers"]
        update(
            series_description,
            perform_check_correlation(
                correlations["cramers"],
                lambda x: x > correlation_threshold,
                Variable.S_TYPE_CORR,
            ),
        )

    # Check recoded
    if config["check_recoded"] and "recoded" in correlations:
        # Overwrites the description with "RECORDED" series
        update(
            series_description,
            perform_check_correlation(
                correlations["recoded"], lambda x: x == 1, Variable.S_TYPE_RECODED
            ),
        )


    # Transform the series_description in a DataFrame
    variable_stats = pd.DataFrame(series_description)

    # Table statistics
    table_stats = describe_table(df, variable_stats)


    return {
        # Overall description
        "table": table_stats,

        # Per variable descriptions
        "variables": series_description,

        # Correlation matrices
        "correlations": correlations
    }
	
	