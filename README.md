# pandas_profiling_data
Pandas profiling code with only information as output. No HTML generated.

Since Pandas Profiling version 2.1.0 takes a lot of time escpecially in case of big data. Plus in our use case we did not need HTML report generation and other extra information it was best to just keep only the required information.

These scripts will only give these information in a dictionary:

            - table: overall statistics.
            - variables: descriptions per series.
            - correlations: correlation matrices.

You can always add additional information as per your requirements.

Here is the official Pandas_Profiling page: https://github.com/pandas-profiling/pandas-profiling
