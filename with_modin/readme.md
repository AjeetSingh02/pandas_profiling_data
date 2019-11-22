These scripts are using modin.pandas instead of simple pandas.

Modin is said to be a faster implementation of pandas.

You can find further information here: https://github.com/modin-project/modin

In this case of Pandas Profiling, Modin is not that helpful.

The timing for scripts with modin is more than without modin because for every pandas function whose implementation 
is not present in modin.pandas it uses default pandas. And that switching takes some time.

