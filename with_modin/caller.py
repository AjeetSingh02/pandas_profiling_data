import time
import pandas as bpd
import modin.pandas as pd
from describe import describe
	
def caller(path):
    
    start_time = time.time()

    df = pd.read_csv(path)
    bdf = bpd.read_csv(path)
    parent_dict = describe(df, bdf)

    end_time = time.time()
    seconds = end_time - start_time

    return parent_dict, seconds