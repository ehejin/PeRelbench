import pandas as pd

from relbench.data import Table


def test_table():
    table = Table(df=pd.DataFrame(), fkey_col_to_pkey_table={})
    assert len(table) == 0
