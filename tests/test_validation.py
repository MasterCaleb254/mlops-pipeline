import pandas as pd
import sys
sys.path.append("src")
from validation import validate

def test_clean_data_passes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert validate(df) == True

def test_missing_values_fails():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    assert validate(df) == False

def test_duplicates_fail():
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    assert validate(df) == False