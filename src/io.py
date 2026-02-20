from __future__ import annotations
import pandas as pd


def detect_file_type(filename: str) -> str:
    name = filename.lower().strip()
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return "excel"
    return "unknown"


def read_csv(uploaded_file, sep: str = ",", decimal: str = ".", encoding: str | None = None) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=sep, decimal=decimal, encoding=encoding)


def read_excel(uploaded_file, sheet_name=0) -> pd.DataFrame:
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)
