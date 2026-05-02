import pandas as pd


def load_nasa_gistemp():
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

    df = pd.read_csv(url, skiprows=1)
    df = df[["Year", "J-D"]]

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["J-D"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(int)
    df["Temp"] = df["J-D"].astype(float)

    df = df[df["Year"] >= 1925]

    return dict(zip(df["Year"], df["Temp"]))


def load_hadcrut():
    url = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv"

    df = pd.read_csv(url)

    # 컬럼명이 버전에 따라 다를 수 있어서 안전 처리
    year_col = df.columns[0]
    temp_col = df.columns[1]

    df = df[[year_col, temp_col]]
    df.columns = ["Year", "Temp"]

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Temp"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(int)
    df["Temp"] = df["Temp"].astype(float)

    df = df[df["Year"] >= 1925]

    return dict(zip(df["Year"], df["Temp"]))


def load_berkeley():
    url = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_summary.txt"

    df = pd.read_csv(
        url,
        sep=r"\s+",
        comment="%",
        header=None,
    )

    df = df[[0, 1]]
    df.columns = ["Year", "Temp"]

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Temp"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(int)
    df["Temp"] = df["Temp"].astype(float)

    df = df[df["Year"] >= 1925]

    return dict(zip(df["Year"], df["Temp"]))


def safe_load(name, loader):
    try:
        return loader()
    except Exception as e:
        print(f"[WARNING] {name} load failed: {e}")
        return None


def load_manual_obs():
    obs_datasets = {}

    nasa_data = safe_load("NASA GISS", load_nasa_gistemp)
    if nasa_data is not None:
        obs_datasets["NASA GISS (GISTEMP v4)"] = nasa_data

    hadcrut_data = safe_load("HadCRUT5", load_hadcrut)
    if hadcrut_data is not None:
        obs_datasets["HadCRUT5"] = hadcrut_data

    berkeley_data = safe_load("Berkeley Earth", load_berkeley)
    if berkeley_data is not None:
        obs_datasets["Berkeley Earth"] = berkeley_data

    return obs_datasets
