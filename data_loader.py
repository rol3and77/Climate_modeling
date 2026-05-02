import pandas as pd
from functools import lru_cache


START_OBS_YEAR = 1925


def _clean_year_temp(df, year_col, temp_col):
    df = df[[year_col, temp_col]].copy()
    df.columns = ["Year", "Temp"]

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Temp"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(int)
    df["Temp"] = df["Temp"].astype(float)

    df = df[df["Year"] >= START_OBS_YEAR]
    return dict(zip(df["Year"], df["Temp"]))


@lru_cache(maxsize=1)
def load_nasa_gistemp():
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

    df = pd.read_csv(url, skiprows=1)
    return _clean_year_temp(df, "Year", "J-D")


@lru_cache(maxsize=1)
def load_hadcrut():
    url = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/analysis/diagnostics/HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv"

    df = pd.read_csv(url)

    return _clean_year_temp(df, "Time", "Anomaly (deg C)")
    

@lru_cache(maxsize=1)
def load_berkeley():
    url = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_summary.txt"

    df = pd.read_csv(
        url,
        sep=r"\s+",
        comment="%",
        header=None,
    )

    # Berkeley Earth annual summary:
    # column 0 = Year
    # column 1 = Annual anomaly
    df = df[[0, 1]]
    df.columns = ["Year", "Temp"]

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Temp"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(int)
    df["Temp"] = df["Temp"].astype(float)

    df = df[df["Year"] >= START_OBS_YEAR]

    return dict(zip(df["Year"], df["Temp"]))


def safe_load(name, loader):
    try:
        data = loader()
        if not data:
            print(f"[WARNING] {name} returned empty data.")
            return None
        return data
    except Exception as e:
        print(f"[WARNING] {name} load failed: {e}")
        return None


def load_manual_obs():
    obs_datasets = {}

    datasets = [
        ("NASA GISS (GISTEMP v4)", load_nasa_gistemp),
        ("HadCRUT5", load_hadcrut),
        ("Berkeley Earth", load_berkeley),
    ]

    for name, loader in datasets:
        data = safe_load(name, loader)
        if data is not None:
            obs_datasets[name] = data

    return obs_datasets
