import pandas as pd
from functools import lru_cache


START_OBS_YEAR = 1925
COMMON_BASELINE_START = 1981
COMMON_BASELINE_END = 2010


def _clean_year_temp(df, year_col, temp_col):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    df = df[[year_col, temp_col]].copy()
    df.columns = ["Year", "Temp"]

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Temp"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(float).astype(int)
    df["Temp"] = df["Temp"].astype(float)

    df = df[df["Year"] >= START_OBS_YEAR]
    df = df.groupby("Year", as_index=False)["Temp"].mean()
    df = df.sort_values("Year")

    return dict(zip(df["Year"], df["Temp"]))


def rebase_temperature_anomaly(data, baseline_start=COMMON_BASELINE_START, baseline_end=COMMON_BASELINE_END):
    baseline_values = [
        temp for year, temp in data.items()
        if baseline_start <= year <= baseline_end
    ]

    if len(baseline_values) == 0:
        return data

    baseline_mean = sum(baseline_values) / len(baseline_values)

    return {
        year: temp - baseline_mean
        for year, temp in data.items()
    }


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

    df = df[[0, 1]]
    df.columns = ["Year", "Temp"]

    return _clean_year_temp(df, "Year", "Temp")


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
            obs_datasets[name] = rebase_temperature_anomaly(data)

    if not obs_datasets:
        raise RuntimeError("No observational datasets could be loaded.")

    return obs_datasets
