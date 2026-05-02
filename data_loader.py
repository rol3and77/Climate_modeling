import pandas as pd
import numpy as np
from functools import lru_cache


START_OBS_YEAR = 1925
END_OBS_YEAR = 2025
COMMON_BASELINE_START = 1981
COMMON_BASELINE_END = 2010
MULTI_MEAN_NAME = "다중 관측 평균"


def _clean_year_temp(df, year_col, temp_col):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    df = df[[year_col, temp_col]].copy()
    df.columns = ["Year", "Temp"]

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Temp"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(float).astype(int)
    df["Temp"] = df["Temp"].astype(float)

    df = df[(df["Year"] >= START_OBS_YEAR) & (df["Year"] <= END_OBS_YEAR)]
    df = df.groupby("Year", as_index=False)["Temp"].mean()
    df = df.sort_values("Year")

    return dict(zip(df["Year"], df["Temp"]))


def rebase_temperature_anomaly(
    data,
    baseline_start=COMMON_BASELINE_START,
    baseline_end=COMMON_BASELINE_END,
):
    baseline_values = [
        temp
        for year, temp in data.items()
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


def _fallback_co2_history():
    """Fallback annual CO₂ pathway for years before the NOAA global series.

    Values before the instrumental global series are approximate anchors used only
    to avoid reverting to a purely linear 1925-2025 pathway when online data are
    unavailable. NOAA annual global values, when loaded, overwrite the modern
    part of the trajectory.
    """
    return {
        1925: 305.9,
        1930: 307.2,
        1940: 310.3,
        1950: 311.3,
        1958: 315.3,
        1960: 316.9,
        1970: 325.7,
        1980: 338.7,
        1990: 354.4,
        2000: 369.5,
        2010: 389.9,
        2020: 414.2,
        2025: 427.0,
    }


@lru_cache(maxsize=1)
def load_noaa_global_co2():
    """Load globally averaged annual atmospheric CO₂ from NOAA GML.

    NOAA's globally averaged marine-surface annual mean record begins around
    1980. The early 20th-century portion is filled with conservative historical
    anchors, then interpolated by the model.
    """
    co2_data = _fallback_co2_history()
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt"

    try:
        df = pd.read_csv(
            url,
            comment="#",
            sep=r"\s+",
            header=None,
            engine="python",
        )

        if df.shape[1] >= 2:
            df = df.iloc[:, :2].copy()
            df.columns = ["Year", "CO2"]
            df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
            df = df[pd.to_numeric(df["CO2"], errors="coerce").notna()]
            df["Year"] = df["Year"].astype(float).astype(int)
            df["CO2"] = df["CO2"].astype(float)
            df = df[(df["Year"] >= 1980) & (df["Year"] <= END_OBS_YEAR)]
            df = df.groupby("Year", as_index=False)["CO2"].mean().sort_values("Year")

            for year, value in zip(df["Year"], df["CO2"]):
                co2_data[int(year)] = float(value)

    except Exception as e:
        print(f"[WARNING] NOAA Global CO2 load failed; fallback anchors used: {e}")

    return dict(sorted(co2_data.items()))


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


def build_multi_dataset_mean(obs_datasets):
    base_names = [name for name in obs_datasets.keys() if name != MULTI_MEAN_NAME]

    if len(base_names) < 2:
        return None

    common_years = sorted(set.intersection(*(set(obs_datasets[name].keys()) for name in base_names)))
    common_years = [year for year in common_years if START_OBS_YEAR <= year <= END_OBS_YEAR]

    if not common_years:
        return None

    return {
        year: float(np.mean([obs_datasets[name][year] for name in base_names]))
        for year in common_years
    }


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

    multi_mean = build_multi_dataset_mean(obs_datasets)
    if multi_mean is not None:
        obs_datasets[MULTI_MEAN_NAME] = multi_mean

    return obs_datasets


def load_co2_observations():
    return load_noaa_global_co2()
