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
    url = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/HadCRUT.5.0.1.0.analysis.summary_series.global.annual.csv"

    df = pd.read_csv(url)

    df = df[["Time", "Anomaly (deg C)"]]
    df.columns = ["Year", "Temp"]

    df = df[df["Year"] >= 1925]

    return dict(zip(df["Year"], df["Temp"]))


def load_berkeley():
    url = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_summary.txt"

    df = pd.read_csv(
        url,
        delim_whitespace=True,
        comment="%",
        header=None
    )

    df = df[[0, 1]]
    df.columns = ["Year", "Temp"]

    df = df[df["Year"] >= 1925]

    return dict(zip(df["Year"], df["Temp"]))


def load_manual_obs():
    obs_datasets = {
        "NASA GISS (GISTEMP v4)": load_nasa_gistemp(),
        "HadCRUT5": load_hadcrut(),
        "Berkeley Earth": load_berkeley(),
    }

    return obs_datasets
