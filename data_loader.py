import pandas as pd


def load_nasa_gistemp():
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

    df = pd.read_csv(url, skiprows=1)

    # Year, J-D 컬럼만 사용
    df = df[["Year", "J-D"]]

    # 숫자가 아닌 값 제거
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["J-D"], errors="coerce").notna()]

    df["Year"] = df["Year"].astype(int)
    df["J-D"] = df["J-D"].astype(float)

    # NASA 값은 보통 0.01°C 단위라서 100으로 나눔
    df["Temp"] = df["J-D"] / 100

    # 네 모델 범위에 맞게 1925년 이후만 사용
    df = df[df["Year"] >= 1925]

    return dict(zip(df["Year"], df["Temp"]))


def load_manual_obs():
    nasa_data = load_nasa_gistemp()

    obs_datasets = {
        "NASA GISS (GISTEMP v4)": nasa_data,
    }

    return obs_datasets
