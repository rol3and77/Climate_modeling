import numpy as np
from scipy.optimize import minimize
from numba import njit


# ── 관측 데이터 ───────────────────────────────────────────────────────────────
START_YEAR, END_YEAR = 1925, 2025
YEARS = END_YEAR - START_YEAR + 1
years_axis = np.arange(START_YEAR, END_YEAR + 1)
dt = 24 * 3600
land_frac, ocean_frac = 0.29, 0.71
C_land, C_mixed, C_deep = 2.0e7, 1.2e8, 2.0e9


# ── 물리 함수 (핵심 계산 – 변경 금지) ─────────────────────────────────────────
@st.cache_data
def co2_forcing(C, T):
    return 5.35 * np.log(C / 280.0) * (1 + 0.01 * max(0, T))


@st.cache_data
def aerosol_effect(y, mult):
    base = np.interp(
        y,
        [1925, 1940, 1960, 1975, 1990, 2005, 2025],
        [0.0, -0.08, -0.35, -0.75, -0.95, -0.55, -0.25],
    )
    return mult * base

@njit
def fast_core(
    total_steps, START_YEAR, dt, land_frac, ocean_frac,
    C_land, C_mixed, C_deep, INITIAL_TEMP,
    lambda_base, aer_mult, k_lo, enso_amp, volc_mult, nonco2_mult, co2_path,
):
    Tl = np.zeros(total_steps)
    Tm = np.zeros(total_steps)
    Td = np.zeros(total_steps)
    Tl[0] = Tm[0] = Td[0] = INITIAL_TEMP
    y_arr = START_YEAR + np.arange(total_steps) / 365.0
    xp_aer = np.array([1925.0, 1940.0, 1960.0, 1975.0, 1990.0, 2005.0, 2025.0])
    fp_aer = np.array([0.0, -0.08, -0.35, -0.75, -0.95, -0.55, -0.25])
    base_aer = np.interp(y_arr, xp_aer, fp_aer)
    aer_arr = aer_mult * base_aer
    f_non_co2 = nonco2_mult * ((y_arr - 1925.0) / 100.0) ** 2.2
    f_osc_arr = enso_amp * (
        np.sin(2 * np.pi * (y_arr - 1925) / 3.8)
        + 0.7 * np.sin(2 * np.pi * (y_arr - 1925) / 5.5)
        + 0.4 * np.sin(2 * np.pi * (y_arr - 1925) / 2.7)
    )
    f_volc_arr = np.zeros(total_steps)
    volc_data = np.array([
        [1963.2, -0.8, 1.2],
        [1982.3, -1.3, 1.5],
        [1991.4, -1.8, 1.8],
    ])
    for v in range(3):
        ys, s, d = volc_data[v, 0], volc_data[v, 1], volc_data[v, 2]
        for i in range(total_steps):
            if y_arr[i] >= ys:
                f_volc_arr[i] += volc_mult * s * np.exp(-(y_arr[i] - ys) / d)
    co2_init = 5.35 * np.log(max(1.0, 306.0) / 280.0) * (
        1.0 + 0.01 * max(0.0, INITIAL_TEMP)
    )
    aer_init = aer_mult * np.interp(float(START_YEAR), xp_aer, fp_aer)
    F_offset = (lambda_base * INITIAL_TEMP) - (co2_init + aer_init)
    base_forcing = f_non_co2 + aer_arr + f_volc_arr + f_osc_arr + F_offset
    for i in range(total_steps - 1):
        curr_T = land_frac * Tl[i] + ocean_frac * Tm[i]
        dynamic_lambda = max(0.8, lambda_base - 0.15 * max(0.0, curr_T))
        f_co2 = 5.35 * np.log(max(1.0, co2_path[i]) / 280.0) * (
            1.0 + 0.01 * max(0.0, curr_T)
        )
        total_f = f_co2 + base_forcing[i]
        h_lo = k_lo * (Tl[i] - Tm[i])
        h_md = 0.45 * (1.0 - 0.05 * max(0.0, Tm[i])) * (Tm[i] - Td[i])
        Tl[i + 1] = Tl[i] + (
            (total_f - dynamic_lambda * Tl[i] - h_lo) / C_land
        ) * dt
        Tm[i + 1] = Tm[i] + (
            (total_f - dynamic_lambda * Tm[i] + h_lo - h_md) / C_mixed
        ) * dt
        Td[i + 1] = Td[i] + (h_md / C_deep) * dt
        if Tl[i + 1] > 100.0:
            Tl[i + 1] = 100.0
        if Tm[i + 1] > 100.0:
            Tm[i + 1] = 100.0
        if Td[i + 1] > 100.0:
            Td[i + 1] = 100.0
    return Tl, Tm, Td


@st.cache_data
def run_model(params, init_temp, end_year=2025, end_co2=427):
    lambda_base, aer_mult, k_lo, enso_amp, volc_mult, nonco2_mult = params
    current_years_count = int(end_year - START_YEAR + 1)
    total_steps = current_years_count * 365
    y_lin = np.linspace(START_YEAR, end_year, total_steps)
    co2_path = np.interp(
        y_lin,
        np.array([1925.0, 2025.0, float(max(2025, end_year))]),
        np.array([306.0, 427.0, float(end_co2)]),
    )
    Tl, Tm, Td = fast_core(
        total_steps, START_YEAR, dt, land_frac, ocean_frac,
        C_land, C_mixed, C_deep, init_temp,
        lambda_base, aer_mult, k_lo, enso_amp, volc_mult, nonco2_mult, co2_path,
    )
    daily_res = land_frac * Tl + ocean_frac * Tm
    return (
        daily_res.reshape(current_years_count, 365).mean(axis=1),
        Tl.reshape(current_years_count, 365).mean(axis=1),
        Tm.reshape(current_years_count, 365).mean(axis=1),
        Td.reshape(current_years_count, 365).mean(axis=1),
        daily_res,
    )


@st.cache_data
def get_optimized_params(obs_data):
    init_temp = obs_data[0]

    def objective(params):
        m, _, _, _, _ = run_model(params, init_temp)
        return np.mean((m - obs_data) ** 2)

    starts = [
        [1.5, 1.0, 2.0, 0.08, 1.0, 0.75],
        [1.0, 1.2, 1.5, 0.06, 0.8, 0.6],
        [2.0, 0.8, 2.5, 0.10, 1.2, 0.9],
        [1.3, 1.5, 3.0, 0.05, 0.6, 0.5],
        [1.8, 0.7, 1.0, 0.12, 1.5, 1.0],
    ]

    bounds = [
        (0.7, 2.3),   # lambda_base
        (0.5, 2.0),   # aerosol multiplier
        (0.5, 3.5),   # land-ocean heat exchange
        (0.03, 0.15), # ENSO amplitude
        (0.3, 2.0),   # volcanic forcing multiplier
        (0.3, 1.5),   # nonco2_mult
    ]

    best_res = None

    for start in starts:
        res = minimize(
            objective,
            start,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )

        if best_res is None or res.fun < best_res.fun:
            best_res = res

    return best_res.x

@st.cache_data
def load_report_file():
    report_candidates = [
        Path("기후모델 웹사이트 분석 리포트.docx"),
        Path("./기후모델 웹사이트 분석 리포트.docx"),
    ]
    for path in report_candidates:
        if path.exists():
            return path.name, path.read_bytes()
    return None, None
