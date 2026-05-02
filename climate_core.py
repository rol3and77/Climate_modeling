import numpy as np
from scipy.optimize import minimize
from numba import njit


# ── Model constants ─────────────────────────────────────────────────────────
START_YEAR, END_YEAR = 1925, 2025
YEARS = END_YEAR - START_YEAR + 1
years_axis = np.arange(START_YEAR, END_YEAR + 1)

DT_SECONDS = 24 * 3600
LAND_FRAC, OCEAN_FRAC = 0.29, 0.71
C_LAND, C_MIXED, C_DEEP = 2.0e7, 1.2e8, 2.0e9

PARAM_NAMES = [
    "lambda_base",
    "aerosol_multiplier",
    "land_ocean_heat_exchange",
    "enso_amplitude",
    "volcanic_multiplier",
    "nonco2_multiplier",
    "solar_multiplier",
]


# ── Forcing functions ───────────────────────────────────────────────────────
def co2_forcing(co2_ppm, temperature_anomaly):
    """Calculate CO₂ radiative forcing with a weak temperature-dependent amplification."""
    co2_ppm = max(1.0, float(co2_ppm))
    temperature_anomaly = float(temperature_anomaly)
    return 5.35 * np.log(co2_ppm / 280.0) * (1.0 + 0.01 * max(0.0, temperature_anomaly))


def aerosol_effect(year, multiplier):
    """Approximate historical aerosol cooling as a piecewise-linear forcing trajectory."""
    base = np.interp(
        year,
        [1925, 1940, 1960, 1975, 1990, 2005, 2025],
        [0.0, -0.08, -0.35, -0.75, -0.95, -0.55, -0.25],
    )
    return multiplier * base


def solar_effect(year, multiplier):
    """Approximate the 11-year solar-cycle contribution as a small natural forcing term."""
    return multiplier * 0.05 * np.sin(2.0 * np.pi * (float(year) - START_YEAR) / 11.0)


def _normalize_params(params):
    """Accept the legacy 6-parameter vector and append the default solar multiplier."""
    if len(params) == 6:
        return list(params) + [1.0]
    if len(params) == 7:
        return list(params)
    raise ValueError(
        "params must contain 6 or 7 values: "
        "lambda_base, aer_mult, k_lo, enso_amp, volc_mult, nonco2_mult, [solar_mult]"
    )


def _build_co2_path(y_lin, end_year, end_co2, co2_history=None):
    """Create the CO₂ pathway used by the model.

    If co2_history is supplied, the historical segment is interpolated from the
    provided annual observations. Future years are linearly connected from the
    latest historical value to the scenario target end_co2.
    """
    if co2_history:
        co2_years = np.array(sorted(co2_history.keys()), dtype=float)
        co2_values = np.array([float(co2_history[int(y)]) for y in co2_years], dtype=float)

        valid = np.isfinite(co2_years) & np.isfinite(co2_values)
        co2_years = co2_years[valid]
        co2_values = co2_values[valid]

        if len(co2_years) >= 2:
            if end_year > co2_years[-1]:
                xp = np.append(co2_years, float(end_year))
                fp = np.append(co2_values, float(end_co2))
            else:
                xp = co2_years
                fp = co2_values
            return np.interp(y_lin, xp, fp)

    return np.interp(
        y_lin,
        np.array([1925.0, 2025.0, float(max(2025, end_year))]),
        np.array([306.0, 427.0, float(end_co2)]),
    )


@njit
def fast_core(
    total_steps, start_year, dt_seconds, land_frac, ocean_frac,
    c_land, c_mixed, c_deep, initial_temp,
    lambda_base, aer_mult, k_lo, enso_amp, volc_mult, nonco2_mult, solar_mult, co2_path,
):
    Tl = np.zeros(total_steps)
    Tm = np.zeros(total_steps)
    Td = np.zeros(total_steps)

    Tl[0] = initial_temp
    Tm[0] = initial_temp
    Td[0] = initial_temp

    y_arr = start_year + np.arange(total_steps) / 365.0

    xp_aer = np.array([1925.0, 1940.0, 1960.0, 1975.0, 1990.0, 2005.0, 2025.0])
    fp_aer = np.array([0.0, -0.08, -0.35, -0.75, -0.95, -0.55, -0.25])

    base_aer = np.interp(y_arr, xp_aer, fp_aer)
    aer_arr = aer_mult * base_aer

    f_non_co2 = nonco2_mult * ((y_arr - 1925.0) / 100.0) ** 2.2

    f_osc_arr = enso_amp * (
        np.sin(2.0 * np.pi * (y_arr - 1925.0) / 3.8)
        + 0.7 * np.sin(2.0 * np.pi * (y_arr - 1925.0) / 5.5)
        + 0.4 * np.sin(2.0 * np.pi * (y_arr - 1925.0) / 2.7)
    )

    # Small 11-year solar-cycle forcing. This is intentionally weak because the
    # long-term warming signal is primarily represented by anthropogenic forcing.
    f_solar_arr = solar_mult * 0.05 * np.sin(2.0 * np.pi * (y_arr - 1925.0) / 11.0)

    f_volc_arr = np.zeros(total_steps)
    volc_data = np.array([
        [1963.2, -0.8, 1.2],
        [1982.3, -1.3, 1.5],
        [1991.4, -1.8, 1.8],
    ])

    for v in range(3):
        ys, strength, decay = volc_data[v, 0], volc_data[v, 1], volc_data[v, 2]
        for i in range(total_steps):
            if y_arr[i] >= ys:
                f_volc_arr[i] += volc_mult * strength * np.exp(-(y_arr[i] - ys) / decay)

    co2_init = 5.35 * np.log(max(1.0, co2_path[0]) / 280.0) * (1.0 + 0.01 * max(0.0, initial_temp))
    aer_init = aer_mult * np.interp(float(start_year), xp_aer, fp_aer)
    solar_init = solar_mult * 0.05 * np.sin(2.0 * np.pi * (float(start_year) - 1925.0) / 11.0)
    forcing_offset = (lambda_base * initial_temp) - (co2_init + aer_init + solar_init)

    base_forcing = f_non_co2 + aer_arr + f_volc_arr + f_osc_arr + f_solar_arr + forcing_offset

    for i in range(total_steps - 1):
        curr_T = land_frac * Tl[i] + ocean_frac * Tm[i]
        dynamic_lambda = max(0.8, lambda_base - 0.15 * max(0.0, curr_T))

        f_co2 = 5.35 * np.log(max(1.0, co2_path[i]) / 280.0) * (1.0 + 0.01 * max(0.0, curr_T))
        total_forcing = f_co2 + base_forcing[i]

        h_lo = k_lo * (Tl[i] - Tm[i])
        h_md = 0.45 * (1.0 - 0.05 * max(0.0, Tm[i])) * (Tm[i] - Td[i])

        Tl[i + 1] = Tl[i] + ((total_forcing - dynamic_lambda * Tl[i] - h_lo) / c_land) * dt_seconds
        Tm[i + 1] = Tm[i] + ((total_forcing - dynamic_lambda * Tm[i] + h_lo - h_md) / c_mixed) * dt_seconds
        Td[i + 1] = Td[i] + (h_md / c_deep) * dt_seconds

        if Tl[i + 1] > 100.0:
            Tl[i + 1] = 100.0
        if Tm[i + 1] > 100.0:
            Tm[i + 1] = 100.0
        if Td[i + 1] > 100.0:
            Td[i + 1] = 100.0
        if Tl[i + 1] < -100.0:
            Tl[i + 1] = -100.0
        if Tm[i + 1] < -100.0:
            Tm[i + 1] = -100.0
        if Td[i + 1] < -100.0:
            Td[i + 1] = -100.0

    return Tl, Tm, Td


def run_model(params, init_temp, end_year=2025, end_co2=427, co2_history=None):
    """Run the simplified three-box energy balance model.

    params order:
    [lambda_base, aerosol_multiplier, land_ocean_exchange, enso_amplitude,
     volcanic_multiplier, nonco2_multiplier, solar_multiplier]
    """
    lambda_base, aer_mult, k_lo, enso_amp, volc_mult, nonco2_mult, solar_mult = _normalize_params(params)

    current_years_count = int(end_year - START_YEAR + 1)
    if current_years_count <= 0:
        raise ValueError("end_year must be greater than or equal to START_YEAR")

    total_steps = current_years_count * 365
    y_lin = np.linspace(START_YEAR, end_year, total_steps)
    co2_path = _build_co2_path(y_lin, end_year, end_co2, co2_history=co2_history)

    Tl, Tm, Td = fast_core(
        total_steps, START_YEAR, DT_SECONDS, LAND_FRAC, OCEAN_FRAC,
        C_LAND, C_MIXED, C_DEEP, float(init_temp),
        float(lambda_base), float(aer_mult), float(k_lo), float(enso_amp),
        float(volc_mult), float(nonco2_mult), float(solar_mult), co2_path,
    )

    daily_global = LAND_FRAC * Tl + OCEAN_FRAC * Tm

    return (
        daily_global.reshape(current_years_count, 365).mean(axis=1),
        Tl.reshape(current_years_count, 365).mean(axis=1),
        Tm.reshape(current_years_count, 365).mean(axis=1),
        Td.reshape(current_years_count, 365).mean(axis=1),
        daily_global,
    )


def get_optimized_params(obs_data, co2_history=None):
    """Estimate model parameters by minimizing mean squared error to observations."""
    obs_data = np.asarray(obs_data, dtype=float)

    if len(obs_data) != YEARS:
        raise ValueError(f"obs_data length must be {YEARS}, got {len(obs_data)}")

    init_temp = obs_data[0]

    def objective(params):
        model, _, _, _, _ = run_model(params, init_temp, co2_history=co2_history)
        return np.mean((model - obs_data) ** 2)

    starts = [
        [1.5, 1.0, 2.0, 0.08, 1.0, 0.75, 1.0],
        [1.0, 1.2, 1.5, 0.06, 0.8, 0.6, 0.8],
        [2.0, 0.8, 2.5, 0.10, 1.2, 0.9, 1.2],
        [1.3, 1.5, 3.0, 0.05, 0.6, 0.5, 0.6],
        [1.8, 0.7, 1.0, 0.12, 1.5, 1.0, 1.5],
    ]

    bounds = [
        (0.7, 2.3),   # climate feedback parameter
        (0.5, 2.0),   # aerosol multiplier
        (0.5, 3.5),   # land-ocean heat exchange
        (0.03, 0.15), # ENSO amplitude
        (0.3, 2.0),   # volcanic forcing multiplier
        (0.3, 1.5),   # non-CO₂ forcing multiplier
        (0.0, 2.0),   # solar forcing multiplier
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
