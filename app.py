import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from numba import njit
from pathlib import Path

# --- 1. 웹 대시보드 설정 ---
st.set_page_config(page_title="기후 모델링 연구 대시보드", layout="wide")

# --- 1-1. 전역 스타일 ---
st.markdown("""
<style>
    .hero-wrap {
        padding: 4rem 2.2rem 3rem 2.2rem;
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(30,41,59,0.98) 58%, rgba(51,65,85,0.96) 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.4rem;
    }

    .hero-kicker {
        font-size: 0.86rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #cbd5e1;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: 2.65rem;
        font-weight: 800;
        line-height: 1.18;
        margin-bottom: 0.9rem;
        max-width: 920px;
    }

    .hero-desc {
        font-size: 1.02rem;
        line-height: 1.9;
        color: #e2e8f0;
        max-width: 940px;
    }

    .hero-note {
        margin-top: 1.25rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.12);
        font-size: 0.95rem;
        color: #cbd5e1;
    }

    .tag-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 1.05rem;
    }

    .tag-chip {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: #e2e8f0;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        font-size: 0.87rem;
        font-weight: 600;
    }

    .section-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .subsection-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.05rem 1rem 0.95rem 1rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        min-height: 118px;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.35rem;
        line-height: 1.5;
    }

    .metric-num {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }

    .metric-note {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    .paper-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.25rem 1.25rem 1.15rem 1.25rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        min-height: 200px;
        margin-bottom: 1rem;
    }

    .paper-index {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.55rem;
    }

    .paper-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.65rem;
        line-height: 1.45;
    }

    .paper-desc {
        font-size: 0.95rem;
        line-height: 1.75;
        color: #475569;
    }

    .abstract-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #334155;
        border-radius: 18px;
        padding: 1.2rem 1.2rem 1.05rem 1.2rem;
        margin-top: 0.3rem;
        margin-bottom: 1.1rem;
    }

    .abstract-label {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .abstract-text {
        font-size: 0.96rem;
        line-height: 1.85;
        color: #334155;
    }

    .mini-note {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        font-size: 0.93rem;
        color: #475569;
        line-height: 1.7;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 설정 및 관측 데이터 ---
START_YEAR, END_YEAR = 1925, 2025
YEARS = END_YEAR - START_YEAR + 1
years_axis = np.arange(START_YEAR, END_YEAR + 1)
dt = 24 * 3600
land_frac, ocean_frac = 0.29, 0.71
C_land, C_mixed, C_deep = 2.0e7, 1.2e8, 2.0e9

obs_datasets = {
    "NASA GISS (GISTEMP v4)": {
        1925:-0.22, 1940:0.10, 1950:-0.18, 1965:-0.10, 1980:0.26, 1990:0.48, 2000:0.62, 2010:0.88, 2020:1.18, 2025:1.38
    },
    "Berkeley Earth (Land + Ocean)": {
        1925:-0.18, 1940:0.12, 1950:-0.15, 1965:-0.08, 1980:0.28, 1990:0.51, 2000:0.65, 2010:0.92, 2020:1.24, 2025:1.42
    },
    "Hadley Centre (HadCRUT5)": {
        1925:-0.25, 1940:0.08, 1950:-0.20, 1965:-0.12, 1980:0.22, 1990:0.45, 2000:0.60, 2010:0.85, 2020:1.15, 2025:1.35
    }
}

# --- 3. 물리 함수 ---
@st.cache_data
def co2_forcing(C, T):
    return 5.35 * np.log(C / 280.0) * (1 + 0.01 * max(0, T))

@st.cache_data
def aerosol_effect(y, mult):
    base = np.interp(y, [1925, 1960, 1985, 2025], [0, -0.4, -0.9, -0.15])
    return mult * (base - 0.12 * np.abs(base)**0.5)

@njit
def fast_core(total_steps, START_YEAR, dt, land_frac, ocean_frac, C_land, C_mixed, C_deep, INITIAL_TEMP, lambda_base, aer_mult, k_lo, enso_amp, co2_path):
    Tl = np.zeros(total_steps)
    Tm = np.zeros(total_steps)
    Td = np.zeros(total_steps)
    Tl[0] = Tm[0] = Td[0] = INITIAL_TEMP
    y_arr = START_YEAR + np.arange(total_steps) / 365.0
    xp_aer = np.array([1925.0, 1960.0, 1985.0, 2025.0])
    fp_aer = np.array([0.0, -0.4, -0.9, -0.15])
    base_aer = np.interp(y_arr, xp_aer, fp_aer)
    aer_arr = aer_mult * (base_aer - 0.12 * np.abs(base_aer)**0.5)
    f_non_co2 = 0.75 * ((y_arr - 1925.0) / 100.0)**2.2
    f_osc_arr = enso_amp * (np.sin(2 * np.pi * y_arr / 3.8) + np.sin(2 * np.pi * y_arr / 5.5))
    f_volc_arr = np.zeros(total_steps)
    volc_data = np.array([[1963.2, -0.8, 1.2], [1982.3, -1.3, 1.5], [1991.4, -1.8, 1.8]])
    for v in range(3):
        ys, s, d = volc_data[v, 0], volc_data[v, 1], volc_data[v, 2]
        for i in range(total_steps):
            if y_arr[i] >= ys:
                f_volc_arr[i] += s * np.exp(-(y_arr[i] - ys) / d)
    co2_init = 5.35 * np.log(max(1.0, 306.0) / 280.0) * (1.0 + 0.01 * max(0.0, INITIAL_TEMP))
    aer_init = aer_mult * (np.interp(float(START_YEAR), xp_aer, fp_aer) - 0.12 * np.abs(np.interp(float(START_YEAR), xp_aer, fp_aer))**0.5)
    F_offset = (lambda_base * INITIAL_TEMP) - (co2_init + aer_init)
    base_forcing = f_non_co2 + aer_arr + f_volc_arr + f_osc_arr + F_offset
    for i in range(total_steps - 1):
        curr_T = land_frac * Tl[i] + ocean_frac * Tm[i]
        dynamic_lambda = max(0.1, lambda_base - 0.15 * max(0.0, curr_T))
        f_co2 = 5.35 * np.log(max(1.0, co2_path[i]) / 280.0) * (1.0 + 0.01 * max(0.0, curr_T))
        total_f = f_co2 + base_forcing[i]
        h_lo = k_lo * (Tl[i] - Tm[i])
        h_md = 0.45 * (1.0 - 0.05 * max(0.0, Tm[i])) * (Tm[i] - Td[i])
        Tl[i+1] = Tl[i] + ((total_f - dynamic_lambda * Tl[i] - h_lo) / C_land) * dt
        Tm[i+1] = Tm[i] + ((total_f - dynamic_lambda * Tm[i] + h_lo - h_md) / C_mixed) * dt
        Td[i+1] = Td[i] + (h_md / C_deep) * dt
        if Tl[i+1] > 100.0:
            Tl[i+1] = 100.0
        if Tm[i+1] > 100.0:
            Tm[i+1] = 100.0
        if Td[i+1] > 100.0:
            Td[i+1] = 100.0
    return Tl, Tm, Td

@st.cache_data
def run_model(params, init_temp, end_year=2025, end_co2=427):
    lambda_base, aer_mult, k_lo, enso_amp = params
    current_years_count = int(end_year - START_YEAR + 1)
    total_steps = current_years_count * 365
    y_lin = np.linspace(START_YEAR, end_year, total_steps)
    co2_path = np.interp(
        y_lin,
        np.array([1925.0, 2025.0, float(max(2025, end_year))]),
        np.array([306.0, 427.0, float(end_co2)])
    )
    Tl, Tm, Td = fast_core(total_steps, START_YEAR, dt, land_frac, ocean_frac, C_land, C_mixed, C_deep, init_temp, lambda_base, aer_mult, k_lo, enso_amp, co2_path)
    daily_res = (land_frac * Tl + ocean_frac * Tm)
    return (
        daily_res.reshape(current_years_count, 365).mean(axis=1),
        Tl.reshape(current_years_count, 365).mean(axis=1),
        Tm.reshape(current_years_count, 365).mean(axis=1),
        Td.reshape(current_years_count, 365).mean(axis=1),
        daily_res
    )

@st.cache_data
def get_optimized_params(obs_data):
    init_temp = obs_data[0]
    def objective(params):
        m, _, _, _, _ = run_model(params, init_temp)
        return np.mean((m - obs_data)**2)
    res = minimize(
        objective,
        [1.5, 1.0, 2.0, 0.12],
        bounds=[(0.7, 2.3), (0.5, 2.0), (0.5, 3.5), (0.05, 0.25)],
        method='L-BFGS-B',
        options={'maxiter': 15, 'ftol': 1e-4}
    )
    return res.x

@st.cache_data
def load_report_file():
    report_candidates = [
        Path("기후모델 웹사이트 분석 리포트.docx"),
        Path("./기후모델 웹사이트 분석 리포트.docx"),
        Path("기후모델 웹사이트 분석 리포트.docx"),
        Path("./기후모델 웹사이트 분석 리포트.docx"),
    ]
    for path in report_candidates:
        if path.exists():
            return path.name, path.read_bytes()
    return None, None

def render_section_note(title, body):
    st.markdown(
        f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #334155;
            border-radius: 14px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 1rem;
        ">
            <div style="
                font-size: 0.92rem;
                font-weight: 800;
                color: #334155;
                margin-bottom: 0.35rem;
            ">{title}</div>
            <div style="
                font-size: 0.96rem;
                line-height: 1.8;
                color: #475569;
            ">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_metric_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{title}</div>
            <div class="metric-num">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 4. 사이드바 구성 ---
st.sidebar.title("기후 모델링 연구 대시보드")

if "page" not in st.session_state:
    st.session_state["page"] = "시작 페이지"

if st.sidebar.button("시작 페이지", use_container_width=True):
    st.session_state["page"] = "시작 페이지"
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("탐색")

page_options = [
    "시나리오 기반 기후 변화 예측",
    "기후 시스템 파라미터 실험",
    "모델 적합도 및 관측자료 비교",
    "모델 검증 및 불확실성 정량화",
    "기후 모델링 용어 및 개념 정의",
    "연구 요약 및 보고서"
]

if "sidebar_selected_page" not in st.session_state:
    st.session_state["sidebar_selected_page"] = page_options[0]

if st.session_state["page"] in page_options:
    st.session_state["sidebar_selected_page"] = st.session_state["page"]

selected_page = st.sidebar.radio(
    "섹션 선택",
    page_options,
    index=page_options.index(st.session_state["sidebar_selected_page"])
)

if selected_page != st.session_state["sidebar_selected_page"]:
    st.session_state["sidebar_selected_page"] = selected_page
    st.session_state["page"] = selected_page
    st.rerun()

page = st.session_state["page"]

st.sidebar.markdown("---")
st.sidebar.header("설정")

if page == "시나리오 기반 기후 변화 예측":
    scenario_options = [
        "탄소중립 안정화 시나리오",
        "저온난화 경로 시나리오",
        "현재 정책 유지 시나리오",
        "고배출 시나리오",
        "극단적 배출 시나리오"
    ]
    policy = st.sidebar.select_slider("배출 시나리오 선택", options=scenario_options, value="현재 정책 유지 시나리오")

elif page == "기후 시스템 파라미터 실험":
    def reset_experiment():
        st.session_state["exp_co2_slider"] = 550
        st.session_state["exp_lambda_slider"] = 1.5
        st.session_state["exp_aer_slider"] = 1.0
        st.session_state["exp_klo_slider"] = 2.0
        st.session_state["exp_enso_slider"] = 0.12
    st.sidebar.button("실험 설정 초기화", on_click=reset_experiment)
    st.sidebar.subheader("실험 파라미터")
    exp_co2 = st.sidebar.slider("2100년 CO2 농도 (ppm)", 250, 1500, 550, step=10, key="exp_co2_slider")
    exp_lambda = st.sidebar.slider("기후 피드백 파라미터", 0.5, 3.0, 1.5, step=0.1, key="exp_lambda_slider")
    exp_aer = st.sidebar.slider("에어로졸 강도", 0.0, 3.0, 1.0, step=0.1, key="exp_aer_slider")
    exp_klo = st.sidebar.slider("해양 열흡수 계수", 0.5, 4.0, 2.0, step=0.1, key="exp_klo_slider")
    exp_enso = st.sidebar.slider("ENSO 진폭", 0.0, 0.5, 0.12, step=0.01, key="exp_enso_slider")

elif page == "모델 적합도 및 관측자료 비교":
    obs_choice = st.sidebar.selectbox("관측 데이터셋 선택", list(obs_datasets.keys()))
    current_obs_data = np.interp(years_axis, list(obs_datasets[obs_choice].keys()), list(obs_datasets[obs_choice].values()))
    target_year = st.sidebar.slider("일별 상세 분석 연도 선택", 1925, 2025, 2024)

elif page == "모델 검증 및 불확실성 정량화":
    diag_obs_choice = st.sidebar.selectbox("검증용 데이터셋 선택", list(obs_datasets.keys()))
    diag_obs_data = np.interp(years_axis, list(obs_datasets[diag_obs_choice].keys()), list(obs_datasets[diag_obs_choice].values()))
    sens_param = st.sidebar.selectbox(
        "민감도 분석 파라미터",
        [
            "기후 피드백 파라미터",
            "에어로졸 강도",
            "해양 열흡수 계수",
            "ENSO 진폭"
        ]
    )
else:
    st.sidebar.markdown(
        """
        <div class="mini-note">
            현재 페이지는 설명형 또는 요약형 페이지입니다. 좌측 섹션 메뉴에서 원하는 분석 페이지로 이동하면 해당 설정이 활성화됩니다.
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar.expander("자료 출처", expanded=False):
    st.caption("본 모델은 주요 공인 데이터를 바탕으로 구성되었습니다.")
    st.markdown("""
    - **기온 관측 데이터**: [NASA GISS GISTEMP v4](https://data.giss.nasa.gov/gistemp/)
    - **CO2 농도 및 시나리오 참조**: [IPCC 제6차 평가보고서 AR6](https://www.ipcc.ch/report/ar6/wg1/)
    - **대기 CO2 실측값**: [NOAA Global Monitoring Laboratory](https://gml.noaa.gov/ccgg/trends/)
    - **화산 강제력 자료**: [Smithsonian Institution Global Volcanism Program](https://volcano.si.edu/)
    - **해수면 해석 자료**: [NASA Sea Level Change Portal](https://sealevel.nasa.gov/)
    """)

# --- 5. 페이지 렌더링 ---
if page == "시작 페이지":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-kicker">Research Presentation Interface</div>
        <div class="hero-title">기후 모델링 연구 대시보드</div>
        <div class="hero-desc">
            관측 자료와 물리 기반 기후 모형을 결합하여 역사적 온도 변화를 재현하고,
            미래 배출 시나리오에 따른 전지구 평균기온의 장기 경로를 탐색하기 위한 연구형 대시보드입니다.
        </div>
        <div class="tag-row">
            <div class="tag-chip">물리 기반 모델</div>
            <div class="tag-chip">관측자료 비교</div>
            <div class="tag-chip">불확실성 분석</div>
        </div>
        <div class="hero-note">
            학술 발표, 모델 해석, 시나리오 기반 기후 분석을 위해 설계된 인터페이스입니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="abstract-box">
        <div class="abstract-label">Overview</div>
        <div class="abstract-text">
            본 대시보드는 단순한 시각화 도구를 넘어, 기후 시스템의 주요 강제력과 반응 변수를
            하나의 흐름 안에서 검토할 수 있도록 구성되어 있습니다. 사용자는 시나리오 예측,
            파라미터 실험, 관측자료 비교, 불확실성 분석을 통해 모델의 구조와 해석 가능성을
            단계적으로 탐색할 수 있습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>핵심 결과 미리보기</div>", unsafe_allow_html=True)

    preview_res, _, _, _, _ = run_model([1.5, 1.0, 2.0, 0.12], -0.22, end_year=2100, end_co2=550)
    preview_2100 = preview_res[-1]
    preview_rmse = np.sqrt(np.mean((
        run_model([1.5, 1.0, 2.0, 0.12], -0.22)[0] -
        np.interp(
            years_axis,
            list(obs_datasets["NASA GISS (GISTEMP v4)"].keys()),
            list(obs_datasets["NASA GISS (GISTEMP v4)"].values())
        )
    )**2))
    preview_slr = preview_2100 * 35

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("2100년 예상 온난화", f"+{preview_2100:.2f} °C", "현재 정책 유지 시나리오 기준")
    with c2:
        render_metric_card("예상 해수면 상승", f"+{preview_slr:.1f} cm", "단순 비례 가정 기반 참고값")
    with c3:
        render_metric_card("기본 모델 RMSE", f"{preview_rmse:.3f} °C", "관측자료와의 기본 적합도")

    st.markdown("<div class='section-title'>온도 변화 개요</div>", unsafe_allow_html=True)

    fig_home, ax_home = plt.subplots(figsize=(12, 4.8))
    obs_preview = np.interp(
        years_axis,
        list(obs_datasets["NASA GISS (GISTEMP v4)"].keys()),
        list(obs_datasets["NASA GISS (GISTEMP v4)"].values())
    )
    ax_home.plot(years_axis, obs_preview, color='black', lw=2, label='Observed Temperature')
    ax_home.plot(np.arange(1925, 2101), preview_res, color='crimson', lw=2.2, label='Reference Projection')
    ax_home.axhline(1.5, color='orange', ls='--', label='1.5 C Threshold')
    ax_home.set_title("Global Temperature Trend Overview")
    ax_home.set_xlabel("Year")
    ax_home.set_ylabel("Temperature Anomaly (C)")
    ax_home.grid(True, alpha=0.25)
    ax_home.legend(loc='upper left')
    st.pyplot(fig_home)

    render_section_note(
        "시작 페이지 안내",
        "이 첫 화면은 전체 프로젝트의 방향과 핵심 결과를 빠르게 확인하기 위한 요약 페이지입니다. "
        "보다 자세한 해석은 각 분석 페이지에서 확인할 수 있으며, 시나리오별 비교·파라미터 실험·적합도 검증·불확실성 분석으로 이어집니다."
    )

    st.markdown("<div class='section-title'>주요 모듈</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="paper-card">
            <div class="paper-index">Module 01</div>
            <div class="paper-title">시나리오 기반 기후 변화 예측</div>
            <div class="paper-desc">
                탄소중립 안정화부터 고배출 경로까지 다양한 배출 시나리오를 설정하여
                2100년까지의 전지구 평균기온 및 해수면 상승 경향을 비교합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="paper-card">
            <div class="paper-index">Module 02</div>
            <div class="paper-title">기후 시스템 파라미터 실험</div>
            <div class="paper-desc">
                기후 피드백 파라미터, 에어로졸 강도, 해양 열흡수 계수, ENSO 진폭을 직접 조정해
                모델 응답이 어떻게 달라지는지 실험할 수 있습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="paper-card">
            <div class="paper-index">Module 03</div>
            <div class="paper-title">모델 적합도 및 관측자료 비교</div>
            <div class="paper-desc">
                관측 자료와 모델 출력의 차이를 시계열, 상대오차, 강제력 기여 요소로 분해하여 제시합니다.
                모델이 역사적 기후 변화를 어느 정도 설명하는지 평가합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="paper-card">
            <div class="paper-index">Module 04</div>
            <div class="paper-title">모델 검증 및 불확실성 정량화</div>
            <div class="paper-desc">
                잔차 진단, 불확실성 범위, 민감도 분석을 통해 예측 신뢰성과 구조적 특성을 검토합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>빠른 탐색</div>", unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button("시나리오 분석", use_container_width=True):
            st.session_state["page"] = "시나리오 기반 기후 변화 예측"
            st.rerun()

    with b2:
        if st.button("파라미터 실험", use_container_width=True):
            st.session_state["page"] = "기후 시스템 파라미터 실험"
            st.rerun()

    with b3:
        if st.button("모델 검증", use_container_width=True):
            st.session_state["page"] = "모델 검증 및 불확실성 정량화"
            st.rerun()

    with b4:
        if st.button("연구 요약", use_container_width=True):
            st.session_state["page"] = "연구 요약 및 보고서"
            st.rerun()

    st.caption("좌측 사이드바 또는 위 버튼을 통해 각 분석 페이지로 이동할 수 있습니다.")

elif page == "시나리오 기반 기후 변화 예측":
    st.title("시나리오 기반 기후 변화 예측")
    render_section_note(
        "분석 목적",
        "이 섹션은 서로 다른 배출 시나리오에 따라 장기 온난화 경로가 어떻게 달라지는지를 비교하기 위한 페이지입니다. "
        "1.5도와 2.0도 임계선과의 관계를 함께 제시하여 각 시나리오의 상대적 기후 위험 수준을 해석할 수 있도록 구성했습니다."
    )

    emission_map = {
        "탄소중립 안정화 시나리오": 280,
        "저온난화 경로 시나리오": 380,
        "현재 정책 유지 시나리오": 550,
        "고배출 시나리오": 850,
        "극단적 배출 시나리오": 1500
    }

    res_full, _, _, _, _ = run_model([1.5, 1.0, 2.0, 0.12], -0.22, end_year=2100, end_co2=emission_map[policy])
    p_2100 = res_full[-1]
    trend_21c = np.polyfit(np.arange(1925, 2101), res_full, 1)[0]

    st.markdown("<div class='section-title'>핵심 결과</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("2100년 예상 온난화", f"+{p_2100:.2f} °C", "선택한 시나리오 기준 장기 예측값")
    with c2:
        render_metric_card("2100년 예상 해수면 상승", f"+{p_2100*35:.1f} cm", "단순 비례 가정에 따른 참고 지표")
    with c3:
        render_metric_card("장기 평균 온난화 속도", f"{trend_21c:.3f} °C/year", "1925–2100 전체 구간 평균 추세")

    fig_fut, ax_fut = plt.subplots(figsize=(12, 5.5))
    ax_fut.plot(
        years_axis,
        np.interp(years_axis, list(obs_datasets["NASA GISS (GISTEMP v4)"].keys()), list(obs_datasets["NASA GISS (GISTEMP v4)"].values())),
        'k',
        label='Historical Observation'
    )
    years_full = np.arange(1925, 2101)
    ax_fut.plot(years_full[len(years_axis)-1:], res_full[len(years_axis)-1:], 'r--', lw=2, label='Projected Response')
    ax_fut.axhline(1.5, color='orange', ls='--', label='1.5 C Threshold')
    ax_fut.axhline(2.0, color='red', ls='--', label='2.0 C Threshold')
    ax_fut.set_xlabel("Year")
    ax_fut.set_ylabel("Temperature Anomaly (C)")
    ax_fut.set_title("Projected Global Temperature Trajectory")
    ax_fut.legend()
    ax_fut.grid(True, alpha=0.2)
    st.pyplot(fig_fut)

    render_section_note(
        "해석",
        "고배출에 가까운 경로일수록 온도 상승 속도가 빠르게 커지며, 임계 온도 도달 시점도 앞당겨집니다. "
        "다만 본 결과는 전지구 평균 기반의 단순화된 모델에서 도출된 것으로, 정밀 예측값이라기보다 시나리오 간 상대적 차이와 장기 경향을 비교하는 해석용 결과로 보는 것이 적절합니다."
    )

elif page == "기후 시스템 파라미터 실험":
    st.title("기후 시스템 파라미터 실험")
    render_section_note(
        "분석 목적",
        "이 페이지는 기후 피드백 강도, 에어로졸 냉각, 해양 열흡수, 내부 변동성과 같은 핵심 파라미터가 "
        "장기 온난화 결과에 어떤 영향을 주는지 탐색하기 위해 구성되었습니다."
    )

    custom_params = [exp_lambda, exp_aer, exp_klo, exp_enso]
    res_exp, tl_exp, tm_exp, td_exp, _ = run_model(custom_params, -0.22, end_year=2100, end_co2=exp_co2)

    st.markdown("<div class='section-title'>핵심 결과</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("2100년 육지 온도 상승", f"+{tl_exp[-1]:.2f} °C", "육지는 상대적으로 빠르게 반응")
    with c2:
        render_metric_card("2100년 해양 표층 온도 상승", f"+{tm_exp[-1]:.2f} °C", "표층 해양의 완충 효과 반영")
    with c3:
        render_metric_card("2100년 심해 온도 상승", f"+{td_exp[-1]:.2f} °C", "심해는 가장 느리게 반응")

    fig_exp, ax_exp = plt.subplots(figsize=(12, 5.5))
    years_exp = np.arange(1925, 2101)
    ax_exp.plot(
        years_axis,
        np.interp(years_axis, list(obs_datasets["NASA GISS (GISTEMP v4)"].keys()), list(obs_datasets["NASA GISS (GISTEMP v4)"].values())),
        color='black',
        label='Observed Temperature',
        lw=2
    )
    ax_exp.plot(years_exp, res_exp, color='crimson', label='Experimental Simulation', lw=2.5)
    ax_exp.axhline(0, color='gray', lw=1)
    ax_exp.axhline(1.5, color='orange', ls='--', label='1.5 C Threshold')
    ax_exp.set_title(f"Projected Warming under User-Defined Parameters: {res_exp[-1]:.2f} C in 2100", fontsize=16, fontweight='bold')
    ax_exp.set_xlabel("Year")
    ax_exp.set_ylabel("Temperature Anomaly (C)")
    ax_exp.grid(True, alpha=0.3)
    ax_exp.legend(loc='upper left')
    st.pyplot(fig_exp)

    render_section_note(
        "해석",
        "같은 배출 조건에서도 파라미터 선택에 따라 최종 온도 상승폭과 경로가 달라질 수 있습니다. "
        "특히 해양은 열용량이 크기 때문에 육지보다 더 느리게 반응하며, 이러한 반응 속도 차이는 장기 기후 변화 해석에서 중요한 의미를 가집니다."
    )

elif page == "모델 적합도 및 관측자료 비교":
    st.title("모델 적합도 및 관측자료 비교")
    render_section_note(
        "분석 목적",
        "이 섹션은 모델이 실제 관측자료의 장기 기온 변화를 어느 정도 재현할 수 있는지를 평가하기 위한 페이지입니다. "
        "최적화된 파라미터를 이용해 관측값과 모의값의 차이를 정량화하고, 인위적 요인과 자연적 요인의 상대적 기여를 함께 해석할 수 있도록 구성했습니다."
    )

    with st.spinner(f'{obs_choice} 데이터에 맞춰 모델을 최적화하는 중입니다...'):
        best_params = get_optimized_params(current_obs_data)
        best_global, best_tl, best_tm, best_td, daily_all = run_model(best_params, current_obs_data[0])

    err = np.where(
        np.abs(current_obs_data) > 0.1,
        ((best_global - current_obs_data) / np.abs(current_obs_data)) * 100,
        0
    )
    avg_err = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean((best_global - current_obs_data)**2))
    mae = np.mean(np.abs(best_global - current_obs_data))
    bias = np.mean(best_global - current_obs_data)

    st.markdown("<div class='section-title'>핵심 지표</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("RMSE", f"{rmse:.3f} °C", "제곱근 평균 오차")
    with c2:
        render_metric_card("MAE", f"{mae:.3f} °C", "절대값 평균 오차")
    with c3:
        render_metric_card("Bias", f"{bias:.3f} °C", "전반적 과대/과소 예측 경향")
    with c4:
        render_metric_card("Mean % Error", f"{avg_err:.2f}%", "상대 오차의 평균 크기")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    axes[0, 0].plot(years_axis, best_tl, color='sienna', label='Land Surface')
    axes[0, 0].plot(years_axis, best_tm, color='royalblue', label='Ocean Mixed Layer')
    axes[0, 0].plot(years_axis, best_td, color='navy', label='Deep Ocean', lw=2)
    axes[0, 0].set_title("Layer-Specific Thermal Response", fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(years_axis, best_global, 'r', label='Model Simulation', lw=2.5)
    axes[0, 1].plot(years_axis, current_obs_data, 'k--', label='Observed Series', alpha=0.6)
    axes[0, 1].set_title("Observed vs Simulated Temperature Anomaly", fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(years_axis, err, color=['#ff7675' if x > 0 else '#74b9ff' for x in err])
    axes[1, 0].set_title(
        f"Relative Error Structure | Mean % Error: {avg_err:.2f}% | RMSE: {rmse:.3f} C | MAE: {mae:.3f} C | Bias: {bias:.3f} C",
        fontweight='bold'
    )
    axes[1, 0].axhline(0, color='k', lw=0.8)

    f_co2 = [co2_forcing(np.interp(y, [1925, 2025], [306, 427]), best_global[int(y-1925)]) for y in years_axis]
    f_non_co2 = [0.75 * ((y - 1925)/100)**2.2 for y in years_axis]
    f_aero = [aerosol_effect(y, best_params[1]) for y in years_axis]
    axes[1, 1].stackplot(years_axis, f_co2, f_non_co2, labels=['CO2', 'Other Anthropogenic Forcing'], alpha=0.7)
    axes[1, 1].plot(years_axis, f_aero, color='blue', label='Aerosol Cooling', lw=2)
    axes[1, 1].set_title("Anthropogenic Forcing Components", fontweight='bold')
    axes[1, 1].legend(loc='upper left')

    f_volc = [sum([s * np.exp(-(y - ys) / d) for ys, s, d in [(1963.2, -0.8, 1.2), (1982.3, -1.3, 1.5), (1991.4, -1.8, 1.8)] if y >= ys]) for y in years_axis]
    f_osc = [best_params[3] * (np.sin(2 * np.pi * y / 3.8) + np.sin(2 * np.pi * y / 5.5)) for y in years_axis]
    axes[2, 0].fill_between(years_axis, 0, f_volc, color='gray', alpha=0.6, label='Volcanic Forcing')
    axes[2, 0].plot(years_axis, f_osc, color='orange', label='Internal Oscillatory Component')
    axes[2, 0].set_title("Natural Forcing Components", fontweight='bold')
    axes[2, 0].legend()

    anthro = np.array(f_co2) + np.array(f_non_co2) + np.array(f_aero)
    natural = np.array(f_volc) + np.array(f_osc)
    axes[2, 1].plot(years_axis, anthro, color='red', lw=2, label='Anthropogenic Contribution')
    axes[2, 1].plot(years_axis, natural, color='black', label='Natural Contribution')
    axes[2, 1].set_title("Relative Contribution of Anthropogenic and Natural Factors", fontweight='bold')
    axes[2, 1].legend()

    st.pyplot(fig)

    render_section_note(
        "해석",
        "모델은 전체적인 장기 온난화 추세를 비교적 잘 재현하지만, 일부 시기에는 과대예측 또는 과소예측이 나타납니다. "
        "이는 단순화된 강제력 입력과 내부 변동성 표현의 한계에서 비롯될 수 있으며, 장기 추세 설명에는 유용하지만 단기 변동 재현에는 제한이 있음을 보여줍니다."
    )

    st.markdown("<div class='section-title'>일별 상세 보기</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"<div class='subsection-title'>{target_year}년 일별 기온 변화</div>", unsafe_allow_html=True)
    with c2:
        df_export = pd.DataFrame({
            "Year": years_axis,
            "Observed": current_obs_data,
            "Model_Anomaly": best_global,
            "Deep_Ocean_Temp": best_td
        })
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(label="기후 데이터 CSV 다운로드", data=csv, file_name="climate_data.csv", mime="text/csv")

    y_idx = int(target_year - START_YEAR)
    daily_segment = daily_all[y_idx*365 : (y_idx+1)*365]
    fig_d, ax_d = plt.subplots(figsize=(12, 4.5))
    ax_d.plot(range(1, 366), daily_segment, color='darkred', lw=1.5)
    ax_d.axhline(np.mean(daily_segment), color='blue', ls='--', label='Annual Mean')
    ax_d.set_xlabel("Day of Year")
    ax_d.set_ylabel("Temperature Anomaly (C)")
    ax_d.set_title("Daily Temperature Evolution")
    ax_d.legend()
    ax_d.grid(True, alpha=0.2)
    st.pyplot(fig_d)

elif page == "모델 검증 및 불확실성 정량화":
    st.title("모델 검증 및 불확실성 정량화")
    render_section_note(
        "분석 목적",
        "이 페이지는 모델의 잔차 구조, 파라미터 변화에 따른 예측 범위, 그리고 특정 파라미터의 민감도를 함께 확인함으로써 "
        "모델 결과의 안정성과 해석 가능성을 점검하기 위해 구성되었습니다."
    )

    with st.spinner(f'{diag_obs_choice} 자료를 기준으로 검증을 수행하는 중입니다...'):
        diag_best_params = get_optimized_params(diag_obs_data)
        diag_best_global, _, _, _, _ = run_model(diag_best_params, diag_obs_data[0])

    residuals = diag_best_global - diag_obs_data
    rmse_diag = np.sqrt(np.mean((diag_best_global - diag_obs_data)**2))
    mae_diag = np.mean(np.abs(diag_best_global - diag_obs_data))
    bias_diag = np.mean(diag_best_global - diag_obs_data)

    st.markdown("<div class='section-title'>핵심 지표</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("RMSE", f"{rmse_diag:.3f} °C", "예측과 관측의 전체 오차 수준")
    with c2:
        render_metric_card("MAE", f"{mae_diag:.3f} °C", "평균 절대 오차")
    with c3:
        render_metric_card("Bias", f"{bias_diag:.3f} °C", "전반적 편향")

    st.markdown("<div class='section-title'>잔차 진단</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_res_line, ax_res_line = plt.subplots(figsize=(10, 4.5))
        ax_res_line.plot(years_axis, residuals, color='purple', lw=2)
        ax_res_line.axhline(0, color='black', lw=1)
        ax_res_line.set_title("Residual Time Series")
        ax_res_line.set_xlabel("Year")
        ax_res_line.set_ylabel("Residual")
        ax_res_line.grid(True, alpha=0.3)
        st.pyplot(fig_res_line)
    with c2:
        fig_res_hist, ax_res_hist = plt.subplots(figsize=(10, 4.5))
        ax_res_hist.hist(residuals, bins=15, color='slateblue', edgecolor='black', alpha=0.8)
        ax_res_hist.axvline(0, color='black', lw=1)
        ax_res_hist.set_title("Residual Distribution")
        ax_res_hist.set_xlabel("Residual")
        ax_res_hist.set_ylabel("Frequency")
        ax_res_hist.grid(True, alpha=0.2)
        st.pyplot(fig_res_hist)

    st.markdown("<div class='section-title'>불확실성 범위</div>", unsafe_allow_html=True)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(20):
        noisy_params = np.array(diag_best_params) + rng.normal(0, [0.08, 0.08, 0.10, 0.01], size=4)
        noisy_params[0] = np.clip(noisy_params[0], 0.7, 2.3)
        noisy_params[1] = np.clip(noisy_params[1], 0.5, 2.0)
        noisy_params[2] = np.clip(noisy_params[2], 0.5, 3.5)
        noisy_params[3] = np.clip(noisy_params[3], 0.05, 0.25)
        res_tmp, _, _, _, _ = run_model(noisy_params.tolist(), diag_obs_data[0])
        samples.append(res_tmp)

    samples = np.array(samples)
    mean_path = samples.mean(axis=0)
    std_path = samples.std(axis=0)
    lower_path = mean_path - std_path
    upper_path = mean_path + std_path

    fig_unc, ax_unc = plt.subplots(figsize=(12, 5.5))
    ax_unc.plot(years_axis, diag_obs_data, 'k--', label='Observed', alpha=0.7)
    ax_unc.plot(years_axis, mean_path, color='darkred', lw=2, label='Mean Prediction')
    ax_unc.fill_between(years_axis, lower_path, upper_path, color='red', alpha=0.2, label='Uncertainty Band')
    ax_unc.set_title("Prediction Uncertainty Envelope")
    ax_unc.set_xlabel("Year")
    ax_unc.set_ylabel("Temperature Anomaly (C)")
    ax_unc.legend()
    ax_unc.grid(True, alpha=0.3)
    st.pyplot(fig_unc)

    render_section_note(
        "해석",
        "파라미터를 조금씩 변화시켜도 평균 경로는 유지되지만, 후반부로 갈수록 예측 범위가 넓어지는 경향이 나타납니다. "
        "이는 장기 예측일수록 파라미터 선택에 따른 민감도가 커진다는 의미이며, 단일 경로보다 불확실성 범위를 함께 제시하는 것이 더 정직한 해석 방식임을 보여줍니다."
    )

    st.markdown("<div class='section-title'>민감도 분석</div>", unsafe_allow_html=True)
    if sens_param == "기후 피드백 파라미터":
        test_range = np.linspace(0.7, 2.3, 12)
        index_to_change = 0
    elif sens_param == "에어로졸 강도":
        test_range = np.linspace(0.5, 2.0, 12)
        index_to_change = 1
    elif sens_param == "해양 열흡수 계수":
        test_range = np.linspace(0.5, 3.5, 12)
        index_to_change = 2
    else:
        test_range = np.linspace(0.05, 0.25, 12)
        index_to_change = 3

    sens_results = []
    for val in test_range:
        params = diag_best_params.copy()
        params[index_to_change] = val
        res_tmp, _, _, _, _ = run_model(params, diag_obs_data[0], end_year=2100, end_co2=550)
        sens_results.append(res_tmp[-1])

    fig_sens, ax_sens = plt.subplots(figsize=(12, 5))
    ax_sens.plot(test_range, sens_results, marker='o', lw=2)
    ax_sens.set_title("Sensitivity of Projected 2100 Warming")
    ax_sens.set_xlabel("Parameter Value")
    ax_sens.set_ylabel("Projected Temperature in 2100 (C)")
    ax_sens.grid(True, alpha=0.3)
    st.pyplot(fig_sens)

elif page == "기후 모델링 용어 및 개념 정의":
    st.title("기후 모델링 용어 및 개념 정의")
    render_section_note(
        "페이지 안내",
        "본 페이지는 모델에서 사용되는 주요 기후학 개념, 물리 파라미터, 검증 지표를 정리한 참고용 자료입니다. "
        "그래프 해석 전에 필요한 개념을 빠르게 확인할 수 있도록 구성했습니다."
    )

    st.subheader("기후 변화의 원인")
    with st.expander("온실가스 복사 강제력 (Greenhouse Gas Radiative Forcing)", expanded=True):
        st.write("대기 중 온실가스가 지구 복사에너지의 우주 방출을 억제하여 지표를 따뜻하게 만드는 효과입니다. 이산화탄소 농도가 증가할수록 강제력이 커지며, 본 모델에서는 로그 함수 형태로 반영됩니다.")
    with st.expander("에어로졸 효과 (Aerosol Effect)"):
        st.write("대기 중 에어로졸 입자가 태양복사를 반사하거나 구름의 반사도를 높여 지표를 냉각시키는 효과입니다. 온난화를 일부 상쇄하지만 시기와 지역에 따라 영향이 달라집니다.")
    with st.expander("화산 강제력 (Volcanic Forcing)"):
        st.write("대규모 화산 분출 이후 성층권에 주입된 입자가 태양복사를 차단하여 단기 냉각을 유도하는 효과입니다. 본 모델은 시간에 따라 약해지는 지수 감쇠 형태로 이를 표현합니다.")
    with st.expander("엘니뇨-남방진동 (ENSO, El Nino-Southern Oscillation)"):
        st.write("열대 태평양의 해수면 온도와 대기 순환이 수년 주기로 변동하는 자연 내부 변동성입니다. 장기 온난화 추세와 별개로 특정 연도의 기온을 높이거나 낮출 수 있습니다.")

    st.subheader("기후 시스템의 물리적 반응")
    with st.expander("기온 편차 (Temperature Anomaly)"):
        st.write("절대기온이 아니라 기준 기간 평균으로부터 얼마나 벗어났는지를 나타내는 값입니다. 서로 다른 시기와 자료를 비교할 때 널리 사용됩니다.")
    with st.expander("열용량 (Heat Capacity)"):
        st.write("물질의 온도를 1도 높이는 데 필요한 에너지 양입니다. 바다는 열용량이 커서 육지보다 천천히 가열되고 천천히 냉각됩니다.")
    with st.expander("해양 층위 분리 (Ocean Layering)"):
        st.write("모델에서 해양을 혼합층과 심해로 구분하여 열 저장과 전달 과정을 표현하는 방식입니다. 이는 해양의 열관성과 장기 온난화 반응을 설명하는 데 중요합니다.")
    with st.expander("열 교환 계수 (Heat Exchange Rate)"):
        st.write("육지와 바다, 혹은 해양 표층과 심해 사이에 열이 얼마나 빠르게 이동하는지를 나타내는 계수입니다.")
    with st.expander("기후 피드백 파라미터 (Climate Feedback Parameter)"):
        st.write("기후 시스템이 따뜻해질수록 얼마나 강하게 복사 냉각으로 되돌리려 하는지를 나타내는 계수입니다. 값이 클수록 온난화 억제 효과가 큽니다.")

    st.subheader("모델 평가와 검증 지표")
    with st.expander("기계학습 최적화 알고리즘 (L-BFGS-B)"):
        st.write("관측값과 모델값의 차이가 최소가 되도록 파라미터를 자동 조정하는 수치 최적화 알고리즘입니다.")
    with st.expander("오차율 (Error Percentage)"):
        st.write("모델 예측이 관측값에 비해 상대적으로 얼마나 높거나 낮은지를 백분율로 나타낸 지표입니다. 관측값이 0에 가까운 구간에서는 왜곡이 커질 수 있습니다.")
    with st.expander("평균제곱근오차 (RMSE, Root Mean Squared Error)"):
        st.write("예측값과 관측값 차이의 제곱 평균에 제곱근을 취한 지표입니다. 큰 오차에 민감하므로 모델이 특정 시점에서 크게 빗나가는지를 평가하는 데 유용합니다.")
    with st.expander("평균절대오차 (MAE, Mean Absolute Error)"):
        st.write("예측값과 관측값 차이의 절댓값 평균입니다. 평균적으로 몇 도 정도 오차가 나는지를 직관적으로 해석하기 좋습니다.")
    with st.expander("편향 (Bias)"):
        st.write("예측값에서 관측값을 뺀 차이의 평균입니다. 양수이면 전반적으로 높게, 음수이면 낮게 예측하는 경향을 의미합니다.")
    with st.expander("잔차 (Residual)"):
        st.write("각 시점에서 모델 예측값과 실제 관측값의 차이입니다. 잔차 패턴을 보면 특정 시기에 구조적인 과대예측 또는 과소예측이 존재하는지 확인할 수 있습니다.")
    with st.expander("불확실성 범위 (Uncertainty Band)"):
        st.write("모델 파라미터를 조금씩 변화시켜 여러 번 계산했을 때 나타나는 예측 범위입니다. 단일 예측선보다 모델의 불확실성을 더 정직하게 보여줍니다.")
    with st.expander("민감도 분석 (Sensitivity Analysis)"):
        st.write("하나의 파라미터를 바꾸었을 때 결과가 얼마나 크게 달라지는지 평가하는 분석입니다. 어떤 변수가 모델 출력에 가장 큰 영향을 주는지 파악할 수 있습니다.")
    with st.expander("인위적 요인과 자연적 요인의 상대 기여 (Forcing Dominance)"):
        st.write("온실가스, 에어로졸, 화산, 내부 변동 등 여러 강제력 요소를 비교하여 온도 변화에 어떤 요인이 더 크게 작용하는지 해석하는 개념입니다.")

    st.info("그래프 내부 제목과 축 라벨은 글꼴 호환성을 위해 영어로 유지했고, UI와 설명문은 한국어로 정리했습니다.")

elif page == "연구 요약 및 보고서":
    st.title("연구 요약 및 보고서")
    render_section_note(
        "연구 목적",
        "본 프로젝트는 관측 자료와 단순화된 물리 기반 기후 모델을 결합하여 역사적 기온 변화를 재현하고, "
        "배출 시나리오와 주요 물리 파라미터 변화에 따라 미래 온난화 경로가 어떻게 달라지는지를 해석하는 것을 목표로 합니다."
    )

    render_section_note(
        "모델 구조와 물리적 가정",
        "모델은 육지, 해양 혼합층, 심해의 세 층으로 구성된 간이 에너지 균형 구조를 사용합니다. "
        "이산화탄소 복사 강제력, 에어로졸 냉각, 비이산화탄소 인위적 강제력, 화산 강제력, 내부 변동성을 포함하며, "
        "전지구 평균 규모에서 장기 추세와 주요 기여 요인을 해석하는 데 초점을 둡니다."
    )

    st.markdown("<div class='section-title'>연구 요약</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="paper-card">
            <div class="paper-index">Summary</div>
            <div class="paper-title">핵심 분석 구성</div>
            <div class="paper-desc">
                시나리오 기반 장기 온난화 예측, 기후 피드백 및 해양 열흡수 파라미터 실험,
                관측자료와의 적합도 비교, 잔차와 불확실성 범위 검토, 민감도 분석을 하나의 흐름으로 통합했습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="paper-card">
            <div class="paper-index">Interpretation</div>
            <div class="paper-title">해석상의 주의점</div>
            <div class="paper-desc">
                본 모델은 정밀 예측 모델이 아니라 해석 중심의 교육·연구용 모델입니다.
                지역별 차이보다 전지구 평균 경향을 다루며, 일부 강제력과 내부 변동성은 단순화된 함수로 표현됩니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>연구 의의</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="abstract-box">
        <div class="abstract-text">
            이 대시보드는 단순한 시각화 도구를 넘어, 기후 시스템의 핵심 강제력과 반응 변수를 하나의 흐름 안에서 탐색할 수 있도록 구성되어 있습니다.
            특히 관측자료 비교, 파라미터 실험, 불확실성 정량화를 하나의 인터페이스에 통합함으로써 학부 수준에서 기후 모델링의 기본 구조와 검증 과정을 체계적으로 학습할 수 있도록 설계되었습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>보고서 다운로드</div>", unsafe_allow_html=True)
    report_name, report_bytes = load_report_file()
    if report_bytes is not None:
        st.download_button(
            label="분석 리포트 다운로드",
            data=report_bytes,
            file_name=report_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    else:
        st.info("리포트 파일을 찾지 못했습니다. app.py와 같은 위치에 .docx 파일이 있는지 확인하세요.")
