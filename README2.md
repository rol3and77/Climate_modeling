# Climate Modeling Research Dashboard
기후 모델링 연구 대시보드

---

## 1. Overview | 개요

**EN**  
This project presents an academic-oriented interactive dashboard for climate modeling, integrating simplified physical models with observational data to explore climate dynamics, scenario projections, and uncertainty analysis.

**KR**  
본 프로젝트는 물리 기반 기후 모델과 관측 데이터를 결합하여 기후 시스템의 동역학, 시나리오 예측, 그리고 불확실성 분석을 수행할 수 있도록 설계된 인터랙티브 대시보드이다.

---

## 2. Objectives | 연구 목적

**EN**
- To develop an interactive platform for climate system exploration
- To analyze the impact of emission scenarios on global temperature
- To investigate sensitivity of climate models to key parameters
- To evaluate model performance against observational datasets

**KR**
- 기후 시스템 분석을 위한 인터랙티브 플랫폼 구축
- 배출 시나리오에 따른 기온 변화 분석
- 주요 파라미터에 대한 모델 민감도 평가
- 관측 데이터 기반 모델 검증 수행

---

## 3. Methodology | 연구 방법

This project is based on simplified climate modeling approaches, including energy balance concepts and parameterized forcing mechanisms.

- Scenario-based forcing inputs
- Parameter-driven model adjustment
- Residual and error analysis
- Visualization of temporal climate evolution

---

## 4. Features | 주요 기능

### 4.1 Scenario-Based Climate Projection
- Simulation of temperature response under multiple emission pathways
- Comparative analysis of low, medium, and high emission scenarios
- Long-term projection up to the year 2100

### 4.2 Climate System Parameter Experiment
- Adjustment of key parameters such as:
  - Climate feedback factor
  - Aerosol forcing
  - Ocean heat uptake
  - ENSO variability
- Analysis of system sensitivity to parameter variation

### 4.3 Model Fit and Observation Comparison
- Comparison between modeled outputs and observational datasets
- Residual analysis and visualization
- Evaluation of forcing contributions

### 4.4 Model Validation and Uncertainty Quantification
- Residual diagnostics
- Sensitivity analysis
- Visualization of prediction uncertainty

### 4.5 Climate Modeling Glossary
- Explanation of key concepts including:
  - Radiative forcing
  - Climate sensitivity
  - Feedback mechanisms
  - Energy balance models

---

## 5. Technical Stack | 사용 기술

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- SciPy
- Numba
- Google Colab
- Cloudflared

---

## 6. Project Structure | 프로젝트 구조

climat_modeling/
├── README.md
├── README2.md
├── app.py
├── requirements.txt

---

## 7. Installation and Execution | 실행 방법

It can be run on Google Colab, please refer to the README.md.

After running the final step, a public URL will be generated to access the dashboard.

---

## 8. Results and Usage | 활용

This dashboard enables users to:

- Explore climate response under different emission scenarios
- Analyze sensitivity of climate systems to parameter changes
- Compare model outputs with observational data
- Understand uncertainty in climate predictions

---

## 9. Future Work | 향후 연구 방향

- Integration of real-world datasets (e.g., NASA, NOAA)
- Implementation of advanced climate models
- Incorporation of machine learning techniques
- Deployment on cloud-based platforms

---

## 10. Author

Name: 유환빈  
GitHub: https://github.com/rol3and77

---

## 11. License

This project is licensed under the MIT License.
