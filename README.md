# Climate Modeling Research Dashboard
기후 모델링 연구 대시보드

Live Demo  
(https://climatemodeling-28m9wh2qjv2p7qcajwxm8s.streamlit.app/?module=home)

---

## 1. Overview | 개요

**EN**  
This project is an academic-oriented interactive climate modeling dashboard that integrates a simplified physical energy-balance model with observational datasets. It enables exploration of long-term climate dynamics, emission scenarios, and prediction uncertainty.

**KR**  
본 프로젝트는 단순화된 물리 기반 에너지 균형 모델과 관측 데이터를 결합하여 기후 변화의 장기 경향, 배출 시나리오, 그리고 예측 불확실성을 분석할 수 있도록 설계된 연구형 대시보드이다.

---

## 2. Objectives | 연구 목적

**EN**
- Build an interactive platform for climate system exploration  
- Analyze long-term temperature response under emission scenarios  
- Evaluate sensitivity of climate models to key parameters  
- Validate model outputs using observational datasets  

**KR**
- 기후 시스템 분석을 위한 인터랙티브 플랫폼 구축  
- 배출 시나리오에 따른 장기 기온 변화 분석  
- 주요 파라미터에 대한 모델 민감도 평가  
- 관측 데이터 기반 모델 검증 수행  

---

## 3. Model Description | 모델 설명

This project is based on a simplified three-layer Energy Balance Model (EBM).

### Structure
- Land layer  
- Ocean mixed layer  
- Deep ocean  

### Physical Processes
- CO₂ radiative forcing (logarithmic relationship)  
- Aerosol cooling effect  
- Volcanic forcing  
- Internal variability (ENSO-like oscillation)  

Model parameters are optimized using observational temperature datasets.

---

## 4. Features | 주요 기능

### 4.1 Scenario-Based Climate Projection
- CO₂ emission scenarios (Net Zero to high emission)  
- Temperature projection up to 2100  
- Threshold comparison (e.g., 1.5°C, 2.0°C)

### 4.2 Climate System Parameter Experiment
- Adjustable parameters:
  - Climate feedback (λ)  
  - Aerosol forcing  
  - CO₂ concentration  
- Real-time simulation update  

### 4.3 Model Fit and Observation Comparison
- Comparison with NASA GISTEMP data  
- RMSE, MAE, Bias, sMAPE evaluation  
- Decomposition of forcing contributions  

### 4.4 Model Validation and Uncertainty
- Residual diagnostics  
- Monte Carlo-based uncertainty estimation  
- Parameter sensitivity analysis  

### 4.5 Glossary
- Explanation of key climate modeling concepts  
- Radiative forcing, feedback, ENSO, etc.

---

## 5. Data Sources | 데이터 출처

- NASA GISS (GISTEMP v4)  
- NOAA CO₂ Data  
- IPCC AR6  
- Smithsonian Volcano Database  

---

## 6. Technical Stack | 사용 기술

- Python  
- Streamlit  
- NumPy  
- Pandas  
- Matplotlib  
- SciPy  
- Numba  

---

## 7. Project Structure | 프로젝트 구조

climate_modeling/
├── app.py # Streamlit UI
├── climate_core.py # climate model and numerical computation
├── data_loader.py # observational data loading
├── style.css # UI styling
├── requirements.txt

---

## 8. Installation | 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py

또는 Live Demo 링크를 통해 바로 실행 가능하다.

---

## 9. Results and Usage | 활용

This dashboard enables:

Exploration of climate response under emission scenarios
Analysis of model sensitivity to parameter changes
Quantitative comparison with observational data
Understanding uncertainty in long-term climate projections

---

## 10. Limitations | 한계
Global mean 기반 단순 모델
지역별 기후 변화 반영 불가
일부 forcing은 경험적 함수 기반
단기 변동성 재현에는 한계 존재

---

## 11. Future Work | 향후 연구 방향
고해상도 기후 모델 적용
실제 forcing 데이터 연동
머신러닝 기반 보정 모델 도입
웹 애플리케이션 성능 개선

---

## 12. Author

Name: 유환빈
GitHub: https://github.com/rol3and77

---

## 13. License

MIT License
