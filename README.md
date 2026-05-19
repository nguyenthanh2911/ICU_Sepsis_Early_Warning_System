# 🏥 ICU Sepsis Early Warning System

[![CI](https://github.com/nguyenthanh2911/CNM/actions/workflows/ci.yml/badge.svg)](https://github.com/nguyenthanh2911/CNM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Django](https://img.shields.io/badge/Django-5.0-092E20?logo=django)](https://www.djangoproject.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-blue?logo=mlflow)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql)](https://www.postgresql.org/)
[![Prefect](https://img.shields.io/badge/Prefect-2.19-7B46E6)](https://www.prefect.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F2F4FF?logo=grafana)](https://grafana.com/)

> **Hệ thống MLOps cảnh báo sớm Sepsis trong ICU** — Dự đoán nguy cơ nhiễm khuẩn huyết trước **6 giờ** bằng XGBoost, phục vụ real-time qua FastAPI, hiển thị dashboard WebSocket Django, giám sát drift và tự động retrain.

---

## Mục lục

1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Công nghệ sử dụng](#4-công-nghệ-sử-dụng)
5. [Dữ liệu](#5-dữ-liệu)
6. [Các dịch vụ (Services)](#6-các-dịch-vụ-services)
7. [Hướng dẫn cài đặt](#7-hướng-dẫn-cài-đặt)
8. [Hướng dẫn sử dụng](#8-hướng-dẫn-sử-dụng)
9. [Quy trình ML Pipeline](#9-quy-trình-ml-pipeline)
10. [API Reference](#10-api-reference)
11. [Giám sát & Retrain](#11-giám-sát--retrain)
12. [Testing](#12-testing)
13. [Kết quả và đánh giá](#13-kết-quả-và-đánh-giá)

---

## 1. Giới thiệu đề tài

### Bài toán

**Sepsis** (nhiễm khuẩn huyết) là phản ứng đe dọa tính mạng của cơ thể khi nhiễm trùng, gây ra hơn **270.000 ca tử vong mỗi năm** tại Mỹ. Phát hiện sớm trong **1–6 giờ đầu** giúp tăng tỉ lệ sống sót lên 80%, nhưng y tá ICU phải theo dõi hàng chục chỉ số sinh tồn (heart rate, SpO₂, huyết áp, nhiệt độ…) liên tục cho nhiều bệnh nhân cùng lúc — dẫn đến nguy cơ bỏ sót dấu hiệu chuyển nặng.

### Giải pháp

Xây dựng hệ thống **MLOps hoàn chỉnh** theo 3 tầng:

- **🧪 Data & Training**: Sinh dữ liệu synthetic ICU mô phỏng sinh lý bệnh nhân (có confounders: tuổi, nhiễu thiết bị, overlap sepsis/non-sepsis), xây dựng features lâm sàng (SOFA, NEWS2, qSOFA + rolling statistics), train mô hình **XGBoost** dự đoán nguy cơ T+6h, theo dõi thí nghiệm qua **MLflow**
- **🚀 Serving & Deployment**: API dự đoán real-time qua **FastAPI** (kết hợp XGBoost + SHAP + rule-based early warning), dashboard quan sát bằng **Django + Channels WebSocket**, cảnh báo qua **Alert Service**, đóng gói bằng **Docker Compose**, CI/CD qua **GitHub Actions**
- **📊 Monitoring & Retraining**: Phát hiện data drift bằng **Evidently AI**, tự động retrain bằng **Prefect**, theo dõi hệ thống bằng **Prometheus + Grafana**

### Mục tiêu kỹ thuật

| Chỉ số | Mục tiêu | Ghi chú |
|--------|----------|---------|
| **AUROC (Test)** | > 0.75 (Production) / > 0.70 (Staging) | Ngưỡng register model trong `train.py` |
| **Sensitivity** | > 75% | Threshold phân lớp mặc định = 0.4 |
| **Specificity** | > 85% | Tính từ confusion matrix trong `evaluate.py` |
| **F1 Score** | > 0.65 | Theo dõi qua MLflow metrics |
| **Inference latency** | < 200ms / single row | Prometheus `inference_seconds` histogram |
| **Cảnh báo (end-to-end)** | < 5 phút | Từ lúc POST /vitals đến WebSocket push |
| **ML alert lead time** | > 6 giờ trước sepsis | Label T+6h (`sepsis_in_next_6h`) |
| **Rule-based lead time** | > 30 phút trước sepsis | `EarlyWarningPredictor` trend + rate + threshold |
| **Data drift threshold** | Drift score > 0.7 | Kích hoạt Prefect retrain flow |
| **Model promotion gap** | New AUROC > Production + 0.01 | Ngưỡng promote trong `retrain_flow.py` |
| **CV variance** | std_auroc < 0.05 | Nếu > 0.08 → tự động tighten regularization |
| **Label imbalance ratio** | ~9:1 (neg:pos) | SMOTE với sampling_strategy=0.4 |

---

## 2. Kiến trúc hệ thống

```
                         ┌──────────────────────────────────────────┐
                         │         DATA & TRAINING LAYER            │
                         │                                          │
                         │  data_generator.py                       │
                         │   ├── PhysiologicalModel (vitals synth)  │
                         │   │   ├── age confounder, equipment noise│
                         │   │   └── overlap: bad_spikes / recovery │
                         │   └── LabResultModel (labs synth)        │
                         │       └── missing labs ~5%, early normal │
                         │          │                               │
                         │          ▼                               │
                         │  labeling.py  (create_t6h_labels)        │
                         │   └── sepsis_in_next_6h = y[t]           │
                         │          │                               │
                         │          ▼                               │
                         │  feature_builder.py                      │
                         │   ├── Rolling stats (15/60/240 phút)     │
                         │   ├── Clinical scores (SOFA,NEWS2,qSOFA) │
                         │   └── Time-since-last-abnormal-HR        │
                         │          │                               │
                         │          ▼                               │
                         │  train.py                                │
                         │   ├── Patient split (~60/20/20)          │
                         │   ├── SimpleImputer + StandardScaler     │
                         │   ├── Auto SMOTE (ratio=0.4)             │
                         │   ├── 5-fold StratifiedKFold CV          │
                         │   └── XGBoost (150 est, max_depth=4)     │
                         │          │                               │
                         │          ▼                               │
                         │  MLflow Tracking + Registry              │
                         │   ├── Log: params, metrics, model        │
                         │   └── Stage: Staging → Production        │
                         └──────────────────────────────────────────┘
                                      │
                              ┌───────┴────────┐
                              │   PostgreSQL   │
                              │  (predictions, │
                              │   alerts,      │
                              │   patients,    │
                              │   vital_records│
                              │   admissions)  │
                              └───────┬────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  SERVING LAYER      │   │  ALERTING LAYER     │   │  WEB LAYER          │
│                     │   │                     │   │                     │
│  FastAPI ML Service │   │  FastAPI Alert      │   │  Django Dashboard   │
│  (port 8001)        │   │  Service (port 8002)│   │  (port 8000)        │
│                     │   │                     │   │                     │
│  POST /vitals       │   │  POST /alerts       │   │  / → patient_list   │
│   ├─ SOFA, NEWS2    │──►│  GET /alerts        │   │  /patients/{id}/    │
│   ├─ Preprocess     │   │  GET /alerts/stats  │   │  /alerts/           │
│   ├─ XGBoost → risk │   │  WebSocket push     │   │  WebSocket Daphne   │
│   ├─ SHAP explain   │   │  PATCH acknowledge  │   │  API endpoints      │
│   ├─ EarlyWarning   │   │  GET /metrics       │   │                     │
│   ├─ Lưu PostgreSQL │   │  (active_alerts)    │   │                     │
│   └─ GET /metrics   │   └─────────────────────┘   └─────────────────────┘
│     (predictions_*) │             │                        │
└─────────────────────┘             └──────────┬─────────────┘
                                               │
                                               ▼
                                      ┌─────────────────────┐
                                      │  WebSocket Client   │
                                      │  (Browser)          │
                                      └─────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & RETRAINING LAYER                       │
│                                                                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │  Prometheus  │◄───│  scrape metrics  │◄───│  ML + Alert svc │       │
│  │  (port 9090) │    │  (15s interval)  │    │  /metrics        │       │
│  └──────┬───────┘    └──────────────────┘    └──────────────────┘       │
│         │                                                                 │
│         ▼                                                                 │
│  ┌──────────────┐    ┌──────────────────┐                                │
│  │   Grafana    │◄───│  icu_dashboard   │                                │
│  │  (port 3000) │    │  (JSON config)   │                                │
│  └──────────────┘    └──────────────────┘                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Evidently AI (DataDriftPreset) ← reference vs current 24h  │        │
│  └──────────────────────┬──────────────────────────────────────┘        │
│                         │ drift_score > 0.7                             │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Prefect retrain_flow                                       │        │
│  │   ├── check_drift()                                         │        │
│  │   ├── run_training() → train.py subprocess                  │        │
│  │   └── compare_and_promote() → New > Production + 0.01 AUROC │        │
│  └─────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Luồng dữ liệu chính

```
╔══════════════════════════════════════════════════════════════════╗
║                    ICU SIMULATION (scripts/simulate_realtime.py) ║
║  20 patients: 40% sepsis, 60% non-sepsis                        ║
║  3 risk groups: LOW (~10), WARNING (~4), HIGH (~6)              ║
║  Mỗi 10s thực = 1h mô phỏng — 240 steps total                   ║
║  Đọc từ data/raw/real_time_icu.csv, map P0001→P001 (seed IDs)   ║
╚══════════════════════════════════════════════════════════════════╝
                           │
                           │ POST /vitals (concurrent.futures, max_workers=20)
                           ▼
╔══════════════════════════════════════════════════════════════════╗
║              FastAPI ML Service — localhost:8001                 ║
║                                                                  ║
║  1️⃣ Validate input (Pydantic VitalRequest)                       ║
║     - heart_rate∈[20,250], spo2∈[50,100], temp∈[30,45], ...     ║
║                                                                  ║
║  2️⃣ Clinical scores (từ raw vitals, không cần preprocess)       ║
║     - calculate_sofa() → respiratory + coagulation + liver       ║
║                          + renal + cardiovascular (0–20)         ║
║     - calculate_news2() → RR + SpO2 + temp + BP + HR (0–20)     ║
║                                                                  ║
║  3️⃣ Preprocess (sklearn Pipeline)                                ║
║     - SimpleImputer(strategy='median') → điền giá trị thiếu      ║
║     - StandardScaler → chuẩn hoá 11 features                     ║
║     - Fallback: fit-on-fly nếu pipeline chưa được fit            ║
║                                                                  ║
║  4️⃣ XGBoost predict_proba → risk_score ∈ [0, 1]                 ║
║     - risk < 0.3  → "LOW"                                       ║
║     - 0.3 – 0.7  → "WARNING"                                    ║
║     - ≥ 0.7      → "CRITICAL" + alert_triggered = true          ║
║                                                                  ║
║  5️⃣ SHAP TreeExplainer → top-5 features (feature + shap_value)  ║
║                                                                  ║
║  6️⃣ EarlyWarningPredictor (rule-based + ML T+6h)                ║
║     - trend_score (30 phút history)              [weight 30%]   ║
║     - rate_of_change_score (đạo hàm bậc 1)      [weight 20%]   ║
║     - threshold_score (mức gần ngưỡng nguy hiểm) [weight 50%]   ║
║     - Kết hợp với risk_score_t6h từ XGBoost                      ║
║     - → early_warning_probability (0..1) + level (LOW/MED/HIGH) ║
║     - → contributing_factors (danh sách yếu tố nguy cơ cụ thể)  ║
║                                                                  ║
║  7️⃣ Lưu vào PostgreSQL (predictions table):                     ║
║     risk_score, risk_level, alert_triggered,                     ║
║     sofa_score, news2_score, inference_time_ms,                  ║
║     raw vitals (HR, BP, SpO2, temp, RR, lactate, wbc, ...),    ║
║     early_warning scores (prob, level, trend, roc, threshold)   ║
║                                                                  ║
║  8️⃣ Prometheus metrics:                                         ║
║     predictions_total (Counter)                                  ║
║     predictions_by_risk_total{risk_level} (Counter)              ║
║     inference_seconds (Histogram)                                ║
╚══════════════════════════════════════════════════════════════════╝
              │
              │
              ├── risk < 0.3 ────────────────────────────────────── log only
              │
              ├── 0.3 ≤ risk < 0.7 ──────────────────────────────── Dashboard WARNING
              │                                                      (Django hiển thị badge vàng)
              │
              ├── risk ≥ 0.7 ──────────────────────────────────────►
              │                                                      │
              │                                                      ▼
              │                                          ╔══════════════════════════════╗
              │                                          ║  Alert Service — :8002       ║
              │                                          ║  POST /alerts                ║
              │                                          ║  - Nếu đã có pending alert    ║
              │                                          ║    cho cùng patient → update  ║
              │                                          ║  - Nếu chưa → tạo mới         ║
              │                                          ║  - Lưu vào PostgreSQL (alert) ║
              │                                          ║  - Push WebSocket real-time   ║
              │                                          ║  - Prometheus: active_alerts  ║
              │                                          ╚══════════════════════════════╝
              │                                                      │
              └── early_warning HIGH (nhưng risk < 0.7) ────────────►│
                                                                     │
                                                                     ▼
                                                    ╔════════════════════════════════╗
                                                    ┌────────────────────────────────┐
                                                    │  Django Dashboard — :8000     │
                                                    │  (Daphne ASGI + WebSocket)    │
                                                    │                               │
                                                    │  Danh sách bệnh nhân (/)      │
                                                    │  ├── Stats: total/critical/   │
                                                    │  │   warning/stable            │
                                                    │  ├── Bảng: ID, name, room,    │
                                                    │  │   risk_score bar, level     │
                                                    │  ├── Sort: unconfirmed trước   │
                                                    │  └── WS real-time update       │
                                                    │                               │
                                                    │  Chi tiết (/patients/{id}/)   │
                                                    │  ├── Risk score card (số lớn) │
                                                    │  ├── Early warning bars        │
                                                    │  ├── Vitals grid (5 cards)     │
                                                    │  ├── Risk chart (2h gần nhất) │
                                                    │  ├── SHAP bar chart (top-5)   │
                                                    │  ├── SOFA + NEWS2 badges       │
                                                    │  └── Acknowledge alert button  │
                                                    │                               │
                                                    │  Alerts (/alerts/)             │
                                                    │  ├── Tabs: all/pending/confirm │
                                                    │  ├── Table: ID, patient, time  │
                                                    │  ├── Severity + status badges  │
                                                    │  ├── Confirm button + toast    │
                                                    │  └── WebSocket push real-time  │
                                                    └────────────────────────────────┘
```

---

## 3. Cấu trúc thư mục

```
CNM/
├── 📄 docker-compose.yml          # Docker Compose: 6 services (postgres, mlflow, prometheus,
│                                  #   grafana, ml_service, alert_service, web) + bridge network
├── 📄 pytest.ini                  # Pytest config: testpaths=tests, verbose, short traceback
├── 📄 README.md                   # Tài liệu dự án
├── 📄 requirements.txt            # 28 dependencies: fastapi, django, xgboost, mlflow, evidently, prefect...
├── 📄 test_vitals.json            # Mẫu vitals JSON dùng cho test (P0001, 1 record)
│
├── 📁 artifacts/                  # Artifacts sinh ra từ quá trình training
│   └── 📦 preprocessor_t6h.joblib # sklearn Pipeline (SimpleImputer + StandardScaler) đã fit
│
├── 📁 data/                       # Dữ liệu
│   ├── 📁 processed/              # Dữ liệu đã tiền xử lý (features_train.parquet cho drift detection)
│   ├── 📁 raw/                    # Dữ liệu gốc (real_time_icu.csv cho simulate_realtime)
│   └── 📁 synthetic/              
│       └── 📄 icu_data_synthetic.csv  # Dữ liệu synthetic (data_generator.py output)
│
├── 📁 data_pipeline/              # Pipeline xử lý dữ liệu đầu vào
│   ├── 📄 data_generator.py       # ICUSepsisGenerator + PhysiologicalModel (vitals) + LabResultModel
│   │                             #   confounders: age multiplier, equipment noise, bad spikes, recovery
│   ├── 📄 labeling.py             # create_t6h_labels() → sepsis_in_next_6h, split_by_patient() (no leakage)
│   └── 📄 preprocessor.py         # ICUPreprocessor: forward-fill (10 phút), KNNImputer (labs, k=3),
│                                  #   IQR outlier → median, StandardScaler, save/load joblib
│
├── 📁 docs/                       # Tài liệu bổ sung
│   └── 📄 database_schema.sql     # PostgreSQL schema: patients, admissions, vital_records,
│                                  #   predictions (15 cột vital + early_warning), alerts, prediction_results
│
├── 📁 feature_engineering/        # Trích xuất & biến đổi đặc trưng
│   ├── 📄 clinical_scores.py      # calculate_sofa() (5 thành phần, 0–20), calculate_news2() (5 thành phần, 0–20),
│   │                             #   calculate_qsofa() (2 thành phần, 0–2)
│   ├── 📄 feature_builder.py      # FeatureBuilder: rolling stats + clinical scores + time-since-abnormal-HR,
│   │                             #   drop raw vitals/labs, giữ patient_id/time/label + features
│   └── 📄 vitals_features.py      # add_rolling_features(): mean/std/min/max (windows 3/12/48 intervals
│                                  #   ~ 15/60/240 phút) + trend diff(1)/interval cho 6 vitals
│
├── 📁 ml/                         # Machine Learning
│   ├── 📄 early_warning.py        # EarlyWarningPredictor: rule engine 30 phút
│   │                             #   trend_score(30%) + rate_of_change_score(20%) + threshold_score(50%)
│   │                             #   + contributing_factors (lactate cao, SpO2 thấp, HR tăng...)
│   ├── 📄 evaluate.py             # evaluate_model(): AUROC, F1 (thr=0.4), Sensitivity, Specificity,
│   │                             #   confusion matrix, ROC curve plot (PNG)
│   ├── 📄 explain.py              # SepsisExplainer: SHAP TreeExplainer → top-5 features theo |shap_value|
│   ├── 📄 mlflow_utils.py         # log_training_run() (params + metrics + xgboost model + feature_names.json),
│   │                             #   register_model(), load_production_model_with_metadata()
│   ├── 📄 train.py                # Training pipeline: load CSV → create_t6h_labels → patient-based split
│   │                             #   (~60/20/20) → SimpleImputer+Scaler → Auto SMOTE (ratio=0.4) →
│   │                             #   5-fold StratifiedKFold CV → XGBoost fit → evaluate → MLflow log →
│   │                             #   register nếu test_auroc > 0.75 (Production) / > 0.70 (Staging)
│   └── 📁 models/
│       └── 📄 xgboost_model.py    # SepsisXGBModel: 150 est, max_depth=4, lr=0.05, subsample=0.65,
│                                  #   scale_pos_weight, early_stopping 30, CV diagnostics, save/load joblib
│
├── 📁 monitoring/                 # Giám sát & tự động retrain
│   ├── 📄 drift_detector.py       # DriftDetector: Evidently AI DataDriftPreset, so sánh reference
│   │                             #   (parquet) vs current 24h (PostgreSQL), drift_score > 0.7 → drift
│   ├── 📄 retrain_flow.py         # Prefect flow: check_drift() → run_training() (subprocess train.py)
│   │                             #   → compare_and_promote() (new_auroc > production + 0.01)
│   ├── 📁 grafana/dashboards/
│   │   └── 📄 icu_dashboard.json  # Grafana dashboard JSON (pre-configured)
│   └── 📁 prometheus/
│       └── 📄 prometheus.yml      # Scrape config: ml_service:8001 + alert_service:8002, interval 15s
│
├── 📁 scripts/                    # Scripts hỗ trợ vòng đời hệ thống
│   ├── 📄 check_health.sh         # Health check 4 services (ML 8001, Alert 8002, Django 8000, MLflow 5000)
│   ├── 📄 ci_seed_data.py         # CI: tạo reference parquet + seed vital_records cho drift detection
│   ├── 📄 run_demo.sh             # 1-lệnh demo: docker up → setup DB → seed patients → generate data →
│   │                             #   train model → simulate realtime
│   ├── 📄 seed_patients.py        # Seed 20 bệnh nhân (tên VN) vào PostgreSQL (P001..P020)
│   ├── 📄 setup_db.ps1            # PowerShell: init DB schema (Windows)
│   ├── 📄 setup_db.sh             # Bash: init DB schema (Linux/macOS) — đợi Postgres ready, chạy SQL
│   └── 📄 simulate_realtime.py    # Mô phỏng real-time: đọc CSV, ThreadPool 20 workers,
│                                  #   POST /vitals mỗi 10s, map P0001→P001 (seed IDs)
│
├── 📁 services/                   # Microservices (FastAPI)
│   ├── 📁 alert_service/          # 🚨 Alert Service — port 8002
│   │   ├── 📄 Dockerfile          # python:3.10-slim, uvicorn services.alert_service.main:app
│   │   ├── 📄 main.py             # FastAPI app: POST /alerts (upsert pending), GET /alerts (filter),
│   │   │                         #   WebSocket push, PATCH acknowledge, Prometheus active_alerts Gauge
│   │   ├── 📄 schemas.py          # AlertCreate (patient_id, risk, top_features, alert_type),
│   │   │                         #   AlertResponse, AlertStats
│   │   └── 📄 websocket_manager.py# ConnectionManager: key=patient_id/"all", send_alert(), broadcast()
│   └── 📁 ml_service/             # 🤖 ML Service — port 8001
│       ├── 📄 Dockerfile          # python:3.10-slim, uvicorn services.ml_service.main:app
│       ├── 📄 main.py             # FastAPI app: POST /vitals (predict + SHAP + early warning + DB + alert call),
│       │                         #   GET /health, GET /vitals/{id}/history, Prometheus /metrics
│       ├── 📄 predictor.py        # SepsisPredictor singleton: load MLflow model, preprocess Pipeline,
│       │                         #   predict_proba(), SHAP explain, EarlyWarningPredictor, cache vitals
│       └── 📄 schemas.py          # VitalRequest (Pydantic v2, 6 vitals + 5 labs optional),
│                                  #   PredictionResponse (risk + early_warning + shap),
│                                  #   EarlyWarningResult, HealthResponse
│
├── 📁 tests/                      # Pytest — 5 test files, ~70+ tests
│   ├── 📁 integration/
│   │   └── 📄 test_pipeline.py    # Integration: generator → FeatureBuilder (columns, scores, rolling),
│   │                             #   generator → labeling → split (no leakage), T+6h label flow
│   └── 📁 unit/
│       ├── 📄 test_api.py         # FastAPI endpoints: /health (status, version, uptime), POST /vitals
│       │                         #   (valid 200, missing field 422, out-of-range 422, risk levels, alert trigger)
│       ├── 📄 test_features.py    # SOFA (normal, critical, partial, boundary), NEWS2, qSOFA,
│       │                         #   Rolling features (mean/std/trend columns, row count preserved)
│       ├── 📄 test_labeling.py    # T+6h label: non-sepsis=0, sepsis có positive, window correct,
│       │                         #   no future leakage, different horizons, split_by_patient no overlap
│       └── 📄 test_model.py       # XGBoost: predict_proba shape/range/sum-to-1, binary predict,
│                                  #   inference speed <200ms, save/load, untrained raises
│
└── 📁 web/                        # 🌐 Django Dashboard — port 8000
    ├── 📄 Dockerfile              # python:3.10-slim, daphne -b 0.0.0.0 -p 8000 config.asgi:application
    ├── 📄 manage.py               # Django CLI entry point
    ├── 📁 config/                  # Cấu hình Django cốt lõi
    │   ├── 📄 __init__.py
    │   ├── 📄 asgi.py             # ASGI: ProtocolTypeRouter (HTTP + WebSocket via Daphne)
    │   ├── 📄 settings.py         # INSTALLED_APPS (channels, dashboard), PostgreSQL, CHANNEL_LAYERS,
    │   │                         #   ML_SERVICE_URL, ALERT_SERVICE_URL
    │   ├── 📄 urls.py             # URL: /admin + / (include dashboard.urls)
    │   └── 📄 wsgi.py             # WSGI fallback
    ├── 📁 dashboard/               # Ứng dụng Dashboard
    │   ├── 📄 __init__.py
    │   ├── 📄 consumers.py        # AlertConsumer (AsyncWebsocketConsumer): group "alerts", push real-time
    │   ├── 📄 models.py           # Unmanaged models: Prediction (16 fields), Alert (10 fields), Patient
    │   ├── 📄 routing.py          # WebSocket: /ws/alerts/ → AlertConsumer
    │   ├── 📄 urls.py             # Routes: /, /patients/{id}/, /alerts/, /alerts/{id}/acknowledge/,
    │   │                         #   /api/patient/{id}/latest/, /api/alert-count/
    │   ├── 📄 views.py            # patient_list (stats + sort unconfirmed first), patient_detail
    │   │                         #   (risk chart + vitals + SHAP + early warning + acknowledge),
    │   │                         #   alerts_page (tabs + confirm), acknowledge_alert, API endpoints
    │   └── 📁 templates/dashboard/
    │       ├── 📄 alerts.html             # Bảng alert: tabs all/pending/confirmed, confirm button + toast
    │       ├── 📄 base.html               # Layout: navbar gradient, brand, clock, alert pill, WS live
    │       ├── 📄 patient_detail.html     # Risk score lớn + EW bars + vitals grid + charts + SHAP
    │       └── 📄 patient_list.html       # Stats cards (total/critical/warning/stable) + table + WS update
```

---

## 4. Công nghệ sử dụng

| Tầng | Công nghệ | Phiên bản | Mục đích sử dụng |
|------|-----------|-----------|------------------|
| **Ngôn ngữ** | Python | 3.10 | Ngôn ngữ lập trình chính |
| **Data Processing** | Pandas | 2.2.1 | Xử lý dữ liệu dạng bảng, rolling statistics, groupby operations |
| | NumPy | 1.26.4 | Tính toán số học, random generation, mảng đa chiều |
| | DuckDB | 0.10.1 | Truy vấn nhanh file Parquet trong quá trình training |
| | PyArrow | 15.0.2 | Định dạng cột cho Parquet, trao đổi dữ liệu hiệu năng cao |
| **Machine Learning** | XGBoost | 2.0.3 | Mô hình chính: predict_proba, early_stopping, scale_pos_weight |
| | Scikit-learn | 1.4.1 | Pipeline (SimpleImputer, StandardScaler), KNNImputer, StratifiedKFold |
| | imbalanced-learn | ≥0.11.0 | SMOTE oversampling (sampling_strategy=0.4) cho label imbalance |
| | SHAP | 0.45.0 | TreeExplainer giải thích top-5 features ảnh hưởng nhất |
| | Joblib | 1.3.2 | Lưu/tải model artifacts (preprocessor pipeline, model) |
| **Experiment Tracking** | MLflow | 2.11.1 | Tracking Server + Model Registry: log params, metrics, model artifacts |
| **Backend API** | FastAPI | 0.110.0 | ML Service (port 8001) + Alert Service (port 8002) |
| | Uvicorn | 0.28.1 | ASGI server cho FastAPI |
| | Pydantic | 2.6.4 | Validation schemas: VitalRequest, PredictionResponse, EarlyWarningResult |
| | httpx | 0.27.0 | Async HTTP client: ML Service → Alert Service, Dashboard ↔ Services |
| **Web Dashboard** | Django | 5.0.3 | Web framework chính (port 8000) |
| | Channels | 4.0.0 | Xử lý WebSocket real-time (AlertConsumer) |
| | Daphne | 4.1.0 | ASGI server cho Django + WebSocket |
| | psycopg2-binary | 2.9.9 | Kết nối Django ORM đến PostgreSQL |
| **Database** | PostgreSQL | 15 | Lưu trữ: patients, predictions, alerts, vital_records, admissions |
| | SQLAlchemy | 2.0.29 | ORM cho FastAPI services (PredictionORM, AlertORM) |
| **Container & CI/CD** | Docker | compose | 6 services: postgres, mlflow, prometheus, grafana, ml_service, alert_service, web |
| | pytest | 8.1.1 | Test framework (~70+ tests, 5 test files) |
| | pytest-cov | 4.1.0 | Code coverage tracking |
| | GitHub Actions | — | CI/CD pipeline: test → build → deploy |
| **Monitoring** | Prometheus | latest | Thu thập metrics: predictions_total, inference_seconds, active_alerts |
| | Grafana | latest | Dashboard trực quan hoá metrics hệ thống (port 3000) |
| | Evidently AI | 0.4.16 | DataDriftPreset: phát hiện drift giữa reference vs production 24h |
| | Prefect | 2.19.1 | Orchestration: retrain_flow (check_drift → run_training → promote) |

### File cấu hình & dependencies

```
📄 requirements.txt     — 28 dependencies (fastapi, django, xgboost, mlflow, evidently, prefect...)
📄 docker-compose.yml   — 6 services + bridge network `icu-network`
📄 pytest.ini            — testpaths=tests, verbose, short traceback
📄 monitoring/prometheus/prometheus.yml — scrape interval 15s, targets: ml_service:8001, alert_service:8002
📄 .env                  — POSTGRES_USER/PASS/DB, MLFLOW_TRACKING_URI, SECRET_KEY (not committed)
```

---

## 5. Dữ liệu

### 5.1 Tổng quan

Dự án sử dụng **dữ liệu synthetic** được sinh bởi `data_pipeline/data_generator.py`, mô phỏng sát các chỉ số sinh lý ICU thực tế. Bộ sinh dữ liệu gồm 2 mô hình thành phần:

```
ICUSepsisGenerator (──patients, ──hours, ──interval)
  ├── PhysiologicalModel   → Sinh 6 vitals: heart_rate, systolic_bp, diastolic_bp,
  │                           temperature, spo2, respiratory_rate
  │                         - Dùng baseline riêng (hr_baseline, temp_baseline, spo2_baseline)
  │                         - Severity ramp: onset→onset+12h, factor * severity_scale
  │                         - Noise: ~N(0, σ) với σ riêng cho từng vital
  │                         - Equipment noise: 2% ngẫu nhiên spike mỗi vital
  │
  └── LabResultModel       → Sinh 5 labs: lactate, wbc, creatinine, bilirubin, platelet
                            - Missing labs ngẫu nhiên ~5%
                            - WBC: 80% tăng / 20% giảm (sepsis có thể gây giảm bạch cầu)
```

### 5.2 Tham số sinh dữ liệu

| Tham số | Default | Ý nghĩa |
|---------|---------|---------|
| `--patients` | 20 | Số bệnh nhân ICU mô phỏng |
| `--hours` | 24 | Số giờ theo dõi |
| `--interval` | 5 phút | Chu kỳ lấy mẫu vitals |
| `--output` | `data/synthetic/icu_data_synthetic.csv` | File output |

Thông số đầu ra (mặc định):
- **20 bệnh nhân** × **24 giờ** × **12 mẫu/giờ** = **5.760 records**
- **40% sepsis** (8 bệnh nhân), **60% non-sepsis** (12 bệnh nhân)
- Severity distribution: mild 40%, moderate 40%, severe 20%
- Sepsis onset: random trong khoảng giờ **8–18** (giữa ca trực)

### 5.3 Các chỉ số (Features)

| Nhóm | Chỉ số | Kiểu | Khoảng sinh lý |
|------|--------|------|----------------|
| **Vitals** (6) | `heart_rate` | float | 30–220 bpm |
| | `systolic_bp` | float | 50–250 mmHg |
| | `diastolic_bp` | float | 30–150 mmHg |
| | `temperature` | float | 34.0–42.0 °C |
| | `spo2` | float | 70–100 % |
| | `respiratory_rate` | float | 4–60 /min |
| **Labs** (5) | `lactate` | float | 0–20 mmol/L |
| | `wbc` | float | 0.1–80.0 K/μL |
| | `creatinine` | float | 0–20 mg/dL |
| | `bilirubin` | float | 0–50 mg/dL |
| | `platelet` | float | 1–1000 K/μL |
| **Metadata** | `patient_id` | str | P0001–P0020 |
| | `timestamp` | datetime | UTC, cách đều interval |
| | `sepsis_label` | int 0/1 | 1 = patient có sepsis |
| | `sepsis_onset_hour` | float/NaN | Giờ onset (NaN nếu non-sepsis) |

### 5.4 Confounders (Built-in Overlap)

Để mô phỏng độ phức tạp của dữ liệu ICU thực tế, bộ sinh dữ liệu cố tình thêm các **confounders** gây khó cho mô hình:

| Confounder | Tỉ lệ | Mô tả |
|-----------|-------|-------|
| **Bad spikes (non-sepsis)** | 20% BN non-sepsis | Thoáng có vitals xấu giống sepsis (HR cao, SpO₂ thấp) |
| **Recovery period (sepsis)** | 15% BN sepsis | Thoáng có vitals bình thường (HR ổn, hết sốt) |
| **Equipment noise** | 2% / vital | Nhiễu ngẫu nhiên: HR±40, BP±35, SpO₂±8, Temp±1.2°C |
| **Missing labs** | ~5% / lab | Giá trị lab bị thiếu ngẫu nhiên |
| **Age confounder** | > 70 tuổi | Baseline vitals dịch chuyển ±10% (HR, BP cao hơn) |
| **Early normal labs (sepsis)** | 20% BN sepsis | Labs bình thường trong 8h đầu trước khi onset |
| **Mild abnormal labs (non-sepsis)** | 25% BN non-sepsis | Labs hơi bất thường dù không sepsis |

### 5.5 Pipeline dữ liệu

```
data_generator.py                labeling.py                     split_by_patient()
     │                               │                                │
     ▼                               ▼                                ▼
┌──────────────┐            ┌──────────────────┐            ┌──────────────────┐
│  Synthetic   │──CSV/DF───►│ create_t6h_labels│            │  Train (~60%)    │
│  ICU Data    │            │                  │            │  Val   (~20%)    │
│  (5760 rows) │            │ sepsis_in_next_6h│            │  Test  (~20%)    │
└──────────────┘            │ y[t] = 1 nếu     │            │                  │
                            │ onset ∈ (t, t+6h]│            │ Patient-based     │
                            │ y[t] = 0 còn lại │            │ (no leakage)     │
                            └──────────────────┘            └──────────────────┘
```

Pipeline training (`train.py`) tự động:
1. Load CSV → `create_t6h_labels()` → tạo cột `sepsis_in_next_6h`
2. `split_by_patient()` → chia theo patient_id, giữ tỉ lệ sepsis giữa các tập
3. `SimpleImputer(median)` → điền missing values
4. `StandardScaler` → chuẩn hoá features
5. Auto SMOTE (nếu imbalance ratio > 5) → xử lý mất cân bằng

### 5.6 Label distribution

| Label | Số lượng | Tỉ lệ |
|-------|----------|-------|
| `sepsis_label = 1` (có sepsis) | ~2.304 rows (40% BN) | ~40% |
| `sepsis_in_next_6h = 1` (positive) | ~576 rows | ~10% |
| `sepsis_in_next_6h = 0` (negative) | ~5.184 rows | ~90% |
| **Imbalance ratio** | **~9:1** | SMOTE ratio=0.4 |

### 5.7 Database schema

Hệ thống lưu dữ liệu vào **PostgreSQL** với 6 bảng (xem `docs/database_schema.sql`):

| Bảng | Mục đích | Số cột | Ghi chú |
|------|----------|--------|---------|
| `patients` | Thông tin bệnh nhân | 6 | patient_id PK, name, age, gender, ward |
| `admissions` | Lịch sử nhập viện | 6 | FK→patients, admitted_at, discharged_at, bed |
| `vital_records` | Chỉ số sinh tồn thô | 14 | FK→patients, timestamp, 6 vitals + 5 labs |
| `predictions` | Kết quả dự đoán (bảng chính) | 21 | risk_score, level, sofa, news2, 11 raw vitals, 5 early_warning scores |
| `alerts` | Cảnh báo | 12 | alert_id UUID, FK→patients, risk, acknowledged, ack_by |
| `prediction_results` | Bảng cũ (DEPRECATED) | 10 | Giữ lại cho tương thích ngược |

---

## 6. Hướng dẫn cài đặt

### Yêu cầu

- Docker & Docker Compose >= 2.0
- Git (hoặc tải source dưới dạng ZIP)

### Bước 1 — Clone và cấu hình môi trường

```bash
git clone https://github.com/nguyenthanh2911/CNM.git
cd CNM

cp .env.example .env
# Chỉnh sửa .env nếu cần (DB password, MLflow URI, ...)
```

### Bước 2 — Khởi động toàn bộ hệ thống

```bash
# Khởi động tất cả services
docker compose up -d

# (tuỳ chọn) Kiểm tra trạng thái nhanh
docker compose ps
```

Services sẽ chạy tại:

| Service | URL | Ghi chú |
|---------|-----|---------|
| Django Dashboard | http://localhost:8000 | Web UI chính — danh sách & chi tiết bệnh nhân |
| FastAPI ML Service | http://localhost:8001/docs | Swagger UI — thử API `/vitals`, `/health`, `/metrics` |
| FastAPI Alert Service | http://localhost:8002/docs | Swagger UI — WebSocket real-time alerts |
| MLflow UI | http://localhost:5000 | Theo dõi thí nghiệm training |
| Grafana | http://localhost:3000 | Dashboard giám sát (user/pass: `admin`/`admin`) |
| Prometheus | http://localhost:9090 | Metrics — nhập PromQL như `predictions_total`, `inference_seconds_count` |

> **Lưu ý:** Một số endpoint chỉ trả về JSON, không hiển thị giao diện web:
> - `http://localhost:8001/health` — kiểm tra ML Service hoạt động (trả về `{"status":"ok"}`)
> - `http://localhost:8002/health` — kiểm tra Alert Service hoạt động
> - `http://localhost:5432` — PostgreSQL (giao thức nhị phân, không phải HTTP). Dùng `psql` hoặc GUI tool (DBeaver, pgAdmin) để kết nối.
> - `http://localhost:9090/targets` — Prometheus targets page, kiểm tra trạng thái scrape

### Bước 3 — Khởi tạo database

```bash
# Khởi tạo bảng trong database
docker compose exec -T postgres sh -c "psql -U sepsis_user -d sepsis_db < /app/docs/database_schema.sql"

# Seed dữ liệu bệnh nhân ban đầu (chạy bên trong container)
docker compose exec -T ml_service python scripts/seed_patients.py
```

### Bước 4 — Chạy demo với Synthetic Data

Toàn bộ các lệnh bên dưới chạy **bên trong Docker** (không cần cài Python trên máy).
Sau thay đổi mới nhất, `PYTHONPATH=/app` đã được cấu hình sẵn trong `docker-compose.yml`, nên không cần thêm `-e PYTHONPATH=/app`.

```bash
# Sinh dữ liệu synthetic (chạy trong container)
docker compose exec -T ml_service python -m data_pipeline.data_generator --patients 20 --hours 24 --output data/synthetic/icu_data_synthetic.csv

# Train model T+6h và đăng ký lên MLflow
docker compose exec -T ml_service python -m ml.train --data data/synthetic/icu_data_synthetic.csv --experiment-name "CNM-Sepsis-T6H" --model-name "sepsis_xgboost_t6h" --augment

# Stream dữ liệu vào hệ thống (chạy ngầm mỗi 30 giây để test)
# Mô phỏng 20 bệnh nhân ICU real-time (chạy ngầm, mỗi 10s gửi vitals)
docker compose exec -d ml_service python scripts/simulate_realtime.py

# Mở dashboard: http://localhost:8000
```

---

## 7. Hướng dẫn sử dụng

### Chạy toàn bộ demo một lệnh

```bash
bash scripts/run_demo.sh
```

### Train mô hình (thủ công)

```bash
# Train model T+6h (khuyên dùng)
docker compose exec -T ml_service python -m ml.train \
    --data data/synthetic/icu_data_synthetic.csv \
    --experiment-name "CNM-Sepsis-T6H" \
    --model-name "sepsis_xgboost_t6h" \
    --augment

# Model sẽ tự động thêm label T+6h và log lên MLflow
# Xem kết quả trên MLflow UI: http://localhost:5000
```

### Mô phỏng dữ liệu real-time

```bash
# Chạy mô phỏng 20 bệnh nhân ICU trong 240 bước (mỗi bước 10 giây)
# 10 BN LOW (bình thường), 4 BN WARN (dần xấu), 6 BN HIGH (dần rất xấu)
docker compose exec -d ml_service python scripts/simulate_realtime.py
```

### Chạy test

```bash
# Toàn bộ test suite
docker compose exec -T ml_service pytest tests/ -v

# Chỉ unit test
docker compose exec -T ml_service pytest tests/unit/ -v
```

### Xoá toàn bộ dữ liệu (Reset database)

Lưu ý: Dashboard đang hiển thị dữ liệu từ bảng `predictions`, nên khi reset cần xoá cả `predictions` (ngoài `prediction_results`).

```bash
docker compose exec -T postgres psql -U sepsis_user -d sepsis_db -c "TRUNCATE TABLE alerts, prediction_results, predictions, vital_records, admissions, patients CASCADE;"
```

Nếu bạn đang chạy `scripts/simulate_realtime.py` thì dữ liệu sẽ được ghi lại ngay sau khi xoá; hãy dừng tiến trình đó trước (`docker compose stop ml_service`), rồi hard refresh Dashboard (`Ctrl+F5`).

---

## 8. Quy trình ML Pipeline

### Tổng quan

Pipeline machine learning gồm **8 bước**, từ sinh dữ liệu synthetic → serving → monitoring, được tổ chức trong các module `data_pipeline/`, `feature_engineering/`, `ml/`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA & TRAINING PIPELINE                              │
│                                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐                │
│  │ Step 1   │   │  Step 2      │   │  Step 3          │                │
│  │ Data Gen │──►│ T+6h Label   │──►│ Preprocessing    │                │
│  └──────────┘   └──────────────┘   └──────────────────┘                │
│       │                │                    │                           │
│  data_generator   labeling.py         ICUPreprocessor /                 │
│  .py              create_t6h_        sklearn Pipeline                   │
│  Physiological    labels()           (SimpleImputer +                   │
│  Model +           + split_by_       StandardScaler)                    │
│  LabResultModel    patient()                                            │
│       │                │                    │                           │
│       ▼                ▼                    ▼                           │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  Step 4: Feature Engineering                         │               │
│  │  ┌────────────────────────────────────────────────┐  │               │
│  │  │ feature_builder.py                             │  │               │
│  │  │ ├── Rolling stats (3/12/48 intervals)          │  │               │
│  │  │ │   ~ 15/60/240 phút — mean, std, min, max     │  │               │
│  │  │ ├── Trend features: diff(1) / interval_minutes │  │               │
│  │  │ ├── Clinical scores: SOFA, NEWS2, qSOFA        │  │               │
│  │  │ ├── Time-since-last-abnormal-HR                │  │               │
│  │  │ └── Drop raw vitals/labs columns               │  │               │
│  │  └────────────────────────────────────────────────┘  │               │
│  └──────────────────────────────────────────────────────┘               │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  Step 5: Training (train.py)                         │               │
│  │                                                      │               │
│  │  Input : CSV + FEATURE_COLS (11 raw features)        │               │
│  │  Label : sepsis_in_next_6h (T+6h)                    │               │
│  │                                                      │               │
│  │  1. Patient-based split (~60/20/20) — no leakage     │               │
│  │  2. SimpleImputer(median) + StandardScaler fit/trans │               │
│  │  3. Auto SMOTE (ratio=0.4) nếu imbalance_ratio > 5  │               │
│  │  4. 5-fold StratifiedKFold cross-validation          │               │
│  │     → Auto-regularize nếu std_auroc > 0.08           │               │
│  │       (max_depth=3, reg_lambda=3)                    │               │
│  │  5. SepsisXGBModel.fit():                            │               │
│  │     ┌────────────────────────────────────────────┐   │               │
│  │     │ n_estimators=150, max_depth=4, lr=0.05     │   │               │
│  │     │ subsample=0.65, colsample_bytree=0.65      │   │               │
│  │     │ min_child_weight=20, gamma=2               │   │               │
│  │     │ reg_alpha=1, reg_lambda=3, max_delta_step=1│   │               │
│  │     │ scale_pos_weight = neg/pos (auto)          │   │               │
│  │     │ eval_metric=['auc','logloss']              │   │               │
│  │     │ early_stopping_rounds=30                   │   │               │
│  │     └────────────────────────────────────────────┘   │               │
│  │  6. Evaluate → MLflow log → Register model           │               │
│  │     ├── test_auroc > 0.75 & gap(train-test) < 0.12  │               │
│  │     │   → Production                                 │               │
│  │     ├── test_auroc > 0.70 & gap < 0.18  → Staging   │               │
│  │     └── else → Not Registered (quality insufficient) │               │
│  └──────────────────────────────────────────────────────┘               │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  Step 6: MLflow Registry (mlflow_utils.py)           │               │
│  │                                                      │               │
│  │  log_training_run() → params + metrics + model       │               │
│  │    + feature_names.json → Tracking Server            │               │
│  │  register_model() → Model Registry (Production/Stag) │               │
│  │  load_production_model_with_metadata()               │               │
│  │    → ML Service loading (prefer Prod → Staging)      │               │
│  │  save to artifacts/preprocessor_t6h.joblib            │               │
│  └──────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE (SepsisPredictor)                  │
│                                                                         │
│  POST /vitals                                                           │
│      │                                                                  │
│      ▼                                                                  │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  1️⃣ Validate input (Pydantic VitalRequest)            │               │
│  │     heart_rate ∈ [20,250], spo2 ∈ [50,100], ...      │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  2️⃣ Clinical scores (từ raw vitals)                   │               │
│  │     calculate_sofa()   → 0–20 (5 thành phần)          │               │
│  │     calculate_news2()  → 0–20 (5 thành phần)          │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  3️⃣ Preprocess (sklearn Pipeline from joblib)         │               │
│  │     SimpleImputer(median) → StandardScaler           │               │
│  │     Fallback: fit-on-fly nếu pipeline chưa fit        │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  4️⃣ XGBoost predict_proba → risk_score ∈ [0, 1]      │               │
│  │                                                        │               │
│  │     risk < 0.3    → "LOW"        → log only           │               │
│  │     0.3 – 0.7    → "WARNING"    → Dashboard view      │               │
│  │     ≥ 0.7        → "CRITICAL"   → Alert Service + WS │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  5️⃣ SHAP TreeExplainer → top-5 features              │               │
│  │     Sắp xếp theo |shap_value| giảm dần, lấy top 5    │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  6️⃣ EarlyWarningPredictor (rule-based + ML T+6h)     │               │
│  │                                                        │               │
│  │     trend_score (30 phút history)    [weight 30%]     │               │
│  │     rate_of_change_score (đạo hàm)   [weight 20%]     │               │
│  │     threshold_score (ngưỡng lâm sàng) [weight 50%]    │               │
│  │     + risk_score_t6h từ XGBoost                        │               │
│  │     → early_warning_probability (0..1)                 │               │
│  │     → level: LOW / MEDIUM / HIGH                       │               │
│  │     → contributing_factors:                            │               │
│  │       "Lactate cao (4.2 mmol/L)",                      │               │
│  │       "Nhịp tim tăng nhanh (+15 trong 30 phút)",       │               │
│  │       "SpO2 giảm liên tục (97% → 90%)"                │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  7️⃣ Lưu PostgreSQL (predictions table, 21 cột)        │               │
│  │     risk_score, risk_level, alert_triggered,           │               │
│  │     sofa_score, news2_score, inference_time_ms,       │               │
│  │     6 raw vitals (HR, BP, SpO2, temp, RR),            │               │
│  │     5 labs (lactate, wbc, creat, bili, platelet),     │               │
│  │     5 early_warning scores (prob, level, trend,        │               │
│  │       rate_of_change, threshold)                      │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  8️⃣ Gọi Alert Service nếu risk ≥ 0.7                  │               │
│  │     hoặc early_warning HIGH (dù risk < 0.7)           │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                         │
│  Prometheus metrics exposed:                                            │
│    predictions_total (Counter)                                          │
│    predictions_by_risk_total{risk_level} (Counter)                      │
│    inference_seconds (Histogram)                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & RETRAINING                               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  Evidently AI — DataDriftPreset                       │               │
│  │  Reference: data/processed/features_train.parquet     │               │
│  │  Current  : vital_records trong 24h qua (PostgreSQL)  │               │
│  │  Drift score > 0.7 → is_drift = True                  │               │
│  │  Save HTML report: reports/drift/drift_YYYYMMDD_*     │               │
│  └──────────────────────┬───────────────────────────────┘               │
│                         ▼                                               │
│              ┌──────────────────────┐                                   │
│              │ drift_score > 0.7 ?  │                                   │
│              └──────┬───────┬──────┘                                   │
│                   Yes       No                                          │
│                     │        │                                          │
│                     ▼        │                                          │
│  ┌──────────────────────┐    │                                          │
│  │ Prefect retrain_flow │    │ No retrain                               │
│  │                      │    │                                          │
│  │ 1. run_training()    │    │                                          │
│  │    → train.py subproc│    │                                          │
│  │      với current data│    │                                          │
│  │ 2. compare_and_      │    │                                          │
│  │    promote()         │    │                                          │
│  │    New AUROC > Prod  │    │                                          │
│  │    AUROC + 0.01 ?    │    │                                          │
│  │      Yes → Promote   │    │                                          │
│  │      No  → Keep old  │    │                                          │
│  └──────────────────────┘    │                                          │
│         │                    │                                          │
│         ▼                    ▼                                          │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  Prometheus + Grafana                                │               │
│  │  Scrape: ml_service:8001/metrics, alert_svc:8002    │               │
│  │  Grafana dashboard: icu_dashboard.json (port 3000)   │               │
│  └──────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. API Reference

Hệ thống gồm **3 API services**, tất cả đều có Swagger UI tự động tại `/docs`:

| Service | Port | Swagger | Base URL (Docker) |
|---------|------|---------|-------------------|
| ML Service (FastAPI) | 8001 | http://localhost:8001/docs | `http://ml_service:8001` |
| Alert Service (FastAPI) | 8002 | http://localhost:8002/docs | `http://alert_service:8002` |
| Django Dashboard | 8000 | — | `http://web:8000` |

---

### 9.1 ML Service — `localhost:8001`

#### `POST /vitals` — Dự đoán nguy cơ Sepsis

Endpoint chính: nhận 11 chỉ số sinh lý, trả về risk score + SHAP + early warning.

**Request** (`VitalRequest` — Pydantic v2 validation):

| Trường | Kiểu | Bắt buộc | Khoảng | Mô tả |
|--------|------|----------|--------|-------|
| `patient_id` | `str` | ✅ | — | Mã bệnh nhân (VD: P001) |
| `timestamp` | `datetime` | ✅ | — | Thời điểm lấy mẫu (ISO 8601) |
| `heart_rate` | `float` | ✅ | 20–250 | Nhịp tim (bpm) |
| `systolic_bp` | `float` | ✅ | 40–300 | Huyết áp tâm thu (mmHg) |
| `diastolic_bp` | `float` | ✅ | 20–200 | Huyết áp tâm trương (mmHg) |
| `temperature` | `float` | ✅ | 30–45 | Nhiệt độ (°C) |
| `spo2` | `float` | ✅ | 50–100 | Độ bão hoà oxy (%) |
| `respiratory_rate` | `float` | ✅ | 4–60 | Nhịp thở (/phút) |
| `lactate` | `float` | ❌ | — | Lactate (mmol/L) |
| `wbc` | `float` | ❌ | — | Bạch cầu (K/μL) |
| `creatinine` | `float` | ❌ | — | Creatinine (mg/dL) |
| `bilirubin` | `float` | ❌ | — | Bilirubin (mg/dL) |
| `platelet` | `float` | ❌ | — | Tiểu cầu (K/μL) |

**Response** (`PredictionResponse`):

```json
{
  "patient_id": "P001",
  "timestamp": "2024-01-15T08:30:00Z",
  "risk_score": 0.82,
  "risk_level": "CRITICAL",
  "alert_triggered": true,
  "top_features": [
    {"feature": "heart_rate",         "shap_value": 0.31},
    {"feature": "spo2",               "shap_value": 0.24},
    {"feature": "lactate",            "shap_value": 0.19},
    {"feature": "systolic_bp",        "shap_value": 0.12},
    {"feature": "temperature",        "shap_value": 0.08}
  ],
  "sofa_score": 6,
  "news2_score": 9,
  "inference_time_ms": 45.2,
  "early_warning": {
    "early_warning_probability": 0.85,
    "early_warning_level": "HIGH",
    "time_window_minutes": 30,
    "trend_score": 0.42,
    "rate_of_change_score": 0.38,
    "threshold_score": 0.91,
    "contributing_factors": [
      "Nhịp tim cao (124 bpm)",
      "Huyết áp thấp (76 mmHg)",
      "SpO2 thấp (90.0%)",
      "Lactate cao (4.2 mmol/L)"
    ]
  }
}
```

**Response fields:**

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `risk_score` | `float` | Xác suất sepsis trong 6h tới (0.0–1.0) |
| `risk_level` | `str` | `LOW` (< 0.3) / `WARNING` (0.3–0.7) / `CRITICAL` (≥ 0.7) |
| `alert_triggered` | `bool` | `true` nếu risk_score ≥ 0.7 |
| `top_features` | `list` | Top-5 SHAP values (feature + shap_value) |
| `sofa_score` | `int` | SOFA score (0–20) từ raw vitals |
| `news2_score` | `int` | NEWS2 score (0–20) từ raw vitals |
| `inference_time_ms` | `float` | Thời gian inference (ms) |
| `early_warning` | `object` | Rule-based warning 30 phút |

**HTTP status codes:**
- `200 OK` — Thành công
- `422 Unprocessable Entity` — Dữ liệu đầu vào không hợp lệ (vượt range, thiếu field)

---

#### `GET /vitals/{patient_id}/history` — Lịch sử vitals + SHAP + early warning

**Response:**

```json
{
  "latest_vitals": {
    "heart_rate": 112,
    "systolic_bp": 88,
    "diastolic_bp": 54,
    "temperature": 39.1,
    "spo2": 93,
    "respiratory_rate": 24,
    "lactate": 4.2,
    "wbc": 15.3,
    "creatinine": 2.1,
    "bilirubin": 3.5,
    "platelet": 120
  },
  "top_features": [
    {"feature": "heart_rate", "shap_value": 0.31},
    {"feature": "spo2", "shap_value": 0.24}
  ],
  "early_warning": {
    "early_warning_probability": 0.85,
    "early_warning_level": "HIGH",
    "trend_score": 0.42,
    "rate_of_change_score": 0.38,
    "threshold_score": 0.91,
    "contributing_factors": ["Nhịp tim cao (124 bpm)"],
    "time_window_minutes": 30
  }
}
```

---

#### `GET /health` — Kiểm tra trạng thái service

```json
{
  "status": "ok",
  "model_version": "1",
  "model_auroc": 0.91,
  "uptime_seconds": 86400.0
}
```

---

#### `GET /metrics` — Prometheus metrics

| Metric | Kiểu | Labels | Mô tả |
|--------|------|--------|-------|
| `predictions_total` | Counter | — | Tổng số predictions đã thực hiện |
| `predictions_by_risk_total` | Counter | `risk_level="LOW\|WARNING\|CRITICAL"` | Số predictions theo mức risk |
| `inference_seconds` | Histogram | — | Latency inference (seconds) |

Middleware ghi `inference_seconds` tự động cho mọi request tới `/vitals`.

---

### 9.2 Alert Service — `localhost:8002`

```
POST  /alerts                        Tạo alert mới (ML Service gọi nội bộ)
GET   /alerts                        Danh sách alerts
GET   /alerts/stats                  Thống kê alerts hôm nay
GET   /health                        Health check
GET   /metrics                       Prometheus metrics
PATCH /alerts/{alert_id}/acknowledge Xác nhận alert (Dashboard gọi)
```

**`POST /alerts`** — Request body (`AlertCreate`):

```json
{
  "patient_id": "P001",
  "risk_score": 0.82,
  "risk_level": "CRITICAL",
  "alert_type": "sepsis",
  "top_features": [{"feature": "heart_rate", "shap_value": 0.31}],
  "sofa_score": 6,
  "news2_score": 9
}
```

**`GET /alerts`** — Query params:

| Param | Kiểu | Mặc định | Mô tả |
|-------|------|----------|-------|
| `patient_id` | `str` | — | Lọc theo bệnh nhân |
| `status` | `str` | `all` | `pending` / `confirmed` / `all` |
| `limit` | `int` | 50 | Tối đa 500 |

**Alert upsert logic:** Nếu đã có pending alert cho cùng patient → update risk_score, features; nếu chưa → tạo mới.

**WebSocket push:** Alert được push real-time tới `ws://localhost:8002/ws/alerts/{patient_id}`.

**Prometheus metric:** `active_alerts` (Gauge) — số alert đang pending.

---

### 9.3 Django Dashboard — `localhost:8000`

| Route | View | Phương thức | Mô tả |
|-------|------|------------|-------|
| `/` | `patient_list` | GET | Danh sách ICU real-time: stats cards + bảng + sort unconfirmed trước |
| `/patients/{patient_id}/` | `patient_detail` | GET | Chi tiết: risk score, vitals, SHAP chart, early warning bars, acknowledge |
| `/alerts/` | `alerts_page` | GET | Trang cảnh báo: tabs all/pending/confirmed |
| `/alerts/{alert_id}/acknowledge/` | `acknowledge_alert` | POST | Xác nhận alert (cập nhật Alert Service + Django DB, redirect về dashboard) |
| `/api/patient/{patient_id}/latest/` | `patient_latest_api` | GET | JSON API: risk_score, level, vitals, early_warning, shap_features |
| `/api/alert-count/` | `alert_count_api` | GET | JSON: `{"pending": N}` cho navbar badge |
| `ws://localhost:8000/ws/alerts/` | `AlertConsumer` | WebSocket | Push alert real-time qua Django Channels (group "alerts") |

---

## 10. Kết quả và đánh giá

### 10.1 Kết quả mô hình

Đánh giá trên **test set** (20% patients, label `sepsis_in_next_6h`, threshold binary = 0.4):

| Metric | Kết quả | Mục tiêu | Đánh giá |
|--------|---------|----------|----------|
| **AUROC** | **0.8270** | > 0.75 (Production) | ✅ Đạt Production threshold |
| **Sensitivity (Recall)** | **79%** | > 75% | ✅ Đạt |
| **Specificity** | **71%** | > 80% | ❌ Chưa đạt (imbalance cao) |
| **F1-score (thr=0.4)** | **0.26** | > 0.65 | ❌ Chưa đạt |
| **Imbalance ratio** | 9:1 | — | SMOTE ratio=0.4 đã áp dụng |
| **Positive label ratio** | ~10% | — | Label T+6h tự nhiên hiếm |
| **Train AUROC** | 0.91 | — | |
| **Val AUROC** | 0.85 | — | |
| **Gap (Train - Test)** | 0.083 | < 0.12 | ✅ Không overfit |

**Kết luận:** Mô hình đạt ngưỡng Production (AUROC > 0.75, gap < 0.12) và đáp ứng yêu cầu Sensitivity > 75%. Specificity chưa đạt target do bài toán T+6h có imbalance ratio ~9:1.

### 10.2 Kết quả hệ thống

| Metric | Kết quả | Mục tiêu | Đánh giá |
|--------|---------|----------|----------|
| **Inference latency (p95)** | ~95ms | < 200ms | ✅ Đạt |
| **Inference latency (p99)** | ~150ms | < 200ms | ✅ Đạt |
| **End-to-end alert latency** | ~2 phút | < 5 phút | ✅ Đạt |
| **Concurrent patients** | ≥ 20 | ≥ 20 | ✅ Đạt (ThreadPool 20 workers) |
| **MLflow Registration** | Staging / Production | — | Auto-register theo ngưỡng AUROC |
| **Data drift detection** | score > 0.7 | 0.7 | ✅ Prefect auto-retrain |
| **Model promotion gap** | +0.01 AUROC | +0.01 | ✅ So sánh tự động |
| **CV 5-fold mean AUROC** | ~0.80 | — | Auto-regularize nếu std > 0.08 |

### 10.3 Phân tích chi tiết

**Confusion Matrix** (test set, threshold = 0.4):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN (cao) | FP (thấp) |
| **Actual 1** | FN (thấp) | TP (cao) |

- **TP rate (Sensitivity):** 79% — phát hiện được phần lớn ca sepsis sắp xảy ra
- **FP rate:** 29% — một số cảnh báo giả, chấp nhận được trong môi trường ICU (an toàn hơn bỏ sót)
- **F1 thấp (0.26):** do precision thấp — hệ quả tất yếu của imbalance ratio cao, ưu tiên Sensitivity

**Inference pipeline breakdown:**

| Stage | Thời gian (ms) | % |
|-------|---------------|---|
| Preprocess (imputer + scaler) | ~5ms | 5% |
| Clinical scores (SOFA + NEWS2) | ~2ms | 2% |
| XGBoost predict_proba | ~35ms | 37% |
| SHAP TreeExplainer (top-5) | ~50ms | 53% |
| DB insert + Alert call | ~3ms | 3% |
| **Total** | **~95ms** | **100%** |

### 10.4 Hạn chế

| Hạn chế | Tác động | Nguyên nhân |
|---------|----------|-------------|
| Dữ liệu synthetic | Mô hình chưa đánh giá được trên dữ liệu ICU thật | Chưa tích hợp MIMIC-IV |
| XGBoost thuần | Chưa khai thác time-series dài hạn | Chỉ dùng rolling features thủ công |
| Specificity thấp (71%) | Nhiều cảnh báo giả | Imbalance ratio 9:1, ưu tiên Sensitivity |
| Chu kỳ 5 phút | Chưa hỗ trợ alert tức thời | Phù hợp hardware hiện tại |
| SHAP overhead | ~50% inference time | TreeExplainer chậm hơn predict |
| Thiếu .env.example | Khó clone cho người mới | File cấu hình chưa được commit |

### 10.5 Hướng phát triển

| Hướng | Mô tả | Ưu tiên |
|-------|-------|---------|
| **Tích hợp MIMIC-IV** | Train trên dữ liệu ICU thực, đánh giá generalization | 🔴 Cao |
| **LSTM / Transformer** | Thay thế rolling features bằng mô hình time-series học end-to-end | 🔴 Cao |
| **Threshold tuning** | Tìm threshold tối ưu trên val set để cân bằng Sens/Spec | 🟡 Trung bình |
| **Cost-sensitive learning** | Weight loss function theo chi phí FN > FP trong ICU | 🟡 Trung bình |
| **Rút chu kỳ < 1 phút** | Hạ interval xuống, tối ưu inference throughput | 🟢 Thấp |
| **Ensemble** | Kết hợp XGBoost + LightGBM để tăng AUROC | 🟢 Thấp |
| **Cấu hình mẫu** | Commit `.env.example` với các biến môi trường mặc định | 🟢 Thấp |

---

# CHƯƠNG 1 — PHÂN TÍCH, THIẾT KẾ

## 1.1 Mô tả bài toán

### Bối cảnh lâm sàng

**Sepsis** (nhiễm khuẩn huyết) là hội chứng đe dọa tính mạng xảy ra khi phản ứng của cơ thể với nhiễm trùng gây tổn thương các mô và cơ quan. Đây là một trong những nguyên nhân gây tử vong hàng đầu tại các đơn vị chăm sóc đặc biệt (ICU), với tỉ lệ tử vong lên đến **30–50%** trong các ca nặng.

Phát hiện sớm trong **1–6 giờ đầu** có thể tăng tỉ lệ sống sót lên 80%, nhưng thách thức lớn là:
- Y tá/bác sĩ ICU phải theo dõi **hàng chục chỉ số sinh lý** (nhịp tim, huyết áp, SpO₂, nhiệt độ, nhịp thở, xét nghiệm máu…) liên tục cho **nhiều bệnh nhân cùng lúc**
- Dấu hiệu sepsis giai đoạn đầu **dễ nhầm với các bệnh lý khác**
- Tần suất lấy mẫu thực tế (mỗi 4–8 giờ) có thể **bỏ sót cửa sổ vàng**
- Thiếu công cụ **tổng hợp đa chỉ số** thành một cảnh báo duy nhất, có giải thích

### Bài toán cụ thể

Xây dựng hệ thống **MLOps hoàn chỉnh** có khả năng:

1. **Thu thập tự động** 11 chỉ số sinh lý (6 vitals + 5 labs) từ thiết bị ICU hoặc mô phỏng
2. **Dự đoán nguy cơ sepsis** bằng mô hình XGBoost **trước 6 giờ** so với thời điểm khởi phát (label T+6h)
3. **Cảnh báo sớm** bằng rule-based engine trong **30 phút** tới dựa trên xu hướng vitals
4. **Giải thích kết quả** bằng SHAP — top-5 yếu tố ảnh hưởng nhất
5. **Hiển thị dashboard real-time** + push alert WebSocket đến nhân viên y tế
6. **Tự động phát hiện data drift** và retrain mô hình khi cần

### Đầu vào (Inputs)

| Nhóm | Chỉ số | Nguồn |
|------|--------|-------|
| **Vitals** (tần số 5 phút) | heart_rate, systolic_bp, diastolic_bp, temperature, spo2, respiratory_rate | Monitor ICU / Simulator |
| **Labs** (tần số thưa) | lactate, wbc, creatinine, bilirubin, platelet | Xét nghiệm máu |
| **Thông tin** | patient_id, timestamp, tuổi, giới tính, phòng | Hồ sơ bệnh nhân |

### Đầu ra (Outputs)

| Thành phần | Mô tả | Định dạng |
|-----------|-------|-----------|
| **Risk score** | Xác suất sepsis trong 6h tới | float ∈ [0, 1] |
| **Risk level** | Phân loại mức nguy hiểm | LOW / WARNING / CRITICAL |
| **Top-5 SHAP** | Đặc trưng ảnh hưởng nhất + giá trị | `[{feature, shap_value}]` |
| **Early warning** | Cảnh báo rule-based 30 phút + contributing factors | object |
| **Alert** | Cảnh báo CRITICAL + push WebSocket | JSON → Dashboard |
| **Dashboard** | Danh sách BN, chi tiết, biểu đồ, alerts | Web UI (Django) |

### Ràng buộc kỹ thuật

| Ràng buộc | Giá trị | Căn cứ |
|-----------|---------|--------|
| Chu kỳ dự đoán | 5 phút/lần | Tần số monitor ICU tiêu chuẩn |
| Độ trễ end-to-end | < 5 phút | Từ POST /vitals → WebSocket push |
| ML lead time | > 6 giờ trước sepsis | Label `sepsis_in_next_6h` |
| Rule-based lead time | > 30 phút trước sepsis | `EarlyWarningPredictor` 30 phút history |
| AUROC (Production) | > 0.75 | Ngưỡng register model `train.py` |
| Sensitivity | > 75% | Threshold binary = 0.4 |
| Inference latency | < 200ms / single row | Prometheus `inference_seconds` |
| Data drift threshold | score > 0.7 | Kích hoạt Prefect retrain |
| Số bệnh nhân hỗ trợ | ≥ 20 | ThreadPool 20 workers |

### Stakeholders

| Vai trò | Mối quan tâm chính |
|---------|-------------------|
| **Y tá ICU** | Dashboard real-time, nhận alert CRITICAL, acknowledge |
| **Bác sĩ ICU** | Risk score, SHAP explain, SOFA/NEWS2, xu hướng vitals |
| **Kỹ sư MLOps** | Training pipeline, MLflow tracking, monitoring drift |
| **Quản trị hệ thống** | Prometheus/Grafana metrics, Docker Compose |

---

## 1.2 Sơ đồ chức năng tổng quát

Hệ thống được chia thành **3 khối chức năng chính**, tương ứng với 3 tầng kiến trúc MLOps:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HỆ THỐNG ICU SEPSIS EARLY WARNING                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │  [1] QUẢN LÝ      │   │  [2] DỰ ĐOÁN &    │   │  [3] GIÁM SÁT &   │  │
│  │  BỆNH NHÂN        │   │  CẢNH BÁO         │   │  VẬN HÀNH         │  │
│  │  (Django Web)     │   │  (FastAPI + ML)   │   │  (MLOps Pipeline) │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│                                                                         │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │ 1.1 Xem danh sách │   │ 2.1 Thu thập      │   │ 3.1 Sinh dữ liệu  │  │
│  │     bệnh nhân     │   │     vitals (API)  │   │     synthetic     │  │
│  │  (patient_list)   │   │  (POST /vitals)   │   │  (data_generator) │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │ 1.2 Xem chi tiết  │   │ 2.2 Tiền xử lý    │   │ 3.2 Tạo label     │  │
│  │     + risk score  │   │     + feature eng │   │     T+6h          │  │
│  │  (patient_detail) │   │  (preprocess.py   │   │  (labeling.py)    │  │
│  │                   │   │   feature_builder)│   │                   │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │ 1.3 Xem lịch sử   │   │ 2.3 Dự đoán       │   │ 3.3 Huấn luyện    │  │
│  │     cảnh báo      │   │     risk score    │   │     mô hình       │  │
│  │  (alerts_page)    │   │ (XGBoost predict) │   │  (train.py)       │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │ 1.4 Acknowledge   │   │ 2.4 Giải thích    │   │ 3.4 Theo dõi      │  │
│  │     cảnh báo      │   │     SHAP top-5    │   │     thí nghiệm    │  │
│  │  (acknowledge)    │   │  (explain.py)     │   │  (MLflow)         │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │ 1.5 API polling   │   │ 2.5 EarlyWarning  │   │ 3.5 Phát hiện     │  │
│  │  (JSON endpoints) │   │     (rule-based)  │   │     data drift    │  │
│  │                   │   │  (early_warning)  │   │  (drift_detector) │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │                   │   │ 2.6 Gửi cảnh báo  │   │ 3.6 Tự động       │  │
│  │                   │   │     CRITICAL + WS │   │     retrain       │  │
│  │                   │   │  (alert_service)  │   │  (Prefect flow)   │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │                   │   │ 2.7 Lưu kết quả   │   │ 3.7 Theo dõi      │  │
│  │                   │   │    vào PostgreSQL │   │     hệ thống      │  │
│  │                   │   │  (predictions DB) │   │  (Prom+Grafana)   │  │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                           CHỨC NĂNG HỆ THỐNG                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  A. Xác thực & Bảo mật   B. CI/CD (GitHub Actions)  C. Container        │
│     ─ (chưa triển khai)     pytest → build → deploy   Docker Compose    │
│                                                         6 services      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Biểu đồ trường hợp sử dụng (Use Case)

### Actors

| Actor | Vai trò | Hệ thống tương tác |
|-------|---------|-------------------|
| **Y tá (Nurse)** | Người dùng chính — theo dõi dashboard, nhận & xác nhận alert | Django Dashboard, WebSocket |
| **Bác sĩ (Doctor)** | Xem risk score, SHAP explain, SOFA/NEWS2 — ra quyết định lâm sàng | Django Dashboard |
| **Admin / Kỹ sư MLOps** | Train/retrain model, theo dõi MLflow, cấu hình hệ thống | MLflow UI, Prefect, Docker |
| **Simulator** | Hệ thống tự động — gửi vitals mô phỏng mỗi 5 phút | FastAPI ML Service |

### Sơ đồ Use Case

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ICU SEPSIS EARLY WARNING SYSTEM                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  ┌──────────┐    ┌──────────────────────────────────────┐    │   │
│  │  │          │    │ UC01: Xem danh sách bệnh nhân        │    │   │
│  │  │          │    │        (patient_list view)           │    │   │
│  │  │  Y TÁ    │────┼──────────────────────────────────────┤    │   │
│  │  │ (Nurse)  │    │ UC02: Xem risk score real-time       │    │   │
│  │  │          │    │        (stats cards + bảng)          │    │   │
│  │  │          ├────┼──────────────────────────────────────┤    │   │
│  │  └──────────┘    │ UC03: Nhận cảnh báo CRITICAL        │    │   │
│  │       │          │        (WebSocket push, toast)       │    │   │
│  │       │          ├──────────────────────────────────────┤    │   │
│  │       │          │ UC04: Xem giải thích SHAP top-5     │    │   │
│  │       ├──────────┤        (bar chart, feature names)   │    │   │
│  │       │          ├──────────────────────────────────────┤    │   │
│  │       │          │ UC05: Acknowledge alert             │    │   │
│  │       └──────────┤        (POST acknowledge, redirect) │    │   │
│  │                  └──────────────────────────────────────┘    │   │
│  │                                                              │   │
│  │  ┌──────────────┐   ┌──────────────────────────────────────┐ │   │
│  │  │              │   │ UC06: Train / Retrain model          │ │   │
│  │  │  BÁC SĨ /    │───│        (train.py, Prefect flow)     │ │   │
│  │  │  ADMIN       │   ├──────────────────────────────────────┤ │   │
│  │  │              │   │ UC07: Xem báo cáo & metrics         │ │   │
│  │  └──────────────┘   │        (MLflow UI, Grafana)         │ │   │
│  │                     └──────────────────────────────────────┘ │   │
│  │                                                              │   │
│  │  ┌──────────────┐   ┌──────────────────────────────────────┐ │   │
│  │  │              │   │ UC08: Gửi vitals tự động            │ │   │
│  │  │  SIMULATOR   │───│        (simulate_realtime.py)       │ │   │
│  │  │              │   │        POST /vitals mỗi 5 phút     │ │   │
│  │  └──────────────┘   └──────────────────────────────────────┘ │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Mô tả chi tiết các Use Case

| ID | Use Case | Actor | Mô tả | Endpoint / Module |
|----|----------|-------|-------|-------------------|
| **UC01** | Xem danh sách bệnh nhân | Y tá | Hiển thị bảng ICU real-time: ID, tên, phòng, risk score bar, mức độ, sort unconfirmed trước | `/` → `patient_list` |
| **UC02** | Xem risk score real-time | Y tá | 4 stats cards (total/critical/warning/stable) + bảng cập nhật qua WebSocket | `/` → `patient_list` |
| **UC03** | Nhận cảnh báo CRITICAL | Y tá | Popup toast real-time khi risk ≥ 0.7, không cần reload trang | WebSocket `ws://:8000/ws/alerts/` |
| **UC04** | Xem giải thích SHAP | Bác sĩ, Y tá | Top-5 features dạng bar chart (heart_rate, lactate, spo2...), SOFA + NEWS2 badges | `/patients/{id}/` → `patient_detail` |
| **UC05** | Acknowledge alert | Y tá | Xác nhận đã xử lý alert, cập nhật Alert Service + Django DB, redirect về dashboard | `POST /alerts/{id}/acknowledge/` |
| **UC06** | Train / Retrain model | Admin, Kỹ sư | Chạy training pipeline (data → label → feature → XGBoost → MLflow), tự động hoặc thủ công | `train.py`, `Prefect retrain_flow` |
| **UC07** | Xem báo cáo & metrics | Admin, Kỹ sư | MLflow UI (params, metrics, model registry), Grafana dashboard (predictions, latency) | MLflow :5000, Grafana :3000 |
| **UC08** | Gửi vitals tự động | Simulator | Đọc CSV, ThreadPool 20 workers, POST /vitals mỗi 10s (mô phỏng 1h), 240 steps | `simulate_realtime.py` |
| **UC09** | Theo dõi hệ thống | Admin, Kỹ sư | Prometheus scrape metrics, Grafana dashboard, health check endpoints | Prometheus :9090, Grafana :3000, `/health` |
| **UC10** | Xem chi tiết bệnh nhân | Bác sĩ, Y tá | Risk score lớn + early warning bars + vitals grid (5 cards) + risk chart (2h) + SHAP chart | `/patients/{id}/` → `patient_detail` |
| **UC11** | Lọc danh sách alerts | Y tá | 3 tabs: all / pending / confirmed, confirm button với optimistic UI | `/alerts/` → `alerts_page` |
| **UC12** | API polling latest risk | Dashboard JS | JSON API trả về risk_score, vitals, early_warning, shap_features cho polling real-time | `/api/patient/{id}/latest/` |

---

## 1.4 Biểu đồ hoạt động

### 1.4.1 Luồng chính: Inference (chu kỳ 5 phút)

```
Simulator / Monitor            FastAPI ML Service            Alert Service          Django Dashboard
      │                              │                          │                        │
      │      1. POST /vitals         │                          │                        │
      │     {patient_id, HR, BP,     │                          │                        │
      │      SpO2, temp, RR, labs}   │                          │                        │
      │─────────────────────────────►│                          │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 2. Validate     │                  │                        │
      │                     │  Pydantic       │                  │                        │
      │                     │  VitalRequest   │                  │                        │
      │                     │  (range checks) │                  │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 3. Clinical     │                  │                        │
      │                     │  scores         │                  │                        │
      │                     │  SOFA + NEWS2   │                  │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 4. Preprocess   │                  │                        │
      │                     │  Imputer +      │                  │                        │
      │                     │  StandardScaler │                  │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 5. XGBoost      │                  │                        │
      │                     │  predict_proba  │                  │                        │
      │                     │  → risk_score   │                  │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 6. SHAP explain │                  │                        │
      │                     │  TreeExplainer  │                  │                        │
      │                     │  → top-5 feats  │                  │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 7. EarlyWarning │                  │                        │
      │                     │  trend + rate + │                  │                        │
      │                     │  threshold      │                  │                        │
      │                     │  → EW prob      │                  │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 8. Lưu PostgreSQL                 │                        │
      │                     │  predictions table               │                        │
      │                     │  (21 cột)                        │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │                     ┌────────▼────────┐                  │                        │
      │                     │ 9. Kiểm ngưỡng  │                  │                        │
      │                     │  risk_score     │                  │                        │
      │                     └────────┬────────┘                  │                        │
      │                              │                          │                        │
      │              ┌───────────────┼───────────────┐           │                        │
      │              │               │               │           │                        │
      │          < 0.3          0.3 - 0.7         ≥ 0.7         │                        │
      │              │               │               │           │                        │
      │         Log only        Dashboard      ┌──────▼──────┐   │                        │
      │                         WARNING        │ 10. POST    │   │                        │
      │                         badge vàng     │  /alerts    │   │                        │
      │                                         │─ ─ ─ ─ ─ ──►│                        │
      │                                         └──────┬──────┘   │                        │
      │                                                │          │                        │
      │                                     ┌──────────▼──────────┐                        │
      │                                     │ 11. Upsert + lưu DB │                        │
      │                                     │ 12. Push WebSocket  │                        │
      │                                     │ 13. Prometheus      │                        │
      │                                     │  active_alerts++    │                        │
      │                                     └──────────┬──────────┘                        │
      │                                                │  WebSocket                        │
      │                                                │  {alert_id, patient_id,           │
      │                                                │   risk_score, level}              │
      │                                                └───────────────────────────────────►│
      │                                                                                    │
      │                                                                           ┌────────▼────────┐
      │                              ◄──────────── response ─────────────────────│ 14. Hiển thị    │
      │                                {risk_score, risk_level,                   │  CRITICAL alert │
      │                                 top_features, early_warning,              │  + toast + glow │
      │                                 sofa_score, news2_score}                  └────────┬────────┘
      │                                                                                    │
      │                                                                         ┌──────────▼──────────┐
      │                                                                         │ 15. Nurse click     │
      │                                                                         │  Acknowledge        │
      │                                                                         │  → POST /alerts/{id}│
      │                                                                         │  → DB update        │
      │                                                                         │  → highlight xanh   │
      │                                                                         └─────────────────────┘
```

### 1.4.2 Luồng Training

```
data_generator.py                     labeling.py                     train.py
      │                                    │                              │
      │  1. Sinh synthetic data            │                              │
      │  PhysiologicalModel +              │                              │
      │  LabResultModel                    │                              │
      │  → icu_data_synthetic.csv          │                              │
      │   (5760 rows, 20 patients)         │                              │
      └───────────────────────────────────►│                              │
                                           │                              │
                                           │  2. Tạo label T+6h           │
                                           │  create_t6h_labels()          │
                                           │  → sepsis_in_next_6h          │
                                           │                              │
                                           │  3. Patient-based split       │
                                           │  split_by_patient()           │
                                           │  Train 60% / Val 20% /       │
                                           │  Test 20%                     │
                                           └─────────────────────────────►│
                                                                          │
                                                                          │  4. Preprocess Pipeline
                                                                          │  SimpleImputer(median)
                                                                          │  + StandardScaler
                                                                          │  → preprocessor.joblib
                                                                          │
                                                                          │  5. Auto SMOTE (ratio=0.4)
                                                                          │  (nếu imbalance > 5)
                                                                          │
                                                                          │  6. 5-fold StratifiedKFold
                                                                          │  CV diagnostics
                                                                          │  (auto-regularize nếu
                                                                          │   std_auroc > 0.08)
                                                                          │
                                                                          │  7. XGBoost fit
                                                                          │  early_stopping=30
                                                                          │  eval_set=[(X_val, y_val)]
                                                                          │
                                                                          │  8. Evaluate test set
                                                                          │  AUROC, F1, Sens, Spec
                                                                          │
                                                                          │  9. MLflow log
                                                                          │  params + metrics + model
                                                                          │
                                                                          │  10. Register model
                                                                          │  test_auroc > 0.75 ? Prod
                                                                          │  test_auroc > 0.70 ? Stag
                                                                          │  else → Not Registered
```

### 1.4.3 Luồng Retrain (tự động)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Prefect retrain_flow (@flow)                         │
│                                                                         │
│   ┌──────────────────────────┐                                          │
│   │  check_drift()           │                                          │
│   │  ┌────────────────────┐  │                                          │
│   │  │ Evidently AI       │  │                                          │
│   │  │ DataDriftPreset    │  │                                          │
│   │  │ Reference: parquet │  │                                          │
│   │  │ Current: 24h DB    │  │                                          │
│   │  └────────┬───────────┘  │                                          │
│   │           │              │                                          │
│   │     drift_score          │                                          │
│   └──────────┬───────────────┘                                          │
│              │                                                          │
│     ┌────────┴────────┐                                                 │
│     │ drift > 0.7 ?   │                                                 │
│     └──┬──────────┬───┘                                                 │
│      Yes          No                                                    │
│        │           │                                                    │
│        ▼           ▼                                                    │
│  ┌───────────┐  No retrain                                              │
│  │ run_      │  (keep current                                           │
│  │ training()│   model)                                                 │
│  │ ┌───────┐ │                                                          │
│  │ │train.py│ │                                                          │
│  │ │subproc │ │                                                          │
│  │ └───┬───┘ │                                                          │
│  │     │     │                                                          │
│  │  new_auroc│                                                          │
│  └─────┬─────┘                                                          │
│        │                                                                  │
│        ▼                                                                  │
│  ┌──────────────────────────┐                                            │
│  │  compare_and_promote()   │                                            │
│  │  new_auroc > prod_auroc  │                                            │
│  │  + 0.01 ?                │                                            │
│  └──┬──────────────┬────────┘                                            │
│   Yes              No                                                    │
│     │               │                                                    │
│     ▼               ▼                                                    │
│  Promote to      Keep old                                                │
│  Production      model                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1.5 Biểu đồ trình tự (Sequence Diagram)

### 1.5.1 Luồng Inference CRITICAL (UC02 + UC03)

```
Simulator        FastAPI_ML          PostgreSQL        AlertService       Django_WS         Nurse
    │                 │                  │                  │                 │               │
    │  POST /vitals   │                  │                  │                 │               │
    │  {HR, BP, SpO2, │                  │                  │                 │               │
    │   temp, RR, lab}│                  │                  │                 │               │
    │────────────────►│                  │                  │                 │               │
    │                 │  validate(Pydantic)                 │                 │               │
    │                 │──────────────────►                  │                 │               │
    │                 │◄──── ok ──────────                  │                 │               │
    │                 │                  │                  │                 │               │
    │                 │  calculate_sofa() │                  │                 │               │
    │                 │  calculate_news2()│                  │                 │               │
    │                 │  preprocess()     │                  │                 │               │
    │                 │  predict_proba()  │                  │                 │               │
    │                 │  shap.explain()   │                  │                 │               │
    │                 │  early_warning()  │                  │                 │               │
    │                 │                  │                  │                 │               │
    │                 │  INSERT INTO predictions            │                 │               │
    │                 │  (risk, level, sofa, news2,         │                 │               │
    │                 │   raw vitals, early_warning)        │                 │               │
    │                 │──────────────────►                  │                 │               │
    │                 │◄──── saved ───────                  │                 │               │
    │                 │                  │                  │                 │               │
    │                 │  [risk_score = 0.82 ≥ 0.7]          │                 │               │
    │                 │                  │                  │                 │               │
    │                 │  POST /alerts     │                  │                 │               │
    │                 │  {patient_id, risk, level,           │                 │               │
    │                 │   top_features, sofa, news2}        │                 │               │
    │                 │────────────────────────────────────►│                 │               │
    │                 │                  │                  │                 │               │
    │                 │                  │  upsert pending   │                 │               │
    │                 │                  │──────────────────►                 │               │
    │                 │                  │◄──── saved ───────                 │               │
    │                 │                  │                  │                 │               │
    │                 │                  │  WebSocket push   │                 │               │
    │                 │                  │  {alert_id,       │                 │               │
    │                 │                  │   patient_id,     │                 │               │
    │                 │                  │   risk_score:0.82,│                 │               │
    │                 │                  │   level:CRITICAL} │                 │               │
    │                 │                  │                  │────────────────►│               │
    │                 │                  │                  │                 │               │
    │                 │                  │                  │  CRITICAL toast  │               │
    │                 │                  │                  │  + glow animation               │
    │                 │                  │                  │                 │──────────────►│
    │                 │                  │                  │                 │               │
    │◄─── response────│                  │                  │                 │               │
    │  {risk_score:   │                  │                  │                 │               │
    │   0.82,         │                  │                  │                 │               │
    │   level:CRITICAL,                  │                  │                 │               │
    │   top_features, │                  │                  │                 │               │
    │   early_warning,│                  │                  │                 │               │
    │   sofa:6,       │                  │                  │                 │               │
    │   news2:9}      │                  │                  │                 │               │
    │                 │                  │                  │                 │               │
```

### 1.5.2 Luồng Acknowledge Alert (UC05)

```
Nurse              Django Dashboard        Alert Service           PostgreSQL
  │                       │                      │                     │
  │  Click "Confirm"      │                      │                     │
  │  POST /alerts/{id}/   │                      │                     │
  │  acknowledge/         │                      │                     │
  │──────────────────────►│                      │                     │
  │                       │  PATCH /alerts/{id}/ │                     │
  │                       │  acknowledge         │                     │
  │                       │  {ack_by: dashboard} │                     │
  │                       │─────────────────────►│                     │
  │                       │                      │                     │
  │                       │            ┌─────────▼─────────┐           │
  │                       │            │ UPDATE alerts     │           │
  │                       │            │ SET acknowledged= │           │
  │                       │            │ true, ack_by,     │           │
  │                       │            │ ack_at=NOW()      │           │
  │                       │            │ WHERE alert_id=?  │           │
  │                       │            └─────────┬─────────┘           │
  │                       │                      │                     │
  │                       │◄──── 200 OK ─────────                      │
  │                       │                      │                     │
  │            ┌──────────▼──────────┐           │                     │
  │            │ UPDATE Django DB    │           │                     │
  │            │ Alert.objects.filter│           │                     │
  │            │ .update(acknowledged│           │                     │
  │            │ =True, ...)        │           │                     │
  │            └──────────┬──────────┘           │                     │
  │                       │                      │                     │
  │  Redirect /?confirmed={patient_id}           │                     │
  │◄──── 302 Redirect ────                      │                     │
  │                       │                      │                     │
  │  Dashboard tải lại     │                      │                     │
  │  → row chuyển màu xanh │                      │                     │
  │  → xuống cuối danh sách│                      │                     │
```

### 1.5.3 Luồng Training (UC06)

```
User/Admin         train.py (CLI)          MLflow Tracking       MLflow Registry
    │                    │                      │                     │
    │  python -m ml.train                      │                     │
    │  --data icu_data_synthetic.csv           │                     │
    │  --augment              │                      │                     │
    │───────────────────────►│                      │                     │
    │                        │                      │                     │
    │                        │  create_t6h_labels() │                     │
    │                        │  split_by_patient() │                     │
    │                        │  SimpleImputer()    │                     │
    │                        │  StandardScaler()   │                     │
    │                        │  SMOTE()            │                     │
    │                        │  5-fold CV          │                     │
    │                        │  XGBoost.fit()      │                     │
    │                        │  evaluate_model()   │                     │
    │                        │                      │                     │
    │                        │  log_params()        │                     │
    │                        │  log_metrics()       │                     │
    │                        │  log_model()         │                     │
    │                        │─────────────────────►│                     │
    │                        │                      │                     │
    │                        │                      │  register_model()   │
    │                        │                      │  (if auroc > 0.75)  │
    │                        │                      │────────────────────►│
    │                        │                      │                     │
    │                        │                      │◄── version "1" ────│
    │                        │                      │                     │
    │                        │  Run ID: abc123      │                     │
    │                        │  Test AUROC: 0.827   │                     │
    │                        │  Stage: Production   │                     │
    │◄────── output ─────────                      │                     │
    │                        │                      │                     │
```

### 1.5.4 Luồng Retrain tự động (Prefect)

```
Prefect Scheduler      Evidently AI            Prefect retrain_flow        MLflow
    │                      │                         │                     │
    │  Trigger (cron)      │                         │                     │
    │─────────────────────►│                         │                     │
    │                      │                         │                     │
    │                      │  check_drift() task      │                     │
    │                      │  ┌─────────────────┐    │                     │
    │                      │  │ DataDriftPreset  │    │                     │
    │                      │  │ ref=parquet      │    │                     │
    │                      │  │ cur=PostgreSQL   │    │                     │
    │                      │  └────────┬────────┘    │                     │
    │                      │           │             │                     │
    │                      │  drift_score = 0.82     │                     │
    │                      │◄──── is_drift = True ───                     │
    │                      │                         │                     │
    │                      │  run_training() task     │                     │
    │                      │  python train.py ...    │                     │
    │                      │─────────────────────────►                     │
    │                      │                         │                     │
    │                      │                         │  log + register     │
    │                      │                         │────────────────────►│
    │                      │                         │                     │
    │                      │◄── new_auroc = 0.84 ────                     │
    │                      │                         │                     │
    │                      │  compare_and_promote()  │                     │
    │                      │  0.84 > 0.82 + 0.01 ?   │                     │
    │                      │  Yes → Promote to Prod  │                     │
    │                      │─────────────────────────►                     │
    │                      │                         │  transition to      │
    │                      │                         │  "Production"       │
    │                      │                         │────────────────────►│
    │                      │                         │                     │
    │                      │◄── promoted = True ─────                     │
```

---

## 1.6 Biểu đồ Lớp (Class Diagram)

### 1.6.1 Database Models (PostgreSQL ORM)

```
┌─────────────────────────────┐       ┌──────────────────────────────┐
│          Patient            │       │         Admission            │
├─────────────────────────────┤       ├──────────────────────────────┤
│ - patient_id: str (PK)      │1──────►│ - admission_id: uuid (PK)   │
│ - name: str                 │       │ - patient_id: str (FK)       │
│ - age: int                  │       │ - admitted_at: timestamp     │
│ - gender: str               │       │ - discharged_at: timestamp   │
│ - ward: str                 │       │ - bed_number: str            │
│ - created_at: timestamp     │       │ - status: str                │
├─────────────────────────────┤       └──────────────────────────────┘
│ + get_latest_risk()         │                    │ 1
│ + get_alert_history()       │                    │
└─────────────────────────────┘                    ▼ *
                                       ┌──────────────────────────────┐
                                       │       VitalRecord            │
                                       ├──────────────────────────────┤
                                       │ - record_id: uuid (PK)       │
                                       │ - patient_id: str (FK)       │
                                       │ - timestamp: timestamp       │
                                       │ - heart_rate: float          │
                                       │ - systolic_bp: float         │
                                       │ - diastolic_bp: float        │
                                       │ - temperature: float         │
                                       │ - spo2: float                │
                                       │ - respiratory_rate: float    │
                                       │ - lactate: float?            │
                                       │ - wbc: float?                │
                                       │ - creatinine: float?         │
                                       │ - bilirubin: float?          │
                                       │ - platelet: float?           │
                                       ├──────────────────────────────┤
                                       │ + to_feature_vector()        │
                                       └──────────────────────────────┘
                                                   │ 1
                                                   ▼ *
┌─────────────────────────────┐       ┌──────────────────────────────┐
│       PredictionORM        │       │         AlertORM              │
├─────────────────────────────┤       ├──────────────────────────────┤
│ - id: int (PK, auto)       │       │ - id: int (PK, auto)         │
│ - patient_id: str (FK)     │       │ - alert_id: str (UUID, UK)   │
│ - timestamp: timestamptz   │       │ - patient_id: str (FK)       │
│ - risk_score: float        │       │ - risk_score: float          │
│ - risk_level: str          │       │ - risk_level: str            │
│ - alert_triggered: bool    │       │ - alert_type: str            │
│ - sofa_score: int          │       │ - top_features: JSONB        │
│ - news2_score: int         │       │ - sofa_score: int            │
│ - inference_time_ms: float │       │ - news2_score: int           │
│ - heart_rate: float?       │       │ - created_at: timestamptz    │
│ - systolic_bp: float?      │       │ - acknowledged: bool         │
│ - diastolic_bp: float?     │       │ - ack_by: str?               │
│ - temperature: float?      │       │ - ack_at: timestamptz?       │
│ - spo2: float?             │       └──────────────────────────────┘
│ - respiratory_rate: float? │
│ - lactate: float?          │
│ - wbc: float?              │
│ - creatinine: float?       │
│ - bilirubin: float?        │
│ - platelet: float?         │
│ - early_warning_prob: float?│
│ - early_warning_level: str?│
│ - trend_score: float?      │
│ - rate_of_change: float?   │
│ - threshold_score: float?  │
│ - created_at: timestamptz  │
└─────────────────────────────┘
```

### 1.6.2 ML Service Classes (FastAPI)

```
┌─────────────────────────────────┐
│     SepsisPredictor (Singleton) │
├─────────────────────────────────┤
│ - model: XGBoost | None         │
│ - explainer: SepsisExplainer    │
│ - preprocess_pipeline: Pipeline │
│ - _ew_predictor: EarlyWarning   │
│ - model_name: str               │
│ - model_version: str            │
│ - model_auroc: float            │
│ - _vitals_cache: dict           │
│ - _shap_cache: dict             │
├─────────────────────────────────┤
│ + get_instance() -> Self        │
│ + predict(req: VitalRequest)    │
│     -> PredictionResponse       │
│ + get_history(patient_id)       │
│     -> dict                     │
│ - _load_artifacts()             │
│ - _predict_proba(X) -> np.array │
└─────────────────────────────────┘
         │
         ├──uses───────────────────────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────┐   ┌────────────────────────────────┐
│   SepsisXGBModel        │   │   EarlyWarningPredictor         │
├─────────────────────────┤   ├────────────────────────────────┤
│ - params: dict          │   │ - history: dict[str, deque]    │
│ - model: XGBClassifier  │   │ - maxlen: int = 6              │
├─────────────────────────┤   ├────────────────────────────────┤
│ + fit(X, y, Xv, yv)     │   │ + update(patient_id, vitals)   │
│ + predict_proba(X)      │   │ + predict_early_warning(       │
│ + predict(X, thr=0.4)   │   │     pid, vitals, ml_score)     │
│ + cross_validate(X, y)  │   │     -> dict                    │
│ + save(path) / load()   │   │ - _trend_score(pid) -> float   │
└─────────────────────────┘   │ - _rate_of_change_score(pid)   │
         │                    │ - _threshold_score(vitals)     │
         │                    │ - _get_contributing_factors()  │
         ▼                    └────────────────────────────────┘
┌─────────────────────────┐
│   SepsisExplainer       │
├─────────────────────────┤
│ - model: XGBModel       │
│ - _explainer: TreeExplainer│
├─────────────────────────┤
│ + explain(X_instance,   │
│     feature_names)      │
│     -> list[dict]       │
└─────────────────────────┘
```

### 1.6.3 Pydantic Schemas

```
┌───────────────────────┐     ┌──────────────────────────────┐
│    VitalRequest       │     │     PredictionResponse       │
├───────────────────────┤     ├──────────────────────────────┤
│ + patient_id: str     │     │ + patient_id: str            │
│ + timestamp: datetime │     │ + timestamp: datetime        │
│ + heart_rate: float   │     │ + risk_score: float          │
│   (ge=20, le=250)     │     │ + risk_level: str            │
│ + systolic_bp: float  │     │ + alert_triggered: bool      │
│ + diastolic_bp: float │     │ + top_features: list[Feature]│
│ + temperature: float  │     │ + sofa_score: int            │
│ + spo2: float         │     │ + news2_score: int           │
│ + respiratory_rate:   │     │ + inference_time_ms: float   │
│     float             │     │ + early_warning: EarlyWarn   │
│ + lactate: float?     │     └──────────────────────────────┘
│ + wbc: float?         │
│ + creatinine: float?  │     ┌──────────────────────────────┐
│ + bilirubin: float?   │     │    EarlyWarningResult        │
│ + platelet: float?    │     ├──────────────────────────────┤
└───────────────────────┘     │ + probability: float (0..1)  │
                              │ + level: str (LOW/MED/HIGH)  │
┌───────────────────────┐     │ + time_window_minutes: 30    │
│   FeatureExplanation  │     │ + trend_score: float         │
├───────────────────────┤     │ + rate_of_change_score: float│
│ + feature: str        │     │ + threshold_score: float     │
│ + shap_value: float   │     │ + contributing_factors: list │
└───────────────────────┘     └──────────────────────────────┘
```

### 1.6.4 Django Models (Unmanaged)

```
┌─────────────────────────────┐   ┌──────────────────────────────┐
│  Prediction (Django Model)  │   │  Alert (Django Model)        │
├─────────────────────────────┤   ├──────────────────────────────┤
│  Meta: managed=False        │   │  Meta: managed=False         │
│  db_table = "predictions"   │   │  db_table = "alerts"         │
│                             │   │                              │
│  (16 fields tương tự        │   │  (10 fields tương tự         │
│   PredictionORM)            │   │   AlertORM)                  │
│                             │   │                              │
└─────────────────────────────┘   └──────────────────────────────┘

┌─────────────────────────────┐
│  Patient (Django Model)     │
├─────────────────────────────┤
│  Meta: managed=False        │
│  db_table = "patients"      │
│                             │
│ - patient_id: str (PK)      │
│ - name: str                 │
│ - age: int?                 │
│ - gender: str?              │
│ - ward: str?                │
│ - created_at: datetime?     │
└─────────────────────────────┘
```

---

## 1.7 Biểu đồ luồng dữ liệu (Data Flow Diagram)

### Mức 0 — Context Diagram

```
                         ┌─────────────────────────────────────┐
                         │                                     │
  ┌────────────┐         │       HỆ THỐNG CẢNH BÁO SỚM        │
  │ Simulator  │◄────────┤          ICU SEPSIS                 │
  │ (thiết bị  │  response│        (HỆ THỐNG)                  │
  │  ICU)      │────────►│                                     │
  └────────────┘  vitals │  ┌─────────────────────────────┐    │
                         │  │ 1.0 Xử lý & Dự đoán        │    │
  ┌────────────┐         │  │ 2.0 Cảnh báo & Dashboard   │    │
  │ Y tá /     │◄────────┤  │ 3.0 Giám sát & Retrain     │    │
  │ Bác sĩ     │  alert  │  └─────────────────────────────┘    │
  │ (Nhân viên │────────►│                                     │
  │  y tế)     │ acknowledge│                                   │
  └────────────┘         └─────────────────────────────────────┘
                              │
                              ▼
                         ┌────────────┐
                         │  Admin    │
                         │ (Kỹ sư)  │
                         └────────────┘
```

### Mức 1 — DFD chi tiết

```
                          ┌──────────────────────────────────────────────┐
                          │               DỮ LIỆU NGOẠI                │
                          │                                              │
  ┌────────────┐         │  ┌─────────┐  ┌──────────┐  ┌────────────┐  │
  │ Simulator  │         │  │patients │  │admissions │  │vital_records│  │
  │ (thiết bị) │────────►│  │  (D1)   │  │  (D2)    │  │   (D3)     │  │
  └────────────┘ vitals  │  └─────────┘  └──────────┘  └────────────┘  │
                          └──────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      1.0             │
                    │  Tiếp nhận &         │
                    │  Kiểm tra vitals     │
                    │  (POST /vitals)      │
                    │  Pydantic Validation │
                    └──────────┬───────────┘
                               │ vitals đã validate
                               ▼
                    ┌──────────────────────┐
                    │      1.1             │
                    │  Clinical Scores     │
                    │  SOFA + NEWS2        │
                    │  (clinical_scores.py) │
                    └──────────┬───────────┘
                               │ scores
                               ▼
                    ┌──────────────────────┐
                    │      1.2             │
                    │  Preprocess Pipeline │
                    │  Imputer + Scaler    │
                    │  (ICUPreprocessor)   │
                    └──────────┬───────────┘
                               │ X_scaled
                               ▼
                    ┌──────────────────────┐
                    │      1.3             │
                    │  XGBoost Predict     │
                    │  predict_proba →     │
                    │  risk_score (0..1)   │
                    │  (SepsisXGBModel)    │
                    └──────────┬───────────┘
                               │ risk_score
                               ▼
                    ┌──────────────────────┐
                    │      1.4             │
                    │  SHAP Explain        │
                    │  TreeExplainer       │
                    │  → top-5 features    │
                    │  (SepsisExplainer)   │
                    └──────────┬───────────┘
                               │ risk + shap
                               ▼
                    ┌──────────────────────┐
                    │      1.5             │
                    │  EarlyWarning        │
                    │  Predictor           │
                    │  (trend+rate+thresh) │
                    │  (EarlyWarningPred)  │
                    └──────────┬───────────┘
                               │ risk + ew
                               ▼
                    ┌──────────────────────┐
                    │      1.6             │
                    │  Lưu PostgreSQL      │
                    │  predictions table   │
                    │  (21 cột)            │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  predictions (D4)    │
                    │  PostgreSQL          │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Kiểm tra ngưỡng     │
                    │  risk_score ≥ 0.7 ?  │
                    └──────────┬───────────┘
                         ┌─────┴─────┐
                      ≥ 0.7       < 0.7
                         │            │
                         ▼            ▼
                    ┌──────────┐  Log only
                    │  2.0     │  (Dashboard
                    │  Alert   │   WARNING
                    │  Service │   nếu ≥0.3)
                    │  :8002   │
                    └────┬─────┘
                         │ alert
                         ▼
                    ┌──────────┐        ┌──────────────────┐
                    │  alerts  │────────│  WebSocket Push   │
                    │  (D5)    │        │ → Django :8000   │
                    │ Postgres │        │ AlertConsumer    │
                    └──────────┘        └────────┬─────────┘
                                                 │
                                                 ▼
                                          ┌──────────────────┐
                                          │  Dashboard       │
                                          │  patient_list    │
                                          │  patient_detail  │
                                          │  alerts_page     │
                                          │  (Django Views)  │
                                          └──────────────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    │                         │
                                    ▼                         ▼
                             ┌──────────────┐         ┌──────────────┐
                             │  Y tá /      │         │  Acknowledge │
                             │  Bác sĩ     │         │  PATCH /     │
                             │  xem alert   │         │  alerts/{id} │
                             └──────────────┘         └──────────────┘
```

### Mức 2 — DFD Giám sát & Retrain

```
                    ┌───────────────────────────────┐
                    │  reference_data.parquet (D6)  │
                    │  (data/processed/)            │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │  3.0 Drift Detector           │
                    │  Evidently AI DataDriftPreset │
                    │  ref vs current 24h (D4)     │
                    └──────────────┬────────────────┘
                                   │ drift_score
                                   ▼
                    ┌───────────────────────────────┐
                    │  drift_score > 0.7 ?          │
                    └──────────────┬────────────────┘
                             ┌─────┴─────┐
                          Yes            No
                             │             │
                             ▼             ▼
                    ┌──────────────┐  Keep current
                    │  3.1 Prefect │  model
                    │  retrain_flow│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  train.py    │
                    │  (subprocess)│
                    └──────┬───────┘
                           │ new model
                           ▼
                    ┌───────────────────────────────┐
                    │  3.2 compare_and_promote()   │
                    │  new_auroc > prod + 0.01 ?  │
                    └──────────────┬────────────────┘
                             ┌─────┴─────┐
                           Yes           No
                             │            │
                             ▼            ▼
                    ┌──────────────┐  Keep
                    │  Promote to  │  current
                    │  Production  │  model
                    └──────────────┘
```

---

## 1.8 Biểu đồ mối quan hệ dữ liệu (ERD)

### Tổng quan

Hệ thống sử dụng **6 bảng PostgreSQL**, trong đó bảng `predictions` là bảng chính lưu toàn bộ kết quả dự đoán (21 cột). Bảng `prediction_results` được giữ lại để tương thích ngược (DEPRECATED).

```
┌──────────────────┐       ┌──────────────────────┐       ┌──────────────────────────────┐
│    patients      │       │     admissions        │       │      vital_records           │
├──────────────────┤       ├──────────────────────┤       ├──────────────────────────────┤
│PK patient_id (PK)│1──────│PK admission_id (UUID) │1─────►│PK record_id (UUID)           │
│   name           │       │FK patient_id          │       │FK patient_id                 │
│   age            │       │   admitted_at         │       │   timestamp                  │
│   gender         │       │   discharged_at       │       │   heart_rate                 │
│   ward           │       │   bed_number          │       │   systolic_bp                │
│   created_at     │       │   status              │       │   diastolic_bp               │
└──────────────────┘       └──────────────────────┘       │   temperature                │
                                                          │   spo2                       │
                                                          │   respiratory_rate           │
                                                          │   lactate                    │
                                                          │   wbc                        │
                                                          │   creatinine                 │
                                                          │   bilirubin                  │
                                                          │   platelet                   │
                                                          └──────────────┬───────────────┘
                                                                         │ 1
                                                                         │
                                    ┌────────────────────────────────────┼───────────┐
                                    │                                    │           │
                                    ▼                                    ▼           │
                          ┌──────────────────┐                  ┌──────────────────┐ │
                          │  predictions     │                  │prediction_results│ │
                          │  (BẢNG CHÍNH)    │                  │  (DEPRECATED)   │ │
                          ├──────────────────┤                  ├──────────────────┤ │
                          │PK id (SERIAL)    │                  │PK result_id (UUID)│ │
                          │FK patient_id     │◄─────────────────│FK patient_id     │ │
                          │   timestamp      │                  │   timestamp      │ │
                          │   risk_score     │                  │   risk_score     │ │
                          │   risk_level     │                  │   risk_level     │ │
                          │   alert_triggered│                  │   sofa_score     │ │
                          │   sofa_score     │                  │   news2_score    │ │
                          │   news2_score    │                  │   inference_ms   │ │
                          │   inference_time │                  │   created_at     │ │
                          │------------------│                  └──────────────────┘ │
                          │   (raw vitals)   │                                      │
                          │   heart_rate     │                                      │
                          │   systolic_bp    │                                      │
                          │   diastolic_bp   │                                      │
                          │   temperature    │                                      │
                          │   spo2           │                                      │
                          │   respiratory_rate│                                     │
                          │   lactate        │                                      │
                          │   wbc            │                                      │
                          │   creatinine     │                                      │
                          │   bilirubin      │                                      │
                          │   platelet       │                                      │
                          │------------------│                                      │
                          │   (early warning)│                                      │
                          │   early_warning_ │                                      │
                          │    probability   │                                      │
                          │   early_warning_ │                                      │
                          │    level         │                                      │
                          │   trend_score    │                                      │
                          │   rate_of_change │                                      │
                          │   threshold_score│                                      │
                          │   created_at     │                                      │
                          └──────────────────┘                                      │
                                    │                                               │
                                    │ 1 (có thể có 0..n)                            │
                                    ▼                                               │
                          ┌──────────────────┐                                      │
                          │   alerts         │                                      │
                          ├──────────────────┤                                      │
                          │PK alert_id (UUID)│                                      │
                          │FK patient_id     │◄─────────────────────────────────────┘
                          │   risk_score     │
                          │   risk_level     │
                          │   alert_type     │
                          │   top_features   │  -- JSONB (SHAP features)
                          │   sofa_score     │
                          │   news2_score    │
                          │   created_at     │
                          │   acknowledged   │  -- BOOLEAN DEFAULT FALSE
                          │   ack_by         │  -- VARCHAR(128)
                          │   ack_at         │  -- TIMESTAMPTZ
                          └──────────────────┘
```

### Mối quan hệ chính

| Quan hệ | Kiểu | Giải thích |
|---------|------|------------|
| patients → admissions | 1:N | Một bệnh nhân có nhiều lần nhập viện |
| patients → vital_records | 1:N | Một bệnh nhân có nhiều bản ghi vitals |
| patients → predictions | 1:N | Một bệnh nhân có nhiều kết quả dự đoán |
| patients → alerts | 1:N | Một bệnh nhân có thể có nhiều cảnh báo |

### Chỉ mục (Indexes)

| Bảng | Index | Cột | Loại |
|------|-------|-----|------|
| `vital_records` | `idx_vital_patient_time` | `(patient_id, timestamp DESC)` | B-tree |
| `predictions` | `idx_predictions_patient_time` | `(patient_id, timestamp DESC)` | B-tree |
| `prediction_results` | `idx_pred_patient_time` | `(patient_id, timestamp DESC)` | B-tree |
| `alerts` | `idx_alerts_patient` | `(patient_id, created_at DESC)` | B-tree |
| `alerts` | `idx_alerts_unacked` | `(acknowledged) WHERE acknowledged = FALSE` | Partial |

---

## 1.9 Thiết kế giao diện

### Giao diện 1 — Dashboard chính (danh sách bệnh nhân)

```
┌──────────────────────────────────────────────────────────────────┐
│  🏥 ICU Sepsis Early Warning          [🔔 3 alerts]  [Admin ▼]  │
├──────────────────────────────────────────────────────────────────┤
│  TỔNG QUAN                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │    20    │  │    3     │  │    5     │  │    12    │        │
│  │ Bệnh nhân│  │ CRITICAL │  │ WARNING  │  │  STABLE  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├──────────────────────────────────────────────────────────────────┤
│  DANH SÁCH BỆNH NHÂN                        [🔍 Tìm kiếm...]    │
│  ┌──────┬──────────┬──────┬────────────┬──────────┬──────────┐  │
│  │  ID  │   Tên    │Phòng │ Risk Score │  Mức độ  │  Action  │  │
│  ├──────┼──────────┼──────┼────────────┼──────────┼──────────┤  │
│  │ P001 │ Nguyễn A │ ICU1 │ ████░ 0.82 │🔴CRITICAL│  [Xem]  │  │
│  │ P002 │ Trần B   │ ICU2 │ ███░░ 0.61 │🟡WARNING │  [Xem]  │  │
│  │ P003 │ Lê C     │ ICU1 │ █░░░░ 0.21 │🟢 STABLE │  [Xem]  │  │
│  └──────┴──────────┴──────┴────────────┴──────────┴──────────┘  │
│  Cập nhật lần cuối: 08:35:00  (chu kỳ 5 phút)                   │
└──────────────────────────────────────────────────────────────────┘
```

### Giao diện 2 — Chi tiết bệnh nhân (real-time)

```
┌──────────────────────────────────────────────────────────────────┐
│  ← Quay lại     BN: Nguyễn Văn A — ICU-1 — Giường 3            │
├─────────────────────────┬────────────────────────────────────────┤
│  RISK SCORE HIỆN TẠI    │  VITALS HIỆN TẠI                       │
│                         │  ┌─────────┬──────────┬─────────────┐  │
│       🔴  0.82          │  │HR: 112  │BP: 88/54 │  Temp: 39.1 │  │
│       CRITICAL          │  │SpO2: 93%│RR: 24   │             │  │
│  [Acknowledge Alert]    │  └─────────┴──────────┴─────────────┘  │
│                         │                                        │
│  SOFA: 6  |  NEWS2: 9   │  BIỂU ĐỒ RISK SCORE (2 giờ gần nhất) │
│                         │  1.0│              ╭──╮               │
│  TOP FEATURES (SHAP)    │  0.7│─ ─ ─ ─ ─ ─╯  ╰──             │
│  lactate_trend  ████    │  0.3│                                 │
│  spo2_min_60m   ███     │  0.0└─────────────────────────        │
│  hr_mean_15m    ██      │     -120m   -60m    -30m    now       │
│  resp_trend     ██      │                                        │
│  temp_max_60m   █       │  Cập nhật lần tới: ~5 phút            │
└─────────────────────────┴────────────────────────────────────────┘
```

### Giao diện 3 — Quản lý Alert

```
┌──────────────────────────────────────────────────────────────────┐
│  QUẢN LÝ CẢNH BÁO                   [Tất cả ▼]  [Hôm nay ▼]   │
├──────┬──────────┬───────────┬──────────┬────────────┬──────────┤
│  ID  │ Bệnh nhân│  Thời gian│ Severity │ Trạng thái │  Action  │
├──────┼──────────┼───────────┼──────────┼────────────┼──────────┤
│ A001 │   P001   │ 08:32:00  │ CRITICAL │ ⏳ Pending  │[Confirm]│
│ A002 │   P005   │ 07:45:00  │ WARNING  │ ✅ Confirmed│  [Xem]  │
│ A003 │   P012   │ 06:10:00  │ CRITICAL │ ✅ Confirmed│  [Xem]  │
└──────┴──────────┴───────────┴──────────┴────────────┴──────────┘
```

---

## 1.10 Thiết kế giải thuật

### 1.10.1 Tổng quan

Hệ thống kết hợp **3 giải thuật chính**: (1) Feature Engineering — tính điểm lâm sàng và đặc trưng thống kê từ vitals, (2) XGBoost — dự đoán nguy cơ sepsis T+6h, (3) Rule-based EarlyWarning — cảnh báo 30 phút dựa trên xu hướng. Inference pipeline được tổ chức trong `SepsisPredictor.predict()` tại `services/ml_service/predictor.py`.

### 1.10.2 Giải thuật 1: Clinical Scores (SOFA, NEWS2, qSOFA)

**File:** `feature_engineering/clinical_scores.py`

```
Algorithm: calculate_sofa(pao2_fio2, platelet, bilirubin, map, creatinine)
Input:     pao2_fio2 (mmHg), platelet (K/μL), bilirubin (mg/dL),
           map (mmHg), creatinine (mg/dL)
Output:    SOFA score ∈ [0, 20]

if pao2_fio2 < 100:         resp = 4
elif pao2_fio2 < 200:       resp = 3
elif pao2_fio2 < 300:       resp = 2
elif pao2_fio2 < 400:       resp = 1
else:                       resp = 0

if platelet < 20:            coag = 4
elif platelet < 50:          coag = 3
elif platelet < 100:         coag = 2
elif platelet < 150:         coag = 1
else:                        coag = 0

if bilirubin ≥ 12.0:         liver = 4
elif bilirubin ≥ 6.0:        liver = 3
elif bilirubin ≥ 2.0:        liver = 2
elif bilirubin ≥ 1.2:        liver = 1
else:                        liver = 0

if map < 70:                 cardio = 1    # hoặc có dùng vasopressor
else:                        cardio = 0

if creatinine ≥ 5.0:         renal = 4
elif creatinine ≥ 3.5:       renal = 3
elif creatinine ≥ 2.0:       renal = 2
elif creatinine ≥ 1.2:       renal = 1
else:                        renal = 0

return resp + coag + liver + cardio + renal    # [0, 20]
```

```
Algorithm: calculate_news2(rr, spo2, temperature, systolic_bp, heart_rate)
Input:     rr (/min), spo2 (%), temp (°C), systolic_bp (mmHg), hr (/min)
Output:    NEWS2 score ∈ [0, 20]

RR score:    ≤8→3, 9–11→1, 12–20→0, 21–24→2, ≥25→3
SpO2 score:  ≤91→3, 92–93→2, 94–95→1, ≥96→0
Temp score:  ≤35.0→3, 35.1–36.0→1, 36.1–38.0→0, 38.1–39.0→1, ≥39.1→2
SBP score:   ≤90→3, 91–100→2, 101–110→1, 111–219→0, ≥220→3
HR score:    ≤40→3, 41–50→1, 51–90→0, 91–110→1, 111–130→2, ≥131→3
Consciousness: alert→0, new confusion→3

Return rr_score + spo2_score + temp_score + sbp_score + hr_score + consciousness
```

```
Algorithm: calculate_qsofa(rr, systolic_bp, gcs)
Input:     rr (/min), systolic_bp (mmHg), gcs (3–15)
Output:    qSOFA ∈ [0, 3]

score = 0
if rr ≥ 22:          score += 1
if systolic_bp ≤ 100: score += 1
if gcs < 15:         score += 1
return score
```

### 1.10.3 Giải thuật 2: Feature Engineering (Rolling + Time Features)

**File:** `feature_engineering/vitals_features.py` và `feature_builder.py`

```
Algorithm: add_rolling_features(df, vitals, windows)
Input:     df (DataFrame), vitals = [heart_rate, systolic_bp, ...],
           windows = [3, 12, 48] intervals (~15/60/240 phút)
Output:    df với cột rolling statistics

for each vital in vitals:
    for each w in windows:
        df[f"{vital}_mean_{w}"]  = rolling(df[vital], w).mean()
        df[f"{vital}_std_{w}"]   = rolling(df[vital], w).std()
        df[f"{vital}_min_{w}"]   = rolling(df[vital], w).min()
        df[f"{vital}_max_{w}"]   = rolling(df[vital], w).max()
        df[f"{vital}_trend_{w}"] = rolling(df[vital], w).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x)
        )

# Groupby patient_id để tránh leakage giữa các bệnh nhân
df = df.groupby("patient_id").apply(compute_features)
```

```
Algorithm: FeatureBuilder.build_features(df)
Input:     df với raw vitals + labs
Output:    df với ~85 features

1. add_rolling_features(df, VITAL_COLS, [3, 12, 48])
   → 6 vitals × 4 thống kê × 3 windows = 72 rolling features
2. Tính clinical scores cho mỗi row:
   df["sofa_score"]  = calculate_sofa(...)     # 1 feature
   df["news2_score"] = calculate_news2(...)    # 1 feature
   df["qsofa_score"] = calculate_qsofa(...)    # 1 feature
3. Tính time-since-last-abnormal-HR:
   df["time_since_abnormal_hr"] = time_since(df, "heart_rate", threshold=120)
   → 1 feature
4. Drop raw vitals/labs columns (chỉ giữ features + metadata)
   → ~75 features cuối cùng
```

### 1.10.4 Giải thuật 3: SepsisXGBModel (XGBoost Training)

**File:** `ml/models/xgboost_model.py`

```
Algorithm: SepsisXGBModel.fit(X_train, y_train, X_val, y_val)
Input:     X_train (n_samples × 11 features raw, sau pipeline),
           y_train (0/1 labels T+6h)
Output:    Trained XGBClassifier

Model architecture:
  params = {
      "n_estimators":         150,
      "max_depth":            4,
      "learning_rate":        0.05,
      "subsample":            0.65,
      "colsample_bytree":     0.65,
      "min_child_weight":     20,
      "gamma":                2,
      "reg_alpha":            1,
      "reg_lambda":           3,
      "max_delta_step":       1,
      "scale_pos_weight":     sum(y == 0) / sum(y == 1),  # auto
      "eval_metric":          ["auc", "logloss"],
      "random_state":         42
  }

Training process:
  1. Khởi tạo XGBClassifier với params
  2. fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=30,
         verbose=False)
  3. Chọn best_iteration theo val AUC
  4. Feature importance: gain-based

Threshold tối ưu (trên val set):
  Mặc định: 0.4 (từ bài toán imbalance, ưu tiên Sensitivity)
  val_pred = model.predict_proba(X_val)[:, 1]
  Tìm threshold ∈ [0.2, 0.6] tối đa hoá F1-score
  → threshold ≈ 0.35–0.45 thường đạt Sensitivity > 75%
```

```
Algorithm: cross_validate(X, y, n_splits=5)
Input:     X, y (full training data), n_splits=5
Output:    CV diagnostics dict {mean_auroc, std_auroc, fold_scores}

splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]

    model = SepsisXGBModel(params)
    model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    preds = model.predict_proba(X_fold_val)

    fold_aurocs[fold] = roc_auc_score(y_fold_val, preds)

# Auto-regularize nếu std_auroc > 0.08
if std_auroc > 0.08:
    params["max_depth"] = 3
    params["reg_lambda"] = 5
    # Re-run CV với params đã tighten

return {"mean_auroc": mean(fold_aurocs),
        "std_auroc": std(fold_aurocs),
        "fold_scores": fold_aurocs}
```

### 1.10.5 Giải thuật 4: EarlyWarningPredictor (Rule-based)

**File:** `ml/early_warning.py`

```
Algorithm: EarlyWarningPredictor.predict_early_warning(pid, vitals, ml_score)
Input:     pid (str), vitals (dict {hr, bp, spo2, ...}), ml_score (float)
           history[pid] = deque[dict] maxlen=6 (30 phút)
Output:    {probability, level, trend_score, rate_of_change_score,
             threshold_score, contributing_factors}

1. Update history:
   history[pid].append(vitals)  # lưu 6 bản ghi gần nhất (~30 phút)

2. trend_score (weight=30%):
   if len(history) >= 2:
       hr_trend = (hr_last - hr_first) / hr_first * 100  # % thay đổi
       # Phân loại trend:
       if hr_trend > 15:   trend_score = 1.0    # tăng nhanh
       elif hr_trend > 8:  trend_score = 0.7
       elif hr_trend > 3:  trend_score = 0.4
       else:               trend_score = 0.0
   else: trend_score = 0.0

3. rate_of_change_score (weight=20%):
   # Đạo hàm bậc 1 của các chỉ số chính
   delta_hr   = hr_last - hr_prev
   delta_spo2 = spo2_prev - spo2_last  # giảm SpO2 = nguy cơ
   delta_bp   = bp_prev - bp_last      # tụt huyết áp = nguy cơ
   rate_of_change_score = min(1.0, (abs(delta_hr)*0.02
                                    + abs(delta_spo2)*0.05
                                    + abs(delta_bp)*0.03))

4. threshold_score (weight=50%):
   # Kiểm tra mức độ gần ngưỡng nguy hiểm lâm sàng
   dangers = 0.0
   if hr_last > 120 or hr_last < 40:    dangers += 0.3
   if spo2_last < 90:                   dangers += 0.3
   if temp_last > 39 or temp_last < 35: dangers += 0.2
   if bp_last < 90:                     dangers += 0.2
   threshold_score = min(1.0, dangers)

5. Tính điểm tổng hợp:
   probability = (trend_score * 0.30 +
                  rate_of_change_score * 0.20 +
                  threshold_score * 0.50)
   # Kết hợp với ML score T+6h
   final_probability = max(probability, ml_score)

6. Phân loại level:
   if final_probability ≥ 0.7:  level = "HIGH"
   elif final_probability ≥ 0.4: level = "MEDIUM"
   else:                         level = "LOW"

7. Xác định contributing_factors (các yếu tố nguy cơ cụ thể):
   factors = []
   if hr_last > 110:
       factors.append(f"Nhịp tim cao ({hr_last:.0f} bpm)")
   if spo2_last < 93:
       factors.append(f"SpO2 thấp ({spo2_last:.1f}%)")
   if bp_last < 90:
       factors.append(f"Huyết áp thấp ({bp_last:.0f} mmHg)")
   if lactate is not None and lactate > 2.0:
       factors.append(f"Lactate cao ({lactate:.1f} mmol/L)")
   if hr_trend > 10:
       factors.append(f"Nhịp tim tăng nhanh (+{hr_trend:.0f}% trong 30ph)")

return {probability: final_probability,
        level: level,
        trend_score, rate_of_change_score, threshold_score,
        contributing_factors: factors}
```

### 1.10.6 Giải thuật 5: SHAP Explainability

**File:** `ml/explain.py`

```
Algorithm: SepsisExplainer.explain(X_instance, feature_names)
Input:     X_instance (np.array, shape=(1, n_features)),
           feature_names (list[str])
Output:    list[{feature: str, shap_value: float}]  (top-5)

1. Khởi tạo SHAP TreeExplainer(model)
   - Sử dụng model đã train (xgboost Booster)
   - TreeExplainer tận dụng cấu trúc cây để tính nhanh

2. Tính SHAP values:
   explainer = shap.TreeExplainer(model.booster)
   shap_values = explainer.shap_values(X_instance)
   # shap_values: array shape (1, n_features)

3. Sắp xếp và lọc top-5:
   feature_shap = list(zip(feature_names, shap_values[0]))
   top_features = sorted(feature_shap,
                         key=lambda x: abs(x[1]),
                         reverse=True)[:5]

4. Định dạng kết quả:
   return [{"feature": name, "shap_value": round(val, 4)}
           for name, val in top_features]

# Lưu ý: SHAP chiếm ~50% thời gian inference (50ms / 95ms total)
# Có thể cache kết quả nếu X_instance không đổi (shap_cache)
```

### 1.10.7 Giải thuật 6: Preprocess Pipeline

**File:** `data_pipeline/preprocessor.py` và `services/ml_service/predictor.py`

```
Algorithm: ICUPreprocessor.fit_transform(df, feature_cols)
Input:     df (DataFrame với raw vitals + labs),
           feature_cols = [heart_rate, systolic_bp, ..., platelet] (11 cột)
Output:    X_scaled (np.array), preprocessor (joblib)

Pipeline steps:
  1. forward_fill(limit=2)   # điền thiếu trong 10 phút
  2. KNNImputer(n_neighbors=3, weights="distance")
     # điền labs thiếu dựa trên patients tương tự
  3. IQR clipping → median   # loại outlier
  4. SimpleImputer(strategy="median")
     # điền missing còn sót
  5. StandardScaler()
     # chuẩn hoá: X' = (X - μ) / σ

# Fallback (SepsisPredictor):
# Nếu pipeline chưa được fit (lần đầu chạy),
# fit-on-fly với dữ liệu hiện tại và lưu cache
```

---

## 1.11 Thiết kế Test

### 1.11.1 Tổng quan

Dự án sử dụng **pytest** làm framework kiểm thử chính với **5 file test**, **~70+ test cases**. Cấu hình được định nghĩa trong `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

### 1.11.2 Chiến lược kiểm thử

| Loại test | Công cụ | File | Số lượng | Phạm vi |
|-----------|---------|------|----------|---------|
| **Unit test** | pytest | `tests/unit/` | ~60+ tests | Từng hàm/class riêng lẻ, không phụ thuộc DB |
| **Integration test** | pytest + TestClient | `tests/integration/` | ~10+ tests | Pipeline end-to-end: generator → features |
| **Pytest config** | pytest.ini | — | — | testpaths, verbose, short traceback |
| **CI/CD** | GitHub Actions | `.github/workflows/` | — | Auto-run trên push/PR |

### 1.11.3 Unit Test — `tests/unit/`

**File:** `tests/unit/test_features.py` — Kiểm thử Feature Engineering

```
Test class với fixture `sample_patients` (DataFrame 3 patients × 6 rows)

test_sofa_score_calculation()
  Input: pao2_fio2=250, platelet=200, bilirubin=1.0, map=75, creat=0.8
  Expected: SOFA = 2 (resp=2 + others=0)
  Boundary: pao2_fio2=99 → resp=4, pao2_fio2=100 → resp=3

test_news2_score_calculation()
  Input: rr=22, spo2=94, temp=38.5, sbp=105, hr=95
  Expected: RR=2 + SpO2=1 + Temp=1 + SBP=1 + HR=1 = 6

test_qsofa_score_calculation()
  Input: rr=25, sbp=90, gcs=14 → Expected: qSOFA = 3/3

test_rolling_features()
  Input: DataFrame 6 rows, window=3
  Kiểm tra: cột mean/std/min/max được tạo, row count preserved
```

**File:** `tests/unit/test_labeling.py` — Kiểm thử Label T+6h

```
Fixtures: sample_data (1 patient × 20 rows, onset=10)

test_t6h_label_non_sepsis()
  Input: patient không sepsis → tất cả labels = 0

test_t6h_label_sepsis_positive()
  Input: sepsis onset=10 → rows trong (onset-6, onset] label=1

test_t6h_label_window_correct()
  Row ở t=onset-6 → label=1 ; Row ở t=onset-6.1 → label=0

test_no_future_leakage()
  3 rows đầu → không có label dương giả

test_split_by_patient_no_overlap()
  3 patients → train/val/test không trùng patient_id
  Tỉ lệ sepsis giữa các sets tương đương
```

**File:** `tests/unit/test_model.py` — Kiểm thử SepsisXGBModel

```
Fixtures: trained_model (fit 50 rows), untrained_model (chưa fit)

test_predict_proba_shape()     → output (3, 2)
test_predict_proba_range()     → all ∈ [0, 1]
test_predict_proba_sums_to_one() → sum ≈ 1.0
test_binary_predict()          → output ∈ {0, 1}
test_inference_under_200ms()   → 100 predictions < 200ms
test_save_and_load()           → save/load → predict giống nhau
test_untrained_raises_error()  → XGBoostError khi predict chưa fit
```

**File:** `tests/unit/test_api.py` — Kiểm thử FastAPI Endpoints (18 tests)

```
Session fixture: app (import 1 lần tránh Prometheus Registry error)
Fixtures: client, client_critical (vitals risk ≥ 0.7)

GET /health:
  test_health_returns_200()
  test_health_has_status_field()     → "status": "ok"
  test_health_has_model_version()
  test_health_has_uptime()

POST /vitals — valid:
  test_valid_payload_returns_200()
  test_response_has_risk_score()
  test_risk_score_in_range()         → [0, 1]
  test_risk_level_valid_values()     → LOW/WARNING/CRITICAL
  test_response_has_sofa_news2()
  test_critical_risk_triggers_alert() → alert_triggered = true
  test_low_risk_no_alert()           → alert_triggered = false

POST /vitals — validation errors:
  test_missing_required_field_returns_422()
  test_heart_rate_out_of_range_returns_422()    → HR=300
  test_spo2_below_minimum_returns_422()         → SpO2=20
  test_temperature_out_of_range_returns_422()   → Temp=50
```

### 1.11.4 Integration Test — `tests/integration/`

**File:** `tests/integration/test_pipeline.py` — Kiểm thử end-to-end pipeline

```
Package fixture: pipeline_result (chạy generator → FeatureBuilder)

test_result_not_empty()
test_result_has_rows()
test_sofa_score_column_present()       # cột "sofa_score" tồn tại
test_news2_score_column_present()      # cột "news2_score" tồn tại
test_qsofa_score_column_present()      # cột "qsofa_score" tồn tại
test_rolling_mean_column_present()     # cột "heart_rate_mean_3" tồn tại
```

### 1.11.5 Cách chạy test

```bash
# Toàn bộ test suite (trong Docker)
docker compose exec -T ml_service pytest tests/ -v

# Chỉ unit test
docker compose exec -T ml_service pytest tests/unit/ -v

# Chỉ integration test
docker compose exec -T ml_service pytest tests/integration/ -v

# Test theo file cụ thể
docker compose exec -T ml_service pytest tests/unit/test_api.py -v -k "test_health"

# Với coverage
docker compose exec -T ml_service pytest tests/ -v --cov=.
```

### 1.11.6 CI/CD Test Pipeline (GitHub Actions)

```yaml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=.
      - name: Run integration tests
        run: pytest tests/integration/ -v
      - name: Check coverage >= 70%
        run: coverage report --fail-under=70
      - name: Build Docker images
        run: docker compose build
      - name: Health check
        run: bash scripts/check_health.sh
```

---

# CHƯƠNG 2 — HIỆN THỰC

## 2.1 Công nghệ sử dụng

### 2.1.1 Frontend — Django Dashboard

| Công nghệ | Vai trò | File liên quan |
|-----------|---------|----------------|
| **Django 5.0** | Web framework chính, render template, quản lý session | `web/config/settings.py`, `web/dashboard/views.py` |
| **Bootstrap 5** | CSS framework responsive, cards, badges, tables, navbar | `web/dashboard/templates/dashboard/base.html` |
| **Chart.js** | Biểu đồ risk score 2h, SHAP bar chart (vẽ phía client) | `web/dashboard/templates/dashboard/patient_detail.html` |
| **Django Channels 4.0** | WebSocket real-time: AlertConsumer push alert không cần reload | `web/dashboard/consumers.py`, `web/dashboard/routing.py` |
| **Daphne 4.1** | ASGI server cho Django + WebSocket (HTTP + WS đồng thời) | `web/config/asgi.py` |

### 2.1.2 Dữ liệu

| Công nghệ | Vai trò | File liên quan |
|-----------|---------|----------------|
| **PostgreSQL 15** | Lưu trữ production: patients, predictions, alerts, vital_records | `docs/database_schema.sql` |
| **DuckDB 0.10** | Truy vấn nhanh file Parquet trong training | `data/processed/features_train.parquet` |
| **Pandas 2.2** | Xử lý DataFrame, rolling statistics, groupby operations | `feature_engineering/vitals_features.py` |
| **NumPy 1.26** | Tính toán số học, random generation, mảng đa chiều | — |
| **SQLAlchemy 2.0** | ORM cho FastAPI services (PredictionORM, AlertORM) | `services/ml_service/main.py` |

### 2.1.3 Học máy & MLOps

| Công nghệ | Vai trò | File liên quan |
|-----------|---------|----------------|
| **XGBoost 2.0** | Mô hình chính: predict_proba, early_stopping, scale_pos_weight | `ml/models/xgboost_model.py` |
| **Scikit-learn 1.4** | Pipeline (SimpleImputer + StandardScaler), KNNImputer, StratifiedKFold | `data_pipeline/preprocessor.py`, `ml/train.py` |
| **imbalanced-learn** | SMOTE (ratio=0.4) xử lý imbalance ~9:1 | `ml/train.py` |
| **SHAP 0.45** | TreeExplainer giải thích top-5 features | `ml/explain.py` |
| **Joblib 1.3** | Lưu/tải artifacts: preprocessor pipeline, model | `artifacts/preprocessor_t6h.joblib` |
| **MLflow 2.11** | Experiment tracking + Model Registry | `ml/mlflow_utils.py`, `docker-compose.yml` |

### 2.1.4 Framework & Services

| Công nghệ | Vai trò | Port |
|-----------|---------|------|
| **FastAPI 0.110** | ML Service (inference) + Alert Service | 8001, 8002 |
| **Uvicorn 0.28** | ASGI server cho FastAPI | — |
| **Pydantic v2** | Validation schemas: VitalRequest, PredictionResponse, EarlyWarningResult | `services/ml_service/schemas.py` |
| **httpx 0.27** | Async HTTP client: ML → Alert, Dashboard ↔ services | `services/ml_service/main.py` |
| **Prefect 2.19** | Orchestration: retrain_flow (check_drift → train → promote) | `monitoring/retrain_flow.py` |
| **Evidently AI 0.4** | DataDriftPreset phát hiện data drift | `monitoring/drift_detector.py` |

### 2.1.5 Container & CI/CD

| Công nghệ | Vai trò | File liên quan |
|-----------|---------|----------------|
| **Docker Compose** | 6 services: postgres, mlflow, prometheus, grafana, ml_service, alert_service, web | `docker-compose.yml` |
| **GitHub Actions** | CI/CD: pytest → build → deploy | `.github/workflows/ci.yml` |
| **Prometheus** | Thu thập metrics: predictions_total, inference_seconds, active_alerts | `monitoring/prometheus/prometheus.yml` |
| **Grafana** | Dashboard trực quan hoá metrics hệ thống | `monitoring/grafana/dashboards/icu_dashboard.json` |
| **pytest 8.1** | Test framework (~70+ tests, 5 files) | `pytest.ini` |

---

## 2.2 Kết quả đạt được

### 2.2.1 Chức năng sinh dữ liệu và huấn luyện mô hình

Hệ thống sử dụng `data_pipeline/data_generator.py` để sinh dữ liệu synthetic mô phỏng **20 bệnh nhân ICU trong 24 giờ** với đầy đủ 6 vitals và 5 labs theo chu kỳ 5 phút (5.760 records). Bộ sinh dữ liệu bao gồm **7 confounders** (bad spikes, recovery period, equipment noise, missing labs, age confounder, early normal labs, mild abnormal labs) để mô phỏng độ phức tạp của dữ liệu ICU thực tế.

Pipeline feature engineering (`feature_engineering/feature_builder.py`) tự động tính toán:
- **72 rolling features**: 6 vitals × 4 thống kê (mean, std, min, max) × 3 windows (15/60/240 phút)
- **3 clinical scores**: SOFA (0–20), NEWS2 (0–20), qSOFA (0–3)
- **Time-since-last-abnormal-HR**: 1 feature
- **Tổng cộng**: ~75 features đầu vào cho mô hình

Mô hình **XGBoost** (n_estimators=150, max_depth=4, learning_rate=0.05, scale_pos_weight=auto) được huấn luyện với:
1. Label T+6h (`create_t6h_labels()`) — phát hiện sepsis trước 6 giờ
2. Patient-based split (~60/20/20) — không leakage giữa train/val/test
3. SimpleImputer(median) + StandardScaler
4. Auto SMOTE (ratio=0.4) nếu imbalance > 5
5. 5-fold StratifiedKFold CV + auto-regularize nếu std_auroc > 0.08
6. Early stopping 30 rounds

Toàn bộ thí nghiệm (params, metrics, model artifacts, feature_names.json) được ghi tự động lên **MLflow Tracking Server** tại `http://localhost:5000`. Model đạt **AUROC 0.827** được tự động register lên **Production stage** trong MLflow Model Registry.

*(Chèn ảnh: MLflow experiment list, training metrics, model registry)*

### 2.2.2 Chức năng dự đoán real-time (FastAPI ML Service)

FastAPI ML Service tại `http://localhost:8001` nhận vitals qua `POST /vitals` mỗi 5 phút, thực hiện pipeline 8 bước trong `SepsisPredictor.predict()` (`services/ml_service/predictor.py`):

| Bước | Xử lý | Thời gian | File |
|------|-------|-----------|------|
| 1 | Validate Pydantic (range checks) | ~1ms | `services/ml_service/schemas.py` |
| 2 | Clinical scores SOFA + NEWS2 | ~2ms | `feature_engineering/clinical_scores.py` |
| 3 | Preprocess (Imputer + Scaler) | ~5ms | `artifacts/preprocessor_t6h.joblib` |
| 4 | XGBoost predict_proba | ~35ms | `ml/models/xgboost_model.py` |
| 5 | SHAP TreeExplainer top-5 | ~50ms | `ml/explain.py` |
| 6 | EarlyWarningPredictor rule-based | ~2ms | `ml/early_warning.py` |
| 7 | Lưu PostgreSQL predictions (21 cột) | ~3ms | `services/ml_service/main.py` |
| 8 | Gọi Alert Service nếu risk ≥ 0.7 | — | `services/ml_service/main.py` |

Kết quả trả về gồm: risk_score, risk_level (LOW/WARNING/CRITICAL), alert_triggered, top-5 SHAP features, sofa_score, news2_score, inference_time_ms, early_warning (probability, level, trend_score, rate_of_change_score, threshold_score, contributing_factors).

Các endpoint bổ sung:
- `GET /health` — trạng thái service, model version, model AUROC, uptime
- `GET /vitals/{patient_id}/history` — lịch sử vitals + SHAP + early warning
- `GET /metrics` — Prometheus metrics (predictions_total, predictions_by_risk_total, inference_seconds)

*(Chèn ảnh: Swagger UI /docs, response mẫu với risk score và SHAP features)*

### 2.2.3 Chức năng dashboard và cảnh báo real-time (Django)

Dashboard Django tại `http://localhost:8000` sử dụng **Daphne ASGI** để xử lý đồng thời HTTP + WebSocket:

| Route | View | Chức năng |
|-------|------|-----------|
| `/` | `patient_list` | 4 stats cards (total/critical/warning/stable), bảng ICU sort unconfirmed trước, WebSocket real-time update |
| `/patients/{id}/` | `patient_detail` | Risk score lớn + early warning bars + vitals grid (5 cards) + risk chart (2h) + SHAP bar chart + SOFA/NEWS2 badges + acknowledge button |
| `/alerts/` | `alerts_page` | 3 tabs (all/pending/confirmed), confirm button + toast notification |
| `/alerts/{id}/acknowledge/` | `acknowledge_alert` | PATCH Alert Service → UPDATE Django DB → redirect về dashboard |
| `/api/patient/{id}/latest/` | `patient_latest_api` | JSON endpoint cho polling real-time |
| `/api/alert-count/` | `alert_count_api` | JSON `{"pending": N}` cho navbar badge |
| `ws://:8000/ws/alerts/` | `AlertConsumer` | WebSocket push real-time (group "alerts") |

Khi risk score ≥ 0.7, hệ thống tự động:
1. Gọi Alert Service → upsert pending alert
2. Push WebSocket đến tất cả client → toast + glow animation
3. Prometheus active_alerts Gauge +1

Y tá có thể acknowledge alert trực tiếp từ dashboard, trạng thái cập nhật real-time cho toàn bộ người dùng qua WebSocket.

*(Chèn ảnh: trang danh sách bệnh nhân, trang chi tiết, popup cảnh báo CRITICAL)*

### 2.2.4 Chức năng monitoring và tự động retrain

Hệ thống giám sát được tổ chức qua **3 tầng**:

**Tầng 1 — Prometheus + Grafana (System Metrics)**
- Prometheus scrape targets: `ml_service:8001/metrics`, `alert_service:8002/metrics` (interval 15s)
- Metrics thu thập: `predictions_total` (Counter), `predictions_by_risk_total{risk_level}` (Counter), `inference_seconds` (Histogram), `active_alerts` (Gauge)
- Grafana dashboard pre-configured (`icu_dashboard.json`) hiển thị: prediction throughput, latency distribution, risk level distribution, active alerts
- Health check: `scripts/check_health.sh` kiểm tra 4 services (ML 8001, Alert 8002, Django 8000, MLflow 5000)

**Tầng 2 — Evidently AI (Data Drift Detection)**
- `monitoring/drift_detector.py`: So sánh phân phối dữ liệu reference (`data/processed/features_train.parquet`) vs current 24h (PostgreSQL vital_records)
- Sử dụng `DataDriftPreset` (PSI-based) với threshold drift_score > 0.7
- Lưu HTML report: `reports/drift/drift_YYYYMMDD_*.html`

**Tầng 3 — Prefect (Auto Retrain Pipeline)**
- `monitoring/retrain_flow.py`: Prefect flow gồm 3 tasks:
  1. **check_drift()** → gọi DriftDetector, nếu drift > 0.7 thì trigger retrain
  2. **run_training()** → subprocess `python -m ml.train` với current data
  3. **compare_and_promote()** → new_auroc > production_auroc + 0.01 → promote lên Production, ngược lại giữ model cũ

*(Chèn ảnh: Grafana dashboard metrics, Prefect flow run history, Evidently drift report)*
