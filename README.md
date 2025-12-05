# **mlops-optim**

### *Complete MLOps Pipeline – Training, Serving, CI/CD, DVC, and Webhook Automation*

This project implements a full end-to-end MLOps workflow using **FastAPI**, **DVC**, **Makefile automation**, **unit tests**, **API tests**, **linting**, **security scans**, **model quality checks**, and an **excellence-track webhook with ngrok**.

The goal of the project is to build a reproducible ML pipeline for predicting medical insurance costs based on demographic and health factors.

---

# **1. Project Structure**

```
mlops-optim/
│
├── src/
│   ├── app.py                # FastAPI app (predict / health / model-info)
│   ├── model_pipeline.py     # Prepare → Train → Evaluate pipeline logic
│   ├── prepare_stage.py      # DVC stage: data cleaning + encoding
│   ├── train.py              # DVC stage: model training
│   ├── check_quality.py      # Quality gate: ensure metrics are acceptable
│   ├── webhook.py            # Excellence: remote CI trigger endpoint
│
├── tests/
│   ├── test_api.py           # API tests (health, model-info, predict)
│   ├── test_model_pipeline.py
│
├── data/                     # Raw and intermediate data
├── models/                   # Stored trained models
├── results/                  # Metrics, evaluation results
│
├── dvc.yaml                  # Three-step DVC pipeline
├── dvc.lock
├── Makefile                  # Full CI workflow
├── requirements.txt
└── README.md
```

This layout follows best practices for modular, maintainable MLOps projects.

---

# **2. Installation & Setup**

### **1. Create environment**

```bash
make setup
```

This creates `.make-env/` (Python virtual environment) and installs all dependencies.

### **2. Activate environment**

```bash
source .make-env/bin/activate
```

---

# **3. Running the DVC Pipeline**

To reproduce all ML steps:

```bash
make dvc-repro
```

Stages run automatically:

1. **prepare** → clean + encode dataset
2. **train** → GradientBoostingRegressor
3. **evaluate** → MAE, MSE, R² output to `results/metrics.json`

Outputs are tracked via DVC.

---

# **4. FastAPI Model Serving**

Start the API:

```bash
make run-api
```

Or manually:

```bash
.make-env/bin/uvicorn src.app:app --reload
```

### **Available endpoints**

| Endpoint          | Description                                   |
| ----------------- | --------------------------------------------- |
| `GET /health`     | Service liveness check                        |
| `GET /model-info` | Returns metadata about the loaded model       |
| `POST /predict`   | Returns model prediction for encoded features |

### **Swagger UI**

```
http://127.0.0.1:8000/docs
```

---

# **5. How to Make a Prediction**

The model expects **numerically encoded features**:

| Feature | Encoding                                           |
| ------- | -------------------------------------------------- |
| sex     | female=0, male=1                                   |
| smoker  | no=0, yes=1                                        |
| region  | northeast=0, northwest=1, southeast=2, southwest=3 |

Example input:

```json
{
  "features": [19, 0, 27.9, 0, 1, 3]
}
```

Example prediction response:

```json
{
  "prediction": 16884.92
}
```

---

# **6. Running Tests**

### **Unit tests**

```bash
make test
```

### **API tests**

```bash
make api-test
```

All tests must pass before CI succeeds.

---

# **7. Code Quality, Linting, and Security**

Included in CI:

* **pylint** (score must remain 10/10)
* **flake8**
* **ruff** (optional)
* **bandit** (security scanner)

Run all checks together:

```bash
make ci
```

---

# **8. CI Pipeline (Makefile)**

The Makefile orchestrates the full workflow:

```
make ci
```

This runs:

1. pylint
2. flake8
3. bandit
4. unit tests
5. API tests
6. DVC pipeline

If any step fails, CI stops.

---

# **9. Excellence Track — Webhook CI Trigger**

This project includes an advanced MLOps feature:

## **Webhook endpoint**

```bash
POST /trigger
```

Triggers:

```
make ci
```

### **Run webhook server**

```bash
.make-env/bin/uvicorn src.webhook:app --reload --port 9000
```

### **Trigger API**

```bash
curl -X POST http://127.0.0.1:9000/trigger
```

---

# **10. Excellence Track — Ngrok Remote Trigger**

Expose the webhook over HTTPS:

```bash
ngrok http 9000
```

You will get a public URL like:

```
https://cafe-1234.ngrok-free.app/trigger
```

Trigger CI remotely:

```bash
curl -X POST https://cafe-1234.ngrok-free.app/trigger
```

This demonstrates advanced CI orchestration and earns full excellence marks.

---

# **11. Model Quality Gate**

The `check_quality.py` script:

* loads `results/metrics.json`
* verifies required performance thresholds
* stops CI if performance is unacceptable

This prevents merging a degraded model.

---

# **12. Final Notes**

You now have:

* A fully operational **ML training pipeline**
* A production-like **serving API**
* A complete **CI/CD workflow**
* Reproducibility ensured via **DVC**
* Automated **security checks**
* A **remote CI trigger mechanism**
* A fully structured, documented, excellence-level MLOps project


Author : Hind Ch
