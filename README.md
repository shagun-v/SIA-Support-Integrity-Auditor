
# Support Integrity Auditor (SIA)
## MARS Open Projects 2026 — AI/ML Track

### Overview
SIA detects **Priority Mismatch** in customer support tickets — cases where the
human-assigned priority conflicts with the ticket's inferred objective severity.
It bootstraps its own supervision signal from raw ticket data (no mismatch labels needed).

## Pretrained Models & Outputs

Due to GitHub file size limitations, trained model artifacts and generated outputs are hosted on Google Drive.

### Download Link
https://drive.google.com/drive/folders/1au3rdQT0_NTP1a3i7K1siUj3SR7qcCgh?usp=sharing

Contents:
- model_artifacts/
- mismatch_dossiers.json
- adversarial_results.csv
- evaluation_metrics.csv

After downloading, place the folders/files in the project root directory before running the Streamlit app.

### Architecture
See `ARCHITECTURE.md` or the Mermaid diagram in `notebook.ipynb`.

### Signal Fusion Strategy
| Signal               | Weight | Rationale                                        |
|----------------------|--------|--------------------------------------------------|
| Rule-Based NLP       | 0.30   | Direct severity vocabulary match                 |
| Resolution Time      | 0.25   | Empirical complexity proxy (quantile-normalised) |
| Semantic Embeddings  | 0.20   | Latent urgency beyond surface keywords           |
| Zero-Shot LLM        | 0.25   | Broad language understanding (BART-MNLI)         |

### Ablation Results
*(Fill in from Cell 13 output after training)*

| Signal          | Mismatch Rate (%) | Cohen's Kappa vs Ensemble |
|-----------------|-------------------|---------------------------|
| NLP Only        |                   |                           |
| Resolution Only |                   |                           |
| Embedding Only  |                   |                           |
| ZeroShot Only   |                   |                           |
| **Ensemble**    | **—**             | **1.000**                 |

### Evaluation Results
*(Fill in from Cell 15 / 16 output)*

| Metric              | XGBoost Baseline | DeBERTa Fine-Tuned |
|---------------------|------------------|--------------------|
| Accuracy            |                  |                    |
| Macro F1            |                  |                    |
| Recall (Consistent) |                  |                    |
| Recall (Mismatch)   |                  |                    |
| ROC AUC             |                  |                    |

### Verification Thresholds (from Competition Brief)
- Binary Accuracy ≥ 83%
- Macro F1 ≥ 0.82
- Per-Class Recall ≥ 0.78 (both classes)

### Adversarial Robustness
Score: X/10 tickets correctly classified.

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Reproducibility
```bash
pip install -r requirements.txt
python train_pipeline.py          # regenerate pseudo-labels + train
python predict.py --input tickets.csv --output predictions.json
```
