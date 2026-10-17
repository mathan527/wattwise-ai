# ⚡ WattWise AI

**Unsupervised machine learning that learns what "normal" electricity use looks like, then flags the hours worth investigating.**

[**Live demo →**](https://wattwise-ai-seven.vercel.app) &nbsp;·&nbsp; Course: Machine Learning, **Unit 3** (clustering, mixtures and EM, PCA, factor analysis)

Most energy dashboards show how much was used. WattWise AI asks a harder question: *is this hour unusual for this meter, at this time of day, on this type of day?* There are no labels. The system learns usage profiles from history, measures how far each reading deviates, and ranks alerts for a facility manager to review. It never claims an alert is "wasted energy"; it flags leads.

---

## What it does

| | |
|---|---|
| **Discovers usage profiles** | K-means, GMM and hierarchical clustering group 7,300 meter-days into 8 daily load shapes |
| **Quantifies uncertainty** | GMM + EM gives soft membership probabilities instead of hard labels |
| **Explains the structure** | PCA (3 components keep 82.5% of variance) and varimax factor analysis with hour-level loadings |
| **Detects anomalies** | Combines three independent signals into one configurable score |
| **Keeps a human in the loop** | Review buttons (genuine / legitimate / unsure), precision tracking, CSV export |
| **Estimates cost only when asked** | Enter a tariff to see the cost of deviation; it is shown as a bound, not as confirmed savings |

## How the anomaly score works

```
score = 0.5 · baseline deviation  +  0.3 · cluster-centroid distance  +  0.2 · GMM deviation
```

1. **Baseline:** robust z-score against the meter's own median for that hour and day type (weekday/weekend).
2. **Centroid:** deviation of the hour from the centroid of the day's K-means profile.
3. **GMM:** deviation from the posterior-weighted mixture mean, so low-confidence days widen the expected range automatically.

Each signal is scaled by its 99th percentile on training data. **The alert threshold comes from the validation period only** (99.5th percentile of Jul–Sep scores), then is applied to the untouched test period (Oct–Dec). Weights and threshold are adjustable live in the app.

## Results (UCI Electricity Load Diagrams 2011–2014, 20 meters, 2014)

| Model | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---|---|---|
| **K-means (K=8)** | **0.344** | **0.973** | **954.6** |
| Hierarchical, Ward (K=8) | 0.331 | 0.989 | 854.3 |
| GMM, EM (12 components, min BIC) | 0.182 | 1.335 | 668.0 |

- **Stability:** K-means is stable across random seeds (ARI ≈ 1.00), less so across time (Jan–Mar vs Apr–Jun, ARI 0.68), because usage profiles shift between seasons.
- **K-means vs GMM agreement:** ARI 0.51.
- **Sensitivity check:** 400 test hours were multiplied by 2.5× as a synthetic stress test. The combined score recovered **95.2%**, versus 73.5% for the baseline signal alone. On unmodified test hours the false-alarm rate was 0.22% (95 of 44,160 hours flagged).

## Honest limitations

- The dataset has **no waste labels**. The 95% recall is a *synthetic* spike test, not proof of real-world detection performance.
- GMM BIC was still falling at 12 components, the largest I tried, so 12 is a search limit, not a true optimum.
- The 26 Oct 01:00 hour can be flagged because of the daylight-saving clock change in the source data.
- Meters are 20 hand-picked clients with complete 2014 data, not a random sample of all 370.
- Alerts are leads. A human has to decide whether the extra load was waste or legitimate demand.

## Syllabus map (Unit 3)

| Topic | Where to see it |
|---|---|
| K-means, elbow, choosing K | K-means tab |
| Mixtures of Gaussians, EM, AIC/BIC | GMM tab (log-likelihood by iteration) |
| PCA, scree plot, latent dimensions | PCA tab |
| Factor analysis, loadings, scores | Factor Analysis tab |
| Dissimilarity, hierarchical clustering | Hierarchical tab (dendrogram, selectable cuts) |
| Evaluating clustering output | Evaluation tab |

## Run it yourself

```bash
pip install -r requirements.txt
# download LD2011_2014.txt from https://archive.ics.uci.edu/dataset/321 into data_raw/uci/
python pipeline/build.py          # trains everything, writes web/data/results.json
python -m http.server -d web      # open http://localhost:8000
```

```
pipeline/build.py   data prep, all models, anomaly scoring, evaluation
web/                static dashboard (Chart.js), deployed on Vercel
```

## Next steps

Real meter ingestion, weather/occupancy covariates, and using reviewers' feedback labels to re-tune the weights.

*Companion project for Unit 4: [HarmonyHMM](https://github.com/mathan527/harmonyhmm) — Hidden Markov Models for music.*
