# WattWise AI

Unsupervised ML for electricity-usage analysis and anomaly detection (ML Unit 3).

**Dataset:** UCI Electricity Load Diagrams 2011-2014 (20 meters, year 2014, hourly kWh).
Train Jan-Jun, validation Jul-Sep (thresholds), test Oct-Dec.

| Method | Where |
|---|---|
| K-means (elbow, silhouette, Davies-Bouldin) | `pipeline/build.py`, K-means tab |
| GMM + EM (log-likelihood curve, AIC/BIC, posteriors) | GMM tab |
| PCA (scree, 2-D projection) | PCA tab |
| Factor Analysis (varimax loadings, scores) | Factor tab |
| Hierarchical clustering (Ward dendrogram, cuts) | Hierarchical tab |
| Evaluation (model comparison, seed/temporal stability ARI) | Evaluation tab |
| Anomaly score = baseline deviation + centroid distance + GMM deviation | Anomalies tab |

Alerts are leads for investigation, not proof of wasted energy. The injected-spike recall is a
synthetic sanity check; the dataset has no waste labels.

## Run
```
pip install -r requirements.txt
# download LD2011_2014.txt from https://archive.ics.uci.edu/dataset/321 into data_raw/uci/
python pipeline/build.py        # writes web/data/results.json
python -m http.server -d web    # open http://localhost:8000
```
The `web/` folder is a static site (deployed on Vercel with `vercel.json`).
