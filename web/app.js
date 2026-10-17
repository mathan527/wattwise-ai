/* WattWise AI dashboard: renders precomputed results.json */
const COLORS = ["--c0", "--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const col = i => css(COLORS[i % COLORS.length]);
const $ = id => document.getElementById(id);
const fmt = (x, d = 1) => Number(x).toLocaleString(undefined, { maximumFractionDigits: d });
let D, charts = {};

function mk(id, cfg) {
  if (charts[id]) charts[id].destroy();
  const el = $(id);
  Chart.defaults.color = css("--mut");
  Chart.defaults.borderColor = css("--line");
  cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false }, cfg.options || {});
  charts[id] = new Chart(el, cfg);
}
const card = (title, body, cls = "") => `<div class="card ${cls}"><h3>${title}</h3>${body}</div>`;
const cv = (id, tall) => `<div class="cv ${tall ? "tall" : ""}"><canvas id="${id}"></canvas></div>`;
const hours = [...Array(24).keys()];

function clusterNames() {
  return D.kmeans.centroids.map((c, i) => {
    const mean = c.reduce((a, b) => a + b, 0) / 24;
    const peak = c.indexOf(Math.max(...c));
    const we = D.kmeans.weekend_share[i];
    const lvl = mean < 0.8 ? "Low" : mean < 1.15 ? "Typical" : "High";
    const when = peak >= 6 && peak < 12 ? "morning" : peak >= 12 && peak < 18 ? "afternoon" : peak >= 18 && peak < 23 ? "evening" : "overnight";
    return `P${i + 1}: ${lvl} ${we > 0.6 ? "weekend" : we < 0.18 ? "weekday" : "mixed"} / ${when} peak`;
  });
}

/* ---------- Overview ---------- */
function overview() {
  const m = D.meta, a = D.anomaly;
  $("overview").innerHTML = `
  <h2>Electricity usage intelligence</h2>
  <p class="lead">WattWise AI learns normal consumption behaviour from history, groups similar usage profiles, and flags readings that deviate from what is typical for that meter, hour and day type. Flags are leads to investigate, not proof of waste.</p>
  <div class="kpis">
    <div class="kpi"><div class="v">${m.meters}</div><div class="l">meters analysed</div></div>
    <div class="kpi"><div class="v">${fmt(m.meter_days)}</div><div class="l">meter-days profiled</div></div>
    <div class="kpi"><div class="v">${D.kmeans.k}</div><div class="l">usage profiles (K-means)</div></div>
    <div class="kpi"><div class="v">${(D.pca.cum[2] * 100).toFixed(0)}%</div><div class="l">variance in 3 PCs</div></div>
    <div class="kpi"><div class="v">${fmt(a.n_flagged)}</div><div class="l">test hours flagged</div></div>
    <div class="kpi"><div class="v">${(a.injection.recall * 100).toFixed(0)}%</div><div class="l">synthetic spike recall</div></div>
  </div>
  <div class="grid">
    ${card("Daily consumption by meter (kWh)", `<div style="margin-bottom:8px"><select id="ovMeter">${D.meters.map(x => `<option>${x.id}</option>`).join("")}</select></div>${cv("ovChart")}`, "wide")}
    ${card("Dataset & method", `<p class="note" style="margin-top:0"><b>${m.dataset}</b>, year ${m.year}. ${m.units}. Meters were chosen from clients with complete 2014 data.<br>
    Split by time: train ${m.split.train}, validation ${m.split.validation}, test ${m.split.test}. Alert thresholds come from the validation period only.</p>
    <p class="note">Anomaly score = ${Object.entries(m.weights).map(([k, v]) => `${v} × ${k}`).join(" + ")} (adjustable in the Anomalies tab).</p>`)}
    ${card("Pipeline", `<ol class="note" style="margin:0;padding-left:18px">
      <li>15-min kW readings → hourly kWh per meter</li><li>Meter-day vectors (24 h, scaled by training mean)</li>
      <li>K-means / GMM-EM / hierarchical profile discovery</li><li>PCA &amp; factor analysis for interpretation</li>
      <li>Baseline + centroid + GMM deviation → anomaly score</li><li>Facility-manager review &amp; savings estimate</li></ol>`)}
  </div>`;
  const draw = () => {
    const id = $("ovMeter").value;
    mk("ovChart", { type: "line", data: { labels: D.daily.dates, datasets: [{ label: id, data: D.daily.series[id], borderColor: col(1), pointRadius: 0, borderWidth: 1.5 }] },
      options: { scales: { x: { ticks: { maxTicksLimit: 12 } } }, plugins: { legend: { display: false } } } });
  };
  $("ovMeter").onchange = draw; draw();
}

/* ---------- K-means ---------- */
function profiles() {
  const K = D.kmeans, names = clusterNames();
  $("profiles").innerHTML = `<h2>K-means: usage profiles</h2>
  <p class="lead">Each meter-day is a 24-hour vector (relative to that meter's training average). K was chosen from the elbow and silhouette curves; centroids are the discovered daily load shapes.</p>
  <div class="grid">
    ${card("Elbow plot (inertia) &amp; silhouette", cv("elbow"))}
    ${card("Davies–Bouldin index (lower is better)", cv("dbi"))}
    ${card(`Cluster centroids (K=${K.k})`, cv("cent", true), "wide")}
    ${card("Cluster summary", `<table><tr><th>Profile</th><th class="n">Meter-days</th><th class="n">Weekend share</th><th class="n">Mean level</th></tr>
      ${K.centroids.map((c, i) => `<tr><td><i class="pill" style="background:${col(i)}">&nbsp;</i> ${names[i]}</td><td class="n">${K.sizes[i]}</td><td class="n">${(K.weekend_share[i] * 100).toFixed(0)}%</td><td class="n">${(c.reduce((a, b) => a + b) / 24).toFixed(2)}×</td></tr>`).join("")}</table>
      <p class="note">Names are generated from centroid shape (level, peak hour, weekend share) and should be validated by a person.</p>`, "wide")}
  </div>`;
  const ks = K.elbow.map(e => e.k);
  mk("elbow", { type: "line", data: { labels: ks, datasets: [
    { label: "Inertia", data: K.elbow.map(e => e.inertia), borderColor: col(0), yAxisID: "y" },
    { label: "Silhouette", data: K.elbow.map(e => e.silhouette), borderColor: col(2), yAxisID: "y1" }] },
    options: { scales: { y: { title: { display: true, text: "Inertia" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Silhouette" } }, x: { title: { display: true, text: "K" } } } } });
  mk("dbi", { type: "bar", data: { labels: ks, datasets: [{ label: "Davies–Bouldin", data: K.elbow.map(e => e.davies_bouldin), backgroundColor: col(3) }] }, options: { plugins: { legend: { display: false } } } });
  mk("cent", { type: "line", data: { labels: hours, datasets: K.centroids.map((c, i) => ({ label: names[i], data: c, borderColor: col(i), pointRadius: 0, tension: .25 })) },
    options: { scales: { x: { title: { display: true, text: "Hour of day" } }, y: { title: { display: true, text: "Load ÷ meter average" } } } } });
}

/* ---------- GMM ---------- */
function gmm() {
  const G = D.gmm;
  $("gmm").innerHTML = `<h2>Gaussian Mixture Model with EM</h2>
  <p class="lead">A ${G.k}-component diagonal-covariance GMM fitted with Expectation–Maximisation. Posterior probabilities express how strongly a day belongs to each profile; low confidence alone is not evidence of waste.</p>
  <div class="kpis"><div class="kpi"><div class="v">${G.k}</div><div class="l">components (min BIC)</div></div>
  <div class="kpi"><div class="v">${(G.mean_max_posterior * 100).toFixed(0)}%</div><div class="l">mean top-membership prob.</div></div>
  <div class="kpi"><div class="v">${(G.uncertain_share * 100).toFixed(0)}%</div><div class="l">days with max prob &lt; 0.8</div></div></div>
  <div class="grid">
    ${card("EM log-likelihood by iteration", cv("ll"))}
    ${card("AIC / BIC by number of components", cv("ic"))}
    ${card("Mixture weights", cv("mw"), "wide")}
  </div>`;
  mk("ll", { type: "line", data: { labels: G.ll_curve.map(p => p.iter), datasets: [{ label: "Log-likelihood", data: G.ll_curve.map(p => p.loglik), borderColor: col(1), pointRadius: 0 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "EM iteration" } } } } });
  mk("ic", { type: "line", data: { labels: G.selection.map(s => s.k), datasets: [
    { label: "AIC", data: G.selection.map(s => s.aic), borderColor: col(0) }, { label: "BIC", data: G.selection.map(s => s.bic), borderColor: col(4) }] },
    options: { scales: { x: { title: { display: true, text: "Components" } } } } });
  mk("mw", { type: "bar", data: { labels: G.weights.map((_, i) => "Component " + (i + 1)), datasets: [{ data: G.weights, backgroundColor: G.weights.map((_, i) => col(i)) }] }, options: { plugins: { legend: { display: false } } } });
}

/* ---------- PCA ---------- */
function pca() {
  const P = D.pca;
  $("pca").innerHTML = `<h2>Principal Component Analysis</h2>
  <p class="lead">PCA compresses the 24-dimensional daily profile. The first two components keep ${(P.cum[1] * 100).toFixed(0)}% of the variance; points are coloured by K-means profile.</p>
  <div class="grid">
    ${card("Scree plot &amp; cumulative variance", cv("scree"))}
    ${card("What PC1 and PC2 mean (loadings by hour)", cv("pcl"))}
    ${card("2-D projection of meter-days (sample)", cv("pcs", true) + `<div class="legend">${clusterNames().map((n, i) => `<span><i style="background:${col(i)}"></i>${n}</span>`).join("")}</div>`, "wide")}
  </div>`;
  mk("scree", { data: { labels: P.explained.map((_, i) => "PC" + (i + 1)), datasets: [
    { type: "bar", label: "Explained variance", data: P.explained.map(x => x * 100), backgroundColor: col(0) },
    { type: "line", label: "Cumulative %", data: P.cum.map(x => x * 100), borderColor: col(2) }] }, options: { scales: { y: { title: { display: true, text: "%" } } } } });
  mk("pcl", { type: "line", data: { labels: hours, datasets: P.loadings.map((l, i) => ({ label: "PC" + (i + 1), data: l, borderColor: col(i * 3 + 1), pointRadius: 0 })) }, options: { scales: { x: { title: { display: true, text: "Hour" } } } } });
  const byC = {};
  P.scatter.forEach(p => (byC[p[2]] = byC[p[2]] || []).push({ x: p[0], y: p[1] }));
  mk("pcs", { type: "scatter", data: { datasets: Object.entries(byC).map(([c, pts]) => ({ label: "P" + (+c + 1), data: pts, backgroundColor: col(+c) + "99", pointRadius: 3 })) },
    options: { scales: { x: { title: { display: true, text: `PC1 (${(P.explained[0] * 100).toFixed(0)}%)` } }, y: { title: { display: true, text: `PC2 (${(P.explained[1] * 100).toFixed(0)}%)` } } }, plugins: { legend: { display: false } } } });
}

/* ---------- Factor analysis ---------- */
function factor() {
  const F = D.factor, L = F.loadings;
  const maxAbs = Math.max(...L.flat().map(Math.abs));
  const cell = v => { const t = Math.abs(v) / maxAbs; const c = v >= 0 ? `rgba(11,122,90,${t})` : `rgba(194,65,12,${t})`; return `<div style="background:${c};color:${t > .55 ? "#fff" : "inherit"}">${v.toFixed(2)}</div>`; };
  $("factor").innerHTML = `<h2>Factor Analysis</h2>
  <p class="lead">Three varimax-rotated latent factors summarise the hourly profile. Their meaning comes from the loadings: look at which hours load strongly on each factor rather than assuming a label.</p>
  <div class="grid">
    ${card("Factor loadings (hour × factor)", `<div class="heat" style="grid-template-columns:50px repeat(3,1fr)"><div></div><div>F1</div><div>F2</div><div>F3</div>
      ${L.map((r, h) => `<div style="color:var(--mut)">${String(h).padStart(2, "0")}:00</div>${r.map(cell).join("")}`).join("")}</div>
      <p class="note">Green = positive, orange = negative loading.</p>`)}
    <div>${card("Factor scores: F1 vs F2 (sample)", cv("fs", true))}</div>
  </div>`;
  const byC = {};
  F.scores.forEach(p => (byC[p[2]] = byC[p[2]] || []).push({ x: p[0], y: p[1] }));
  mk("fs", { type: "scatter", data: { datasets: Object.entries(byC).map(([c, pts]) => ({ label: "P" + (+c + 1), data: pts, backgroundColor: col(+c) + "99", pointRadius: 3 })) },
    options: { scales: { x: { title: { display: true, text: "Factor 1" } }, y: { title: { display: true, text: "Factor 2" } } } } });
}

/* ---------- Hierarchical ---------- */
function hier() {
  const H = D.hier;
  $("hier").innerHTML = `<h2>Hierarchical clustering</h2>
  <p class="lead">Meters are compared by their average weekday and weekend load shapes (Euclidean distance, Ward linkage). Choose a cut to see which meters behave alike.</p>
  <div class="card"><div class="ctl"><label>Number of clusters <output id="cutv">3</output><input type="range" id="cut" min="2" max="6" value="3"></label></div>
  <div class="scroll"><div id="dendro"></div></div><div id="cutTable"></div>
  <p class="note">Cophenetic correlation: ${Object.entries(H.cophenetic).map(([k, v]) => `${k} ${v.coph}`).join(" · ")} (closer to 1 = linkage preserves distances better).</p></div>`;
  const draw = () => {
    const k = $("cut").value; $("cutv").textContent = k;
    const lab = H.cuts[k], n = H.names.length;
    const W = Math.max(640, n * 34), Ht = 300, mx = Math.max(...H.dcoord.flat());
    const X = x => 30 + (x - 5) / (10 * n) * (W - 50), Y = y => Ht - 40 - y / mx * (Ht - 70);
    let s = `<svg width="${W}" height="${Ht}" viewBox="0 0 ${W} ${Ht}">`;
    H.icoord.forEach((ic, i) => { const dc = H.dcoord[i]; s += `<polyline fill="none" stroke="${css("--mut")}" stroke-width="1.4" points="${ic.map((x, j) => X(x) + "," + Y(dc[j])).join(" ")}"/>`; });
    H.leaves.forEach((l, i) => { const idx = H.names.indexOf(l); s += `<text x="${X(5 + i * 10)}" y="${Ht - 22}" text-anchor="middle" font-weight="600" style="fill:${col(lab[idx] - 1)}">${l}</text>`; });
    $("dendro").innerHTML = s + "</svg>";
    const groups = {}; H.names.forEach((nm, i) => (groups[lab[i]] = groups[lab[i]] || []).push(nm));
    $("cutTable").innerHTML = `<table><tr><th>Cluster</th><th>Meters</th></tr>${Object.entries(groups).map(([g, m]) => `<tr><td><i class="pill" style="background:${col(g - 1)}">&nbsp;</i> ${g}</td><td>${m.join(", ")}</td></tr>`).join("")}</table>`;
  };
  $("cut").oninput = draw; draw();
}

/* ---------- Evaluation ---------- */
function evaluation() {
  const E = D.evaluation;
  $("eval").innerHTML = `<h2>Model comparison &amp; evaluation</h2>
  <p class="lead">Internal validity metrics on the training period (no ground-truth labels exist). Higher silhouette and Calinski–Harabasz, lower Davies–Bouldin are better.</p>
  <div class="grid">
    ${card("Clustering quality", `<table><tr><th>Model</th><th class="n">K</th><th class="n">Silhouette</th><th class="n">Davies–Bouldin</th><th class="n">Calinski–Harabasz</th></tr>
      ${E.models.map(m => `<tr><td>${m.model}</td><td class="n">${m.k}</td><td class="n">${m.silhouette}</td><td class="n">${m.davies_bouldin}</td><td class="n">${m.calinski_harabasz}</td></tr>`).join("")}</table>`)}
    ${card("Stability", `<table><tr><td>K-means across 5 random seeds (ARI)</td><td class="n">${E.seed_ari.join(", ")}</td></tr>
      <tr><td>Jan–Mar model vs Apr–Jun model (ARI)</td><td class="n">${E.temporal_ari}</td></tr>
      <tr><td>K-means vs GMM agreement (ARI)</td><td class="n">${E.kmeans_vs_gmm_ari}</td></tr></table>
      <p class="note">ARI 1 = identical partitions, 0 = chance. Seed stability is not the same as temporal stability: profiles shift between seasons.</p>`)}
    ${card("Reasoned model selection", `<p style="margin:0">${E.reasoning}</p>`, "wide")}
    ${card("Anomaly signals compared on synthetic spikes", `<p class="note" style="margin-top:0">${D.anomaly.injection.injected} test hours were multiplied by ${D.anomaly.injection.multiplier}× (synthetic, to sanity-check sensitivity; real waste may look different). Each signal uses a threshold set at the same validation quantile.</p>${cv("sig")}
      <p class="note">Combined-score false-alarm rate on unmodified test hours: ${(D.anomaly.injection.false_alarm_rate_clean_test * 100).toFixed(2)}%.</p>`, "wide")}
  </div>`;
  const s = D.anomaly.single_signal_recall;
  mk("sig", { type: "bar", data: { labels: Object.keys(s), datasets: [{ label: "Recall", data: Object.values(s).map(v => v * 100), backgroundColor: Object.keys(s).map((_, i) => col(i)) }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { max: 100, title: { display: true, text: "Recall of injected spikes (%)" } } } } });
}

/* ---------- Anomalies ---------- */
const REV_KEY = "wattwise-reviews-v1";
function loadRev() { try { return JSON.parse(localStorage.getItem(REV_KEY) || "{}"); } catch (e) { return {}; } }
function saveRev(r) { try { localStorage.setItem(REV_KEY, JSON.stringify(r)); } catch (e) { } }
let rev = loadRev();

function anomalies() {
  const A = D.anomaly, W = D.meta.weights;
  $("anomaly").innerHTML = `<h2>Anomaly alerts</h2>
  <div class="warn">An alert means "this hour differs from what is normal for this meter, hour and day type". It does <b>not</b> mean electricity was wasted. Review each alert and record the outcome. The October clock-change hour (26 Oct) can appear because of daylight-saving shifts in the source data.</div>
  <div class="card"><div class="ctl">
    <label>Baseline deviation weight <output id="wbv">${W.baseline}</output><input type="range" id="wb" min="0" max="1" step="0.05" value="${W.baseline}"></label>
    <label>Cluster-centroid weight <output id="wcv">${W.centroid}</output><input type="range" id="wc" min="0" max="1" step="0.05" value="${W.centroid}"></label>
    <label>GMM-membership weight <output id="wgv">${W.gmm}</output><input type="range" id="wg" min="0" max="1" step="0.05" value="${W.gmm}"></label>
    <label>Validation quantile for threshold <output id="qv">${D.meta.val_quantile}</output><input type="range" id="q" min="0.95" max="0.9995" step="0.0005" value="${D.meta.val_quantile}"></label>
    <label>Tariff (₹ per kWh, optional)<input type="number" id="tariff" min="0" step="0.1" placeholder="e.g. 7.5"></label>
    <label>Meter filter<select id="mf"><option value="">All meters</option>${D.meters.map(m => `<option>${m.id}</option>`).join("")}</select></label>
  </div><div id="aSummary"></div></div>
  <div class="grid" style="margin-top:14px">${card("Flagged hours per day (test period)", cv("aDay"))}${card("Review outcomes", `<div id="revSum"></div><p class="note">Reviews are stored in this browser only. Use Export to keep them.</p><button class="pri" id="exp">Export alerts CSV</button>`)}</div>
  <div class="card"><div class="scroll"><table id="aTable"></table></div>
  <p class="note">The table lists the strongest candidates generated at default weights (top ${A.candidates.length}); extreme weight changes may hide alerts outside this pool. Estimated excess = observed − typical (median) kWh for that meter/hour/day type.</p></div>`;
  ["wb", "wc", "wg", "q", "tariff", "mf"].forEach(id => $(id).oninput = renderAnom);
  $("exp").onclick = exportCsv;
  renderAnom();
}
let current = [];
function renderAnom() {
  const A = D.anomaly;
  const w = { baseline: +$("wb").value, centroid: +$("wc").value, gmm: +$("wg").value };
  $("wbv").textContent = w.baseline; $("wcv").textContent = w.centroid; $("wgv").textContent = w.gmm; $("qv").textContent = $("q").value;
  const ws = (w.baseline + w.centroid + w.gmm) || 1;
  const vs = A.val_scores, n = vs.baseline.length;
  const valComb = Array.from({ length: n }, (_, i) => (w.baseline * vs.baseline[i] + w.centroid * vs.centroid[i] + w.gmm * vs.gmm[i]) / ws).sort((a, b) => a - b);
  const thr = valComb[Math.min(n - 1, Math.floor(+$("q").value * n))];
  const tariff = +$("tariff").value, mf = $("mf").value;
  let rows = A.candidates.map(c => ({ ...c, s: (w.baseline * c.rb + w.centroid * c.rc + w.gmm * c.rg) / ws })).filter(c => c.s >= thr && (!mf || c.m === mf)).sort((a, b) => b.s - a.s);
  const top = rows.length ? rows[0].s : 1, span = Math.max(top - thr, 1e-6);
  rows.forEach(r => { const f = (r.s - thr) / span; r.sev = f > .66 ? "High" : f > .33 ? "Medium" : "Low"; r.id = r.m + "|" + r.ts; });
  current = rows;
  const ex = rows.reduce((a, r) => a + r.ex, 0);
  $("aSummary").innerHTML = `<p class="note" style="margin:0"><b>${fmt(rows.length)}</b> alerts above threshold <b>${thr.toFixed(3)}</b> (set on the validation period). Total deviation above typical: <b>${fmt(ex, 0)} kWh</b>${tariff > 0 ? ` ≈ <b>₹${fmt(ex * tariff, 0)}</b> at ₹${tariff}/kWh — an upper-bound of potential cost to investigate, not confirmed savings` : ". Enter a tariff to see a cost estimate."}</p>`;
  const perDay = {}; rows.forEach(r => { const d = r.ts.slice(0, 10); perDay[d] = (perDay[d] || 0) + 1; });
  const days = Object.keys(perDay).sort();
  mk("aDay", { type: "bar", data: { labels: days, datasets: [{ data: days.map(d => perDay[d]), backgroundColor: col(4) }] }, options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxTicksLimit: 10 } } } } });
  const names = clusterNames();
  $("aTable").innerHTML = `<tr><th>Severity</th><th>Meter</th><th>Time</th><th class="n">Score</th><th class="n">kWh</th><th class="n">Typical</th><th class="n">Excess</th><th>Why flagged</th><th>Review</th></tr>` +
    rows.slice(0, 200).map(r => {
      const why = [r.rb >= .5 ? `${r.z > 0 ? "+" : ""}${r.z}σ vs typical hour` : "", r.rc >= .5 ? `${r.cz > 0 ? "+" : ""}${r.cz}σ from ${names[r.cl].split(":")[0]} profile` : "", r.rg >= .5 ? "unlikely under GMM" : ""].filter(Boolean).join("; ") || "combined moderate deviations";
      const st = rev[r.id] || "";
      return `<tr><td><span class="pill sev-${r.sev}">${r.sev}</span></td><td>${r.m}</td><td>${r.ts}</td><td class="n">${r.s.toFixed(3)}</td><td class="n">${fmt(r.kwh)}</td><td class="n">${fmt(r.base)}</td><td class="n">${fmt(r.ex)}</td><td>${why}</td>
      <td>${["Genuine", "Legit", "Unsure"].map(o => `<button class="act ${st === o ? "sel" : ""}" data-id="${r.id}" data-o="${o}">${o}</button>`).join("")}</td></tr>`;
    }).join("");
  $("aTable").querySelectorAll("button.act").forEach(b => b.onclick = () => { rev[b.dataset.id] = rev[b.dataset.id] === b.dataset.o ? "" : b.dataset.o; saveRev(rev); renderAnom(); });
  const cnt = { Genuine: 0, Legit: 0, Unsure: 0 }; Object.values(rev).forEach(v => v && cnt[v]++);
  $("revSum").innerHTML = `<table><tr><td>Genuine issue</td><td class="n">${cnt.Genuine}</td></tr><tr><td>Legitimate demand change</td><td class="n">${cnt.Legit}</td></tr><tr><td>Unsure</td><td class="n">${cnt.Unsure}</td></tr></table>
  <p class="note">${cnt.Genuine + cnt.Legit ? `Precision on reviewed alerts: ${(cnt.Genuine / (cnt.Genuine + cnt.Legit) * 100).toFixed(0)}%. ` : ""}Use these labels to re-tune weights and thresholds.</p>`;
}
function exportCsv() {
  const head = "severity,meter,time,score,kwh,typical_kwh,excess_kwh,review\n";
  const body = current.map(r => [r.sev, r.m, r.ts, r.s.toFixed(4), r.kwh, r.base, r.ex, rev[r.id] || ""].join(",")).join("\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([head + body], { type: "text/csv" })); a.download = "wattwise_alerts.csv"; a.click();
}

/* ---------- boot ---------- */
const renderers = { overview, profiles, gmm, pca, factor, hier, eval: evaluation, anomaly: anomalies };
const done = {};
function show(tab) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.id === tab));
  document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("on", b.dataset.tab === tab));
  if (!done[tab]) { renderers[tab](); done[tab] = true; }
  history.replaceState(null, "", "#" + tab);
}
fetch("data/results.json").then(r => r.json()).then(d => {
  D = d;
  document.querySelectorAll("#nav button").forEach(b => b.onclick = () => show(b.dataset.tab));
  const h = location.hash.slice(1);
  show(renderers[h] ? h : "overview");
}).catch(e => { document.querySelector("main").innerHTML = "<p>Could not load results.json: " + e + "</p>"; });
