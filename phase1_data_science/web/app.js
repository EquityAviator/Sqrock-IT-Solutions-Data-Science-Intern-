/* ============================================================
   Sqrock IT Solutions - Data Science Internship - Phase 1
   Dashboard application logic
   ============================================================ */
(function () {
  "use strict";
  const D = window.PHASE1_DATA;
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const fmt = (n, d = 0) => Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const C = { blue: "#2563eb", amber: "#f59e0b", green: "#10b981", red: "#ef4444",
              purple: "#8b5cf6", slate: "#64748b", grid: "#eef1f6", ink: "#0f172a" };

  if (window.Chart) {
    Chart.defaults.font.family = '-apple-system, "Segoe UI", Roboto, sans-serif';
    Chart.defaults.font.size = 11.5;
    Chart.defaults.color = "#475569";
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.animation.duration = 550;
  }
  const axis = (extra = {}) => Object.assign({
    grid: { color: C.grid, drawBorder: false }, ticks: { maxTicksLimit: 8 }, border: { display: false }
  }, extra);

  /* ---------------- navigation ---------------- */
  const rendered = {};
  function go(page) {
    $$(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + page));
    $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (!rendered[page]) { rendered[page] = true; (RENDER[page] || (() => {}))(); }
  }
  $$("#nav a").forEach((a) => a.addEventListener("click", () => go(a.dataset.page)));

  /* ---------------- helpers ---------------- */
  function histogram(arr, bins, lo, hi) {
    lo = lo === undefined ? Math.min(...arr) : lo;
    hi = hi === undefined ? Math.max(...arr) : hi;
    const w = (hi - lo) / bins, counts = new Array(bins).fill(0), labels = [];
    for (let i = 0; i < bins; i++) labels.push((lo + w * i).toFixed(0));
    arr.forEach((v) => { let i = Math.floor((v - lo) / w); if (i >= bins) i = bins - 1; if (i < 0) i = 0; counts[i]++; });
    return { labels, counts };
  }
  function hbar(id, labels, values, color, opts = {}) {
    return new Chart($("#" + id), {
      type: "bar",
      data: { labels, datasets: [{ data: values, backgroundColor: color, borderRadius: 4, barPercentage: 0.82 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => " " + fmt(c.raw, opts.dp || 2) } } },
        scales: { x: axis({ beginAtZero: true }), y: axis({ grid: { display: false }, ticks: { maxTicksLimit: 30, autoSkip: false } }) } }
    });
  }
  function vbar(id, labels, values, colors, opts = {}) {
    return new Chart($("#" + id), {
      type: "bar",
      data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 5, barPercentage: 0.72 }] },
      options: { plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => " " + fmt(c.raw, opts.dp || 0) + (opts.suffix || "") } } },
        scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, ...(opts.y || {}) }) } }
    });
  }
  const line = (id, datasets, opts = {}) => new Chart($("#" + id), {
    type: "line",
    data: { labels: opts.labels, datasets },
    options: { interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: datasets.length > 1, position: "top" } },
      elements: { point: { radius: 0, hoverRadius: 4 } },
      scales: { x: axis({ ticks: { maxTicksLimit: opts.maxTicks || 8, autoSkip: true } }),
                y: axis({ beginAtZero: opts.zero !== false, ...(opts.y || {}) }) } }
  });

  /* ================= OVERVIEW ================= */
  function renderOverview() {
    const cards = [
      { id: "task1", t: "Task 1 · Sales Forecasting", tag: "Time Series", tagc: "",
        desc: "Forecast future sales from 3M rows of real retail transactions using regression + ARIMA.",
        stat: "MAE " + fmt(D.t1.metrics.models[D.t1.metrics.best_model].MAE), statl: "Best model · " + D.t1.metrics.best_model },
      { id: "task2", t: "Task 2 · Student Performance", tag: "Regression", tagc: "green",
        desc: "Predict a student's final marks from study hours, attendance and prior grades.",
        stat: "R² " + D.t2.metrics.models[D.t2.metrics.best_model].R2.toFixed(3), statl: "Best model · " + D.t2.metrics.best_model },
      { id: "task3", t: "Task 3 · Resume Screening", tag: "NLP", tagc: "amber",
        desc: "Rank and shortlist resumes against a job description with TF-IDF + cosine similarity.",
        stat: D.t3.metrics.resumes_screened + " ranked", statl: D.t3.metrics.shortlist_size + " shortlisted" },
      { id: "task4", t: "Task 4 · Customer Churn", tag: "Classification", tagc: "red",
        desc: "Predict which customers will churn from tenure, charges, contract and service usage.",
        stat: "F1 " + D.t4.metrics.models[D.t4.metrics.best_model].F1.toFixed(3), statl: "Best model · " + D.t4.metrics.best_model }
    ];
    $("#taskCards").innerHTML = cards.map((c) => `
      <div class="card" style="cursor:pointer" data-goto="${c.id}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span class="pill ${c.tagc}">${c.tag}</span><span class="pill green">Complete</span>
        </div>
        <h4 style="font-size:15px; margin-bottom:5px;">${c.t}</h4>
        <p class="note" style="font-size:12.8px; min-height:52px;">${c.desc}</p>
        <div style="display:flex; justify-content:space-between; align-items:baseline; border-top:1px solid var(--line); padding-top:10px;">
          <span class="note">${c.statl}</span><b style="color:var(--primary); font-size:15px;">${c.stat}</b>
        </div>
      </div>`).join("");
    $$("#taskCards [data-goto]").forEach((el) => el.addEventListener("click", () => go(el.dataset.goto)));

    const stages = [
      ["1. Data Collection", "CSV from Hugging Face", "UCI repository", "HF dataset + PDF parse", "IBM Telco dataset"],
      ["2. Cleaning / Preprocessing", "Date parse, impute, de-dup", "Encode, scale, split", "Lowercase, stopwords, lemma", "Coerce, encode, scale, split"],
      ["3. EDA", "Trend, months, seasonality", "Correlation, scatter, histogram", "Score distribution", "Churn dist, charges, tenure, heatmap"],
      ["4. Feature Engineering", "Month, Year, Day, lags", "Derived attendance & scores", "TF-IDF / BoW", "Avg spend, contract group, drop ID"],
      ["5. Modelling", "LinReg, DecTree, RF, ARIMA", "LinReg, DecTree, RF", "Cosine similarity + keywords", "LogReg, DecTree, RF, KNN"],
      ["6. Prediction", "30-day forward forecast", "Predicted vs actual marks", "Ranked candidate list", "Churn probability per customer"],
      ["7. Evaluation", "MAE, RMSE", "MAE, MSE, R²", "Match %, shortlist precision", "Accuracy, Precision, Recall, F1, CM"]
    ];
    $("#pipelineMap").innerHTML = stages.map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td></tr>`).join("");

    const ds = [
      ["T1", "Store Sales (Favorita grocery)", "Hugging Face (Kaggle mirror)", "3,000,888 rows", "Real 4.5-year multi-store retail log — genuine trend &amp; seasonality."],
      ["T2", "Student Performance (Cortez &amp; Silva)", "UCI ML Repository", "1,044 students", "Canonical peer-reviewed dataset with every required factor."],
      ["T3", "Resume Screening Dataset", "Hugging Face (AzharAli05)", "10,174 resumes", "Full CV text with names, 45 roles and ground-truth decisions."],
      ["T4", "Telco Customer Churn", "IBM sample dataset", "7,043 customers", "Industry-standard churn benchmark with real class imbalance."]
    ];
    $("#datasetTable").innerHTML = ds.map((r) => `<tr><td><span class="pill grey">${r[0]}</span></td><td><b>${r[1]}</b></td><td>${r[2]}</td><td>${r[3]}</td><td class="note">${r[4]}</td></tr>`).join("");
  }

  /* ================= TASK 1 ================= */
  function renderTask1() {
    const m = D.t1.metrics, t = D.t1;
    $("#t1-raw").textContent = fmt(m.records_raw);
    $("#t1-daily").textContent = fmt(m.daily_points) + " days";
    $("#t1-range").textContent = m.date_range[0] + " → " + m.date_range[1];
    $("#t1-fctotal").textContent = "$" + fmt(m.forecast_next_30_days_total);
    $("#t1-fcmean").textContent = "$" + fmt(m.forecast_next_30_days_mean);
    $("#t1-best").textContent = m.best_model;

    $("#t1-steps").innerHTML = [
      ["Data Collection", "Loaded 3,000,888 transaction rows (date, store, family, sales, onpromotion)."],
      ["Data Cleaning", "Parsed dates, removed duplicates, re-indexed on a complete calendar and time-interpolated 4 missing days."],
      ["EDA", "Trend line, monthly totals (best/worst month), calendar-month seasonality and day-of-week pattern."],
      ["Feature Engineering", "Year, Month, Day, day-of-week, day-of-year, week-of-year, weekend flag + lag 1/7/14/30 and rolling 7/30 means."],
      ["Model Building", "Linear Regression, Decision Tree, Random Forest and ARIMA(7,1,1)."],
      ["Prediction", "Time-based hold-out of the last 90 days; recursive 30-day forward forecast."],
      ["Evaluation", "MAE and RMSE — best model selected by lowest RMSE."]
    ].map((s) => `<div class="step"><div class="n"></div><div class="t"><b>${s[0]}</b><span>${s[1]}</span></div></div>`).join("");

    // trend
    const ds = t.trend.dates.map((d, i) => ({ x: d, y: t.trend.sales[i] }));
    line("c1-trend", [{ label: "Daily sales", data: ds, borderColor: C.blue, borderWidth: 1.1, fill: true,
      backgroundColor: "rgba(37,99,235,.07)", tension: .15 }],
      { maxTicks: 7, y: { ticks: { callback: (v) => (v / 1e6).toFixed(0) + "M" } } });
    // monthly
    vbar("c1-monthly", t.monthly.labels, t.monthly.values,
      t.monthly.values.map((v) => v === Math.max(...t.monthly.values) ? C.red : v === Math.min(...t.monthly.values) ? C.green : C.blue),
      { y: { ticks: { callback: (v) => (v / 1e6).toFixed(0) + "M" } } });
    // seasonality
    vbar("c1-season", t.seasonality.labels.map((x) => "M" + x), t.seasonality.values, C.purple,
      { y: { ticks: { callback: (v) => (v / 1e6).toFixed(1) + "M" } } });
    // dow
    vbar("c1-dow", t.dow.labels, t.dow.values, C.amber, { y: { ticks: { callback: (v) => (v / 1e6).toFixed(1) + "M" } } });

    // metrics table
    const names = Object.keys(m.models);
    $("#t1-models").innerHTML = names.map((n) => {
      const best = n === m.best_model;
      return `<tr${best ? ' style="background:#ecfaf4"' : ""}><td><b>${n}</b>${best ? ' <span class="pill green">best</span>' : ""}</td>
        <td class="num">${fmt(m.models[n].MAE)}</td><td class="num">${fmt(m.models[n].RMSE)}</td></tr>`;
    }).join("");
    new Chart($("#c1-metrics"), {
      type: "bar",
      data: { labels: names, datasets: [
        { label: "MAE", data: names.map((n) => m.models[n].MAE), backgroundColor: C.blue, borderRadius: 4 },
        { label: "RMSE", data: names.map((n) => m.models[n].RMSE), backgroundColor: C.amber, borderRadius: 4 }] },
      options: { plugins: { legend: { position: "top" } },
        scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, ticks: { callback: (v) => (v / 1e3).toFixed(0) + "k" } }) } }
    });

    // actual vs predicted
    line("c1-avp", [
      { label: "Actual", data: t.avp.actual, borderColor: C.ink, borderWidth: 1.8, tension: .15 },
      { label: "Predicted (" + m.best_model + ")", data: t.avp.predicted, borderColor: C.red, borderWidth: 1.8, borderDash: [5, 4], tension: .15 }
    ], { labels: t.avp.dates, y: { ticks: { callback: (v) => (v / 1e6).toFixed(1) + "M" } } });

    // forecast
    const hx = t.forecast.history_dates.map((d, i) => ({ x: d, y: t.forecast.history_values[i] }));
    const fx = t.forecast.dates.map((d, i) => ({ x: d, y: t.forecast.values[i] }));
    line("c1-forecast", [
      { label: "History (last 120 days)", data: hx, borderColor: C.ink, borderWidth: 1.6, tension: .15 },
      { label: "30-day forecast", data: fx, borderColor: C.green, borderWidth: 2.2, tension: .2,
        pointRadius: 2.5, pointBackgroundColor: C.green, fill: true, backgroundColor: "rgba(16,185,129,.08)" }
    ], { maxTicks: 8, y: { ticks: { callback: (v) => (v / 1e6).toFixed(1) + "M" } } });
  }

  /* ================= TASK 2 ================= */
  function renderTask2() {
    const m = D.t2.metrics, t = D.t2;
    $("#t2-students").textContent = fmt(m.students);
    $("#t2-mean").textContent = m.mean_final_marks;
    $("#t2-steps").innerHTML = [
      ["Data Collection", "Combined the UCI Math (395) and Portuguese (649) cohorts — 1,044 students."],
      ["Preprocessing", "No missing values; mapped studytime→hours, absences→attendance %, G1/G2/G3 to the required schema; encoded 18 categorical variables; standard-scaled; 80/20 split."],
      ["EDA", "Correlation heatmap, study-hours vs marks scatter, attendance bands, score distribution and per-feature correlation."],
      ["Model Building", "Linear Regression, Decision Tree (depth 6) and Random Forest (400 trees)."],
      ["Train & Test", "Trained on 835 students, tested on 209 unseen students."],
      ["Evaluation", "MAE, MSE and R² — best model by R²."]
    ].map((s) => `<div class="step"><div class="n"></div><div class="t"><b>${s[0]}</b><span>${s[1]}</span></div></div>`).join("");

    // study vs marks scatter
    const pts = t.study_vs_marks.x.map((x, i) => ({ x, y: t.study_vs_marks.y[i] }));
    const xs = [...new Set(t.study_vs_marks.x)].sort((a, b) => a - b);
    const n = t.study_vs_marks.x.length;
    const mx = t.study_vs_marks.x.reduce((a, b) => a + b, 0) / n, my = t.study_vs_marks.y.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += (t.study_vs_marks.x[i] - mx) * (t.study_vs_marks.y[i] - my); den += (t.study_vs_marks.x[i] - mx) ** 2; }
    const slope = num / den, icpt = my - slope * mx;
    const trend = xs.map((x) => ({ x, y: slope * x + icpt }));
    new Chart($("#c2-study"), {
      type: "scatter",
      data: { datasets: [
        { label: "Students", data: pts, backgroundColor: "rgba(37,99,235,.35)", pointRadius: 3 },
        { label: "Trend (r=" + t.study_vs_marks.r + ")", data: trend, type: "line", borderColor: C.red, borderWidth: 2, pointRadius: 0, fill: false }] },
      options: { plugins: { legend: { position: "top" } },
        scales: { x: axis({ title: { display: true, text: "Study hours / week" } }),
                  y: axis({ title: { display: true, text: "Final marks (0–20)" } }) } }
    });

    vbar("c2-att", t.attendance.labels.map((l) => l + "%"), t.attendance.values, C.green, { dp: 1, y: { beginAtZero: true } });
    const dh = histogram(t.dist.values, 21, 0, 21);
    vbar("c2-dist", dh.labels, dh.counts, C.purple, { y: { beginAtZero: true } });
    hbar("c2-featcorr", t.featcorr.labels, t.featcorr.values,
      t.featcorr.values.map((v) => v < 0 ? C.red : C.blue), { dp: 3 });
    hbar("c2-imp", t.importance.labels, t.importance.values, C.amber, { dp: 3 });

    const names = Object.keys(m.models);
    $("#t2-models").innerHTML = names.map((n) => {
      const best = n === m.best_model;
      return `<tr${best ? ' style="background:#ecfaf4"' : ""}><td><b>${n}</b>${best ? ' <span class="pill green">best</span>' : ""}</td>
        <td class="num">${m.models[n].MAE}</td><td class="num">${m.models[n].MSE}</td><td class="num">${m.models[n].R2.toFixed(3)}</td></tr>`;
    }).join("");

    // interactive predictor
    const sm = t.simple_model;
    $("#t2-r2").textContent = sm.r2.toFixed(3);
    $("#t2-mae").textContent = sm.mae;
    function predict() {
      const v = ["study", "att", "prev", "asg"].map((k) => +$("#s2-" + k).value);
      ["study", "att", "prev", "asg"].forEach((k, i) => $("#v2-" + k).textContent = v[i]);
      let y = sm.intercept; v.forEach((x, i) => y += sm.coef[i] * x);
      y = Math.max(0, Math.min(20, y));
      $("#t2-pred").textContent = y.toFixed(1);
      const pct = (y / 20) * 100;
      const bar = $("#t2-meter");
      bar.style.width = pct + "%";
      bar.style.background = y >= 14 ? C.green : y >= 9 ? C.amber : C.red;
      $("#t2-pred").style.color = y >= 14 ? C.green : y >= 9 ? C.amber : C.red;
      $("#t2-band").textContent = y >= 14 ? "Strong performance — likely to pass comfortably"
        : y >= 9 ? "Average performance — monitor for extra support"
        : "At risk — recommend targeted academic support";
    }
    ["study", "att", "prev", "asg"].forEach((k) => $("#s2-" + k).addEventListener("input", predict));
    predict();
  }

  /* ================= TASK 3 ================= */
  function renderTask3() {
    const m = D.t3.metrics, t = D.t3;
    $("#t3-role").textContent = m.target_role;
    $("#t3-count").textContent = fmt(m.resumes_screened) + " resumes";
    $("#t3-jd").textContent = t.jd_preview;
    $("#t3-short").textContent = m.shortlist_size;
    $("#t3-meanscore").textContent = m.mean_score + "%";
    $("#t3-topscore").textContent = m.top_candidate.score + "%";
    $("#t3-matrix").textContent = fmt(m.resumes_screened) + " × 3,000";
    $("#t3-topname").textContent = m.top_candidate.name;

    $("#t3-steps").innerHTML = [
      ["Data Collection", "Loaded 10,174 resumes across 45 roles; filtered to Data Scientist (825 unique). 3 resumes rendered to PDF and re-parsed with pypdf to demonstrate PDF→text."],
      ["NLP Preprocessing", "Lowercase → strip URLs/emails/punctuation → tokenize → remove stopwords → WordNet lemmatisation."],
      ["Feature Extraction", "TF-IDF vectoriser (1–2 grams, 3,000 features, sublinear TF); Bag-of-Words computed for comparison."],
      ["Matching Algorithm", "Cosine similarity between each resume vector and the JD vector (65%) blended with skill/keyword coverage (35%)."],
      ["Ranking System", "Sorted by blended score, scaled to 0–100%, top 10 shortlisted above a 30% cut-off."],
      ["Output", "Candidate name, match %, full ranking list and shortlist exported to CSV."]
    ].map((s) => `<div class="step"><div class="n"></div><div class="t"><b>${s[0]}</b><span>${s[1]}</span></div></div>`).join("");

    hbar("c3-top", t.top.names.slice().reverse(), t.top.scores.slice().reverse(),
      t.top.shortlisted.slice().reverse().map((s) => s === "YES" ? C.green : C.slate), { dp: 1 });
    const h = histogram(t.dist, 25);
    vbar("c3-dist", h.labels, h.counts, C.blue, { y: { beginAtZero: true } });

    // skill chips
    $("#t3-skills").innerHTML = t.skills_all.map((s) =>
      `<span class="tag ${t.top1_skills.includes(s) ? "on" : ""}">${t.top1_skills.includes(s) ? "✓ " : ""}${s}</span>`).join("");

    // ranking table with search + filter
    function drawTable() {
      const q = $("#t3-search").value.toLowerCase().trim();
      const f = $("#t3-filter").value;
      const rows = t.ranking.filter((r) => (!q || r.name.toLowerCase().includes(q)) &&
        (f === "all" || r.shortlisted === "YES"));
      $("#t3-table").innerHTML = rows.map((r) => `
        <tr><td class="num ${r.rank === 1 ? "rank-1" : ""}">${r.rank}</td>
          <td><b>${r.name}</b></td>
          <td class="num">${r.match_pct}%</td>
          <td class="num">${r.skill_count}</td>
          <td>${r.shortlisted === "YES" ? '<span class="pill green">Shortlisted</span>' : '<span class="pill grey">Reviewed</span>'}</td></tr>`).join("")
        || '<tr><td colspan="5" class="note" style="padding:18px;">No candidates match your search.</td></tr>';
    }
    $("#t3-search").addEventListener("input", drawTable);
    $("#t3-filter").addEventListener("change", drawTable);
    drawTable();
  }

  /* ================= TASK 4 ================= */
  function renderTask4() {
    const m = D.t4.metrics, t = D.t4;
    $("#t4-cust").textContent = fmt(m.customers);
    $("#t4-rate").textContent = m.churn_rate_pct + "%";
    $("#t4-feats").textContent = m.features.length + " features";
    $("#t4-steps").innerHTML = [
      ["Data Collection", "Loaded 7,043 IBM Telco customers with 21 attributes."],
      ["Preprocessing", "Coerced TotalCharges (11 blanks imputed), removed duplicates, label-encoded 16 categorical columns, standard-scaled, stratified 80/20 split."],
      ["EDA", "Churn distribution (donut + counts), monthly charges vs churn, tenure bands, correlation heatmap."],
      ["Feature Engineering", "avg_monthly_spend, contract_group, tenure_bucket, num_services; dropped CustomerID."],
      ["Model Building", "Logistic Regression, Decision Tree, Random Forest and KNN (class-balanced)."],
      ["Evaluation", "Accuracy, Precision, Recall, F1, confusion matrix and ROC-AUC."]
    ].map((s) => `<div class="step"><div class="n"></div><div class="t"><b>${s[0]}</b><span>${s[1]}</span></div></div>`).join("");

    // churn distribution doughnut
    new Chart($("#c4-dist"), {
      type: "doughnut",
      data: { labels: ["Stayed", "Churned"], datasets: [{ data: [t.churn_dist.stayed, t.churn_dist.churned],
        backgroundColor: [C.green, C.red], borderColor: "#fff", borderWidth: 2 }] },
      options: { cutout: "62%", plugins: { legend: { position: "bottom" },
        tooltip: { callbacks: { label: (c) => " " + c.label + ": " + fmt(c.raw) + " (" + (c.raw / m.customers * 100).toFixed(1) + "%)" } } } }
    });
    vbar("c4-tenure", t.tenure.labels, t.tenure.values, C.amber, { dp: 1, suffix: "%" });

    // charges histogram (two series)
    const all = t.charges.stayed.concat(t.charges.churned);
    const hS = histogram(t.charges.stayed, 26, 18, 120), hC = histogram(t.charges.churned, 26, 18, 120);
    new Chart($("#c4-charges"), {
      type: "bar",
      data: { labels: hS.labels, datasets: [
        { label: "Stayed", data: hS.counts, backgroundColor: "rgba(16,185,129,.75)", borderRadius: 3, barPercentage: 1, categoryPercentage: .95 },
        { label: "Churned", data: hC.counts, backgroundColor: "rgba(239,68,68,.75)", borderRadius: 3, barPercentage: 1, categoryPercentage: .95 }] },
      options: { plugins: { legend: { position: "top" } },
        scales: { x: axis({ title: { display: true, text: "Monthly charges ($)" }, grid: { display: false } }),
                  y: axis({ beginAtZero: true, title: { display: true, text: "Customers" } }) } }
    });

    // metrics table
    const names = Object.keys(m.models);
    $("#t4-models").innerHTML = names.map((n) => {
      const v = m.models[n], best = n === m.best_model;
      return `<tr${best ? ' style="background:#ecfaf4"' : ""}><td><b>${n}</b>${best ? ' <span class="pill green">best</span>' : ""}</td>
        <td class="num">${v.Accuracy.toFixed(3)}</td><td class="num">${v.Precision.toFixed(3)}</td>
        <td class="num">${v.Recall.toFixed(3)}</td><td class="num">${v.F1.toFixed(3)}</td><td class="num">${v.ROC_AUC.toFixed(3)}</td></tr>`;
    }).join("");

    const cm = t.compare;
    new Chart($("#c4-compare"), {
      type: "bar",
      data: { labels: cm.models, datasets: Object.keys(cm.metrics).map((k, i) => ({
        label: k, data: cm.metrics[k], backgroundColor: [C.blue, C.amber, C.green, C.purple][i], borderRadius: 3 })) },
      options: { plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } },
        scales: { x: axis({ grid: { display: false }, ticks: { maxTicksLimit: 4, font: { size: 9 } } }),
                  y: axis({ beginAtZero: true, max: 1 }) } }
    });

    // confusion matrix grid
    const M = t.confusion.matrix, lab = t.confusion.labels;
    const mx = Math.max(...M.flat());
    const cell = (v, i, j) => `<div style="background:rgba(37,99,235,${(v / mx * 0.85 + 0.06).toFixed(2)});
      color:${v / mx > .5 ? "#fff" : "#0f172a"}; border-radius:8px; padding:14px 6px; text-align:center;">
      <div style="font-size:20px; font-weight:750;">${fmt(v)}</div></div>`;
    $("#t4-confusion").innerHTML = `
      <div style="display:grid; grid-template-columns:60px 1fr 1fr; gap:7px; align-items:center; font-size:12px;">
        <div></div><div style="text-align:center; color:var(--muted); font-weight:650;">Pred: Stay</div>
        <div style="text-align:center; color:var(--muted); font-weight:650;">Pred: Churn</div>
        <div style="text-align:right; color:var(--muted); font-weight:650;">Actual<br>Stay</div>${cell(M[0][0])}${cell(M[0][1])}
        <div style="text-align:right; color:var(--muted); font-weight:650;">Actual<br>Churn</div>${cell(M[1][0])}${cell(M[1][1])}
      </div>
      <div class="note" style="margin-top:12px;">Model: <b>${t.confusion.model}</b> · TN ${M[0][0]} · FP ${M[0][1]} · FN ${M[1][0]} · TP ${M[1][1]}</div>`;

    // ROC
    const rocDs = Object.keys(t.roc).map((n, i) => ({
      label: n + " (AUC=" + t.roc[n].auc.toFixed(3) + ")",
      data: t.roc[n].fpr.map((x, k) => ({ x, y: t.roc[n].tpr[k] })),
      borderColor: [C.blue, C.amber, C.green, C.purple][i], borderWidth: 1.8, pointRadius: 0, fill: false, tension: .1
    }));
    rocDs.push({ label: "Random", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: C.slate, borderDash: [4, 4], borderWidth: 1, pointRadius: 0 });
    new Chart($("#c4-roc"), {
      type: "line", data: { datasets: rocDs },
      options: { plugins: { legend: { position: "bottom", labels: { font: { size: 9 }, boxWidth: 9 } } },
        scales: { x: axis({ type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate" } }),
                  y: axis({ min: 0, max: 1, title: { display: true, text: "True positive rate" } }) } }
    });

    hbar("c4-imp", t.importance.labels, t.importance.values, C.purple, { dp: 3 });

    // ---- interactive churn calculator ----
    const L = t.logit_model;
    function sync() {
      const tenure = +$("#s4-tenure").value, monthly = +$("#s4-monthly").value;
      const total = Math.round(tenure * monthly);
      $("#v4-tenure").textContent = tenure;
      $("#v4-monthly").textContent = monthly;
      $("#v4-total").textContent = fmt(total);
      $("#s4-total").value = Math.min(total, 8700);
      const contract = +$("#s4-contract").value, internet = +$("#s4-internet").value;
      const vec = Object.assign({}, L.baseline);
      vec.tenure = tenure; vec.MonthlyCharges = monthly; vec.TotalCharges = total;
      vec.avg_monthly_spend = tenure > 0 ? +(total / tenure).toFixed(2) : monthly;
      vec.Contract = contract; vec.InternetService = internet;
      // contract_group category codes: Annual=0, Long-term=1, No commitment=2
      vec.contract_group = [2, 0, 1][contract];
      let z = L.intercept;
      L.features.forEach((f, i) => { z += L.coef[i] * ((vec[f] - L.scaler_mean[i]) / L.scaler_scale[i]); });
      const p = 1 / (1 + Math.exp(-z));
      const pct = p * 100;
      $("#t4-prob").textContent = pct.toFixed(1) + "%";
      const bar = $("#t4-meter");
      bar.style.width = pct + "%";
      const col = pct >= 60 ? C.red : pct >= 35 ? C.amber : C.green;
      bar.style.background = col; $("#t4-prob").style.color = col;
      $("#t4-band").textContent = pct >= 60 ? "High churn risk — prioritise retention offer"
        : pct >= 35 ? "Moderate risk — worth a proactive check-in" : "Low risk — customer likely to stay";
    }
    ["s4-tenure", "s4-monthly", "s4-contract", "s4-internet"].forEach((id) => $("#" + id).addEventListener("input", sync));
    sync();
  }

  /* ================= SUBMISSION ================= */
  function renderSubmission() {
    const items = [
      ["Upload all project source code on GitHub", "Repository structure, README and per-task scripts are ready — push the <code>phase1_internship/</code> folder.", true],
      ["Deploy / host the project and submit the live link", "This dashboard is a static site — deploy the <code>web/</code> folder to GitHub Pages, Netlify or Vercel for a public URL.", true],
      ["Create a professional LinkedIn post", "Draft post covering the project, technologies and learning experience is in <code>reports/linkedin_post.md</code>.", true],
      ["Include project screenshots / demo images", "Charts and dashboard screenshots are available in <code>web/assets/</code> and the report.", true],
      ["Tag Sqrock IT Solutions in the post", "Remember to tag <b>@Sqrock IT Solutions</b> when publishing — this is a manual LinkedIn step.", false],
      ["Complete at least 3 of 4 tasks", "All 4 tasks completed — exceeds the Phase 1 requirement.", true]
    ];
    $("#submissionList").innerHTML = items.map((i) => `
      <div class="chk"><div class="ic ${i[2] ? "" : "todo"}">${i[2] ? "✓" : "!"}</div>
        <div><b style="font-size:13.8px;">${i[0]}</b><div class="note" style="margin-top:2px;">${i[1]}</div></div></div>`).join("");
  }

  const RENDER = { overview: renderOverview, task1: renderTask1, task2: renderTask2,
                   task3: renderTask3, task4: renderTask4, submission: renderSubmission };

  renderOverview(); rendered.overview = true;
})();
