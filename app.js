// AIML PREDICTOR APP V2.5 - RECOVERY MODE
let sampleData = null;
let latestInputValues = null;
let latestPrediction = null;
let latestDeviation = 0;

const featureGrid = document.querySelector("#featureGrid");
const metricsGrid = document.querySelector("#metricsGrid");
const bestModel = document.querySelector("#bestModel");
const reportBestModel = document.querySelector("#reportBestModel");
const resultBox = document.querySelector("#resultBox");
const probabilities = document.querySelector("#probabilities");
const featureImportance = document.querySelector("#featureImportance");
const featureChart = document.querySelector("#featureChart");
const liveBadge = document.querySelector("#liveBadge");
const analysisFeaturePlot = document.querySelector("#analysisFeaturePlot");
const analysisProbabilityPlot = document.querySelector("#analysisProbabilityPlot");
const analysisLiveBadge = document.querySelector("#analysisLiveBadge");
const analysisRiskTrend = document.querySelector("#analysisRiskTrend");
const analysisPrediction = document.querySelector("#analysisPrediction");
const trainButton = document.querySelector("#trainModel");
const trainStatus = document.querySelector("#trainStatus");
const generateReportButton = document.querySelector("#generateReport");
const downloadReportLink = document.querySelector("#downloadReport");
const reportStatus = document.querySelector("#reportStatus");

const liveClassDistChart = document.querySelector("#liveClassDistChart");
const liveFeatureImpChart = document.querySelector("#liveFeatureImpChart");
const livePcaClusterChart = document.querySelector("#livePcaClusterChart");
const trainConsole = document.querySelector("#trainConsole");
const trainLog = document.querySelector("#trainLog");

function titleCase(value) {
  return value
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function activatePage(pageName) {
  const nextPage = pageName || "home";
  document.querySelectorAll("[data-page]").forEach((page) => {
    page.classList.toggle("active", page.dataset.page === nextPage);
  });
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === nextPage);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function syncRoute() {
  const pageName = window.location.hash.replace("#", "") || "home";
  activatePage(pageName);
}

function renderMetrics(metrics) {
  const bestMetrics = metrics[sampleData.best_model];
  const rows = [
    ["Accuracy", bestMetrics.accuracy],
    ["Precision", bestMetrics.precision_macro],
    ["Recall", bestMetrics.recall_macro],
    ["F1-score", bestMetrics.f1_macro],
  ];
  metricsGrid.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="metric">
          <span>${label}</span>
          <strong>${(value * 100).toFixed(1)}%</strong>
        </div>
      `,
    )
    .join("");
}

function renderFeatureInputs(values) {
  featureGrid.innerHTML = sampleData.feature_names
    .map(
      (feature) => `
        <label>
          ${titleCase(feature)}
          <input
            type="number"
            step="0.0001"
            name="${feature}"
            value="${Number(values[feature]).toFixed(4)}"
            required
          >
        </label>
      `,
    )
    .join("");

  featureGrid.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", renderFeatureChart);
  });
}

function renderImportance() {
  const maxValue = Math.max(...sampleData.feature_importances.map((item) => item.importance));
  featureImportance.innerHTML = sampleData.feature_importances
    .slice(0, 6)
    .map((item) => {
      const width = (item.importance / maxValue) * 100;
      return `
        <div class="bar-row">
          <div class="bar-label">
            <span>${titleCase(item.feature)}</span>
            <span>${item.importance.toFixed(3)}</span>
          </div>
          <div class="bar"><span style="width:${width}%"></span></div>
        </div>
      `;
    })
    .join("");
}

function loadValues(values) {
  for (const feature of sampleData.feature_names) {
    const input = document.querySelector(`input[name="${CSS.escape(feature)}"]`);
    input.value = Number(values[feature]).toFixed(4);
  }
  renderFeatureChart();
}

function collectValues() {
  const values = {};
  for (const feature of sampleData.feature_names) {
    values[feature] = Number(document.querySelector(`input[name="${CSS.escape(feature)}"]`).value);
  }
  return values;
}

function getChartFeatures() {
  return sampleData.feature_importances.slice(0, 8).map((item) => item.feature);
}

function getRiskLevel(deviation) {
  if (deviation > 0.42) {
    return "high";
  }
  if (deviation > 0.22) {
    return "medium";
  }
  return "stable";
}

function renderAnalysisFeaturePlot(values) {
  if (!analysisFeaturePlot || !sampleData) {
    return;
  }

  const features = getChartFeatures();
  const points = features.map((feature) => {
    const average = sampleData.feature_means[feature] || 1;
    const current = values[feature] || 0;
    const ratio = average === 0 ? 1 : current / average;
    return {
      feature,
      ratio: Math.max(0, ratio),
      current,
      average,
    };
  });

  const width = 560;
  const height = 260;
  const margin = { top: 18, right: 18, bottom: 58, left: 52 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const step = innerWidth / points.length;
  const maxRatio = Math.max(1.6, ...points.map((point) => point.ratio * 1.05));
  const riskLevel = getRiskLevel(latestDeviation);
  const fillColor = riskLevel === "high" ? "#b42318" : riskLevel === "medium" ? "#b54708" : "#2454a6";

  const bars = points
    .map((point, index) => {
      const x = margin.left + index * step + 8;
      const barWidth = Math.max(16, step - 16);
      const barHeight = Math.min(innerHeight, (point.ratio / maxRatio) * innerHeight);
      const y = margin.top + innerHeight - barHeight;
      const label = titleCase(point.feature).split(" ").slice(0, 2).join(" ");
      return `
        <rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barHeight.toFixed(2)}" rx="3" fill="${fillColor}" opacity="0.9"></rect>
        <text x="${(x + barWidth / 2).toFixed(2)}" y="${(height - 22).toFixed(2)}" text-anchor="middle" font-size="10" fill="#475467">${label}</text>
      `;
    })
    .join("");

  const baselineY = margin.top + innerHeight - (1 / maxRatio) * innerHeight;
  const gridLines = [0.5, 1.0, 1.5, 2.0]
    .filter((tick) => tick <= maxRatio)
    .map((tick) => {
      const y = margin.top + innerHeight - (tick / maxRatio) * innerHeight;
      return `
        <line x1="${margin.left}" y1="${y.toFixed(2)}" x2="${(margin.left + innerWidth).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#d0d5dd" stroke-width="1" stroke-dasharray="3 4"></line>
        <text x="${(margin.left - 8).toFixed(2)}" y="${(y + 4).toFixed(2)}" text-anchor="end" font-size="10" fill="#667085">${tick.toFixed(1)}x</text>
      `;
    })
    .join("");

  analysisFeaturePlot.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
    <text x="${margin.left}" y="12" font-size="12" fill="#344054" font-weight="700">Live Feature Ratio Plot</text>
    ${gridLines}
    <line x1="${margin.left}" y1="${(margin.top + innerHeight).toFixed(2)}" x2="${(margin.left + innerWidth).toFixed(2)}" y2="${(margin.top + innerHeight).toFixed(2)}" stroke="#98a2b3" stroke-width="1"></line>
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${(margin.top + innerHeight).toFixed(2)}" stroke="#98a2b3" stroke-width="1"></line>
    <line x1="${margin.left}" y1="${baselineY.toFixed(2)}" x2="${(margin.left + innerWidth).toFixed(2)}" y2="${baselineY.toFixed(2)}" stroke="#111827" stroke-width="1.2"></line>
    <text x="${(margin.left + innerWidth + 8).toFixed(2)}" y="${(baselineY + 4).toFixed(2)}" font-size="10" fill="#344054">Baseline</text>
    ${bars}
  `;
}

function renderAnalysisProbabilityPlot() {
  if (!analysisProbabilityPlot || !sampleData) {
    return;
  }

  const width = 560;
  const height = 260;
  const margin = { top: 18, right: 18, bottom: 42, left: 52 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const probabilities = latestPrediction?.probabilities || {
    malignant: 0,
    benign: 0,
  };
  const entries = Object.entries(probabilities).map(([label, value]) => [titleCase(label), Number(value)]);
  const step = innerWidth / Math.max(entries.length, 1);

  const bars = entries
    .map(([label, probability], index) => {
      const x = margin.left + index * step + 30;
      const barWidth = Math.max(70, step - 60);
      const barHeight = Math.max(0, Math.min(innerHeight, probability * innerHeight));
      const y = margin.top + innerHeight - barHeight;
      const color = label.toLowerCase().includes("malignant") ? "#d1495b" : "#2a9d8f";
      return `
        <rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barHeight.toFixed(2)}" rx="4" fill="${color}"></rect>
        <text x="${(x + barWidth / 2).toFixed(2)}" y="${(y - 6).toFixed(2)}" text-anchor="middle" font-size="11" fill="#1d2939">${(probability * 100).toFixed(1)}%</text>
        <text x="${(x + barWidth / 2).toFixed(2)}" y="${(height - 16).toFixed(2)}" text-anchor="middle" font-size="11" fill="#475467">${label}</text>
      `;
    })
    .join("");

  const gridLines = [0.25, 0.5, 0.75, 1.0]
    .map((tick) => {
      const y = margin.top + innerHeight - tick * innerHeight;
      return `
        <line x1="${margin.left}" y1="${y.toFixed(2)}" x2="${(margin.left + innerWidth).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#d0d5dd" stroke-width="1" stroke-dasharray="3 4"></line>
        <text x="${(margin.left - 8).toFixed(2)}" y="${(y + 4).toFixed(2)}" text-anchor="end" font-size="10" fill="#667085">${Math.round(tick * 100)}%</text>
      `;
    })
    .join("");

  analysisProbabilityPlot.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
    <text x="${margin.left}" y="12" font-size="12" fill="#344054" font-weight="700">Prediction Probability Distribution</text>
    ${gridLines}
    <line x1="${margin.left}" y1="${(margin.top + innerHeight).toFixed(2)}" x2="${(margin.left + innerWidth).toFixed(2)}" y2="${(margin.top + innerHeight).toFixed(2)}" stroke="#98a2b3" stroke-width="1"></line>
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${(margin.top + innerHeight).toFixed(2)}" stroke="#98a2b3" stroke-width="1"></line>
    ${bars}
  `;
}

function renderLiveClassDistribution() {
  if (!liveClassDistChart) return;
  
  let labels = [], values = [], title = "Dataset Distribution", colors = [];
  
  if (latestPrediction) {
    title = `Patient: ${titleCase(latestPrediction.prediction_label)} Confidence`;
    labels = Object.keys(latestPrediction.probabilities).map(titleCase);
    values = Object.values(latestPrediction.probabilities);
    colors = labels.map(l => l.toLowerCase().includes("malignant") ? "#d1495b" : "#2a9d8f");
  } else if (sampleData?.class_distribution) {
    labels = Object.keys(sampleData.class_distribution).map(titleCase);
    values = Object.values(sampleData.class_distribution);
    colors = labels.map(l => l.toLowerCase().includes("malignant") ? "#d1495b" : "#2a9d8f");
  }

  // FORCE FALLBACK if data is missing
  if (!labels || labels.length === 0) {
    labels = ["Malignant", "Benign"];
    values = [212, 357];
    colors = ["#d1495b", "#2a9d8f"];
  }

  const trace = {
    x: labels,
    y: values,
    type: "bar",
    marker: { color: colors, line: { color: "#111827", width: 1 } },
    text: values.map(v => latestPrediction ? `${(v * 100).toFixed(1)}%` : v),
    textposition: "auto",
  };

  const layout = {
    title: { text: title, font: { size: 13, color: "#111827" } },
    margin: { t: 40, b: 30, l: 40, r: 10 },
    height: 310,
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { automargin: true },
    yaxis: { title: latestPrediction ? "Prob" : "Count", automargin: true }
  };

  Plotly.newPlot(liveClassDistChart, [trace], layout, { displayModeBar: false, responsive: true });
}

function renderLiveFeatureImportanceChart() {
  if (!liveFeatureImpChart || !sampleData?.feature_importances) return;
  const data = [...sampleData.feature_importances].reverse().slice(-10);
  const labels = data.map(d => titleCase(d.feature));
  const importanceValues = data.map(d => d.importance);
  
  const traces = [
    {
      y: labels,
      x: importanceValues,
      type: "bar",
      orientation: "h",
      name: "Feature Weight",
      marker: { color: "rgba(58, 134, 255, 0.3)", line: { color: "#3a86ff", width: 1 } },
      hoverinfo: "x",
    }
  ];

  if (latestInputValues) {
    const deviationValues = data.map(d => {
      const avg = sampleData.feature_means[d.feature] || 1;
      const val = latestInputValues[d.feature] || 0;
      // Show ratio as a secondary bar overlay
      return (val / avg) * (Math.max(...importanceValues) / 2); 
    });

    traces.push({
      y: labels,
      x: deviationValues,
      type: "scatter",
      mode: "markers",
      name: "Patient Input (rel. to avg)",
      marker: { color: "#f59e0b", size: 10, line: { color: "#111827", width: 1 } },
      hoverinfo: "text",
      text: data.map(d => {
        const avg = sampleData.feature_means[d.feature] || 1;
        const val = latestInputValues[d.feature] || 0;
        return `Feature: ${titleCase(d.feature)}<br>Patient Value: ${val.toFixed(3)}<br>Avg: ${avg.toFixed(3)}<br>Ratio: ${((val/avg)*100).toFixed(1)}%`;
      })
    });
  }

  const layout = {
    title: { text: "Impact & Patient Input Comparison", font: { size: 14 } },
    margin: { t: 40, b: 40, l: 140, r: 20 },
    height: 340,
    paper_bgcolor: "transparent",
    plot_bgcolor: "white",
    xaxis: { gridcolor: "#f1f3f5", title: "Scale" },
    yaxis: { gridcolor: "#f1f3f5" },
    legend: { orientation: "h", y: -0.2 }
  };

  Plotly.react("liveFeatureImpChart", traces, layout, { displayModeBar: false, responsive: true });
}

function projectCurrentSample(values) {
  if (!sampleData?.cluster_summary?.projection_params) return null;
  const { scaler_mean, scaler_scale, pca_components } = sampleData.cluster_summary.projection_params;
  const features = sampleData.feature_names;
  
  // 1. Standardize (X - mean) / std
  const scaled = features.map((f, i) => (values[f] - scaler_mean[i]) / scaler_scale[i]);
  
  // 2. Project onto 2 PCA components (dot product)
  const pc1 = scaled.reduce((sum, val, i) => sum + val * pca_components[0][i], 0);
  const pc2 = scaled.reduce((sum, val, i) => sum + val * pca_components[1][i], 0);
  
  return { x: pc1, y: pc2 };
}

function renderLivePcaClusters() {
  if (!livePcaClusterChart || !sampleData?.cluster_summary?.pca_points) return;
  const points = sampleData.cluster_summary.pca_points;
  
  const malignant = points.filter(p => p.label.toLowerCase() === "malignant");
  const benign = points.filter(p => p.label.toLowerCase() === "benign");

  const traces = [
    {
      x: malignant.map(p => p.x),
      y: malignant.map(p => p.y),
      mode: "markers",
      type: "scatter",
      name: "Malignant",
      marker: { color: "#d1495b", size: 6, opacity: 0.5, symbol: "circle" },
      hoverinfo: "text",
      text: malignant.map(p => `Label: Malignant<br>PC1: ${p.x.toFixed(2)}<br>PC2: ${p.y.toFixed(2)}`),
    },
    {
      x: benign.map(p => p.x),
      y: benign.map(p => p.y),
      mode: "markers",
      type: "scatter",
      name: "Benign",
      marker: { color: "#2a9d8f", size: 6, opacity: 0.5, symbol: "circle" },
      hoverinfo: "text",
      text: benign.map(p => `Label: Benign<br>PC1: ${p.x.toFixed(2)}<br>PC2: ${p.y.toFixed(2)}`),
    }
  ];

  const currentCoords = latestInputValues ? projectCurrentSample(latestInputValues) : null;
  if (currentCoords) {
    traces.push({
      x: [currentCoords.x],
      y: [currentCoords.y],
      mode: "markers",
      type: "scatter",
      name: "Current Patient",
      marker: { 
        color: "#f59e0b", 
        size: 14, 
        line: { color: "#111827", width: 2 },
        symbol: "star"
      },
      hoverinfo: "text",
      text: "Current Patient Input Location"
    });
  }

  const layout = {
    title: { text: "PCA View: Patient Mapping", font: { size: 14 } },
    margin: { t: 50, b: 50, l: 50, r: 100 },
    height: 440,
    paper_bgcolor: "transparent",
    plot_bgcolor: "white",
    xaxis: { title: "PC1", gridcolor: "#f1f3f5" },
    yaxis: { title: "PC2", gridcolor: "#f1f3f5" },
    legend: { x: 1.05, y: 1 },
    hovermode: "closest",
  };

  Plotly.react("livePcaClusterChart", traces, layout, { responsive: true });
}

function renderChartForValues(targetElement, badgeElement, values) {
  const features = getChartFeatures();
  let totalDeviation = 0;

  targetElement.innerHTML = features
    .map((feature) => {
      const average = sampleData.feature_means[feature] || 1;
      const current = values[feature] || 0;
      const ratio = Math.max(0, current / average);
      const deviation = Math.abs(ratio - 1);
      totalDeviation += deviation;
      const currentWidth = Math.min(100, (ratio / 1.8) * 100);
      const averageMarker = (1 / 1.8) * 100;
      const status = deviation > 0.45 ? "high" : deviation > 0.22 ? "medium" : "normal";

      return `
        <div class="chart-row ${status}">
          <div class="chart-label">
            <strong>${titleCase(feature)}</strong>
            <span>Input ${current.toFixed(3)} | Avg ${average.toFixed(3)}</span>
          </div>
          <div class="chart-track" aria-label="${titleCase(feature)} current value compared to average">
            <span class="average-marker" style="left:${averageMarker}%"></span>
            <span class="chart-fill" style="width:${currentWidth}%"></span>
          </div>
          <div class="chart-ratio">${Math.round(ratio * 100)}%</div>
        </div>
      `;
    })
    .join("");

  const averageDeviation = totalDeviation / features.length;
  badgeElement.textContent = `Deviation ${(averageDeviation * 100).toFixed(1)}%`;
  return averageDeviation;
}

function updateAnalysisSummary() {
  if (!analysisRiskTrend || !analysisPrediction) {
    return;
  }
  const riskText = latestDeviation > 0.42 ? "High shift" : latestDeviation > 0.22 ? "Moderate shift" : "Stable";
  analysisRiskTrend.textContent = riskText;
  analysisPrediction.textContent = latestPrediction ? titleCase(latestPrediction.prediction_label) : "Not predicted yet";
}

function renderFeatureChart() {
  if (!sampleData) {
    return;
  }
  const values = collectValues();
  latestInputValues = values;
  latestDeviation = renderChartForValues(featureChart, liveBadge, values);
  if (analysisLiveBadge) {
    analysisLiveBadge.textContent = `Deviation ${(latestDeviation * 100).toFixed(1)}%`;
  }
  renderAnalysisFeaturePlot(values);
  renderLiveClassDistribution();
  renderLiveFeatureImportanceChart();
  renderLivePcaClusters();
  updateAnalysisSummary();
}

function renderPrediction(data) {
  latestPrediction = data;
  const label = data.prediction_label;
  resultBox.className = `result-box ${label}`;
  resultBox.innerHTML = `
    <span>Predicted diagnosis</span>
    <strong>${titleCase(label)}</strong>
  `;
  probabilities.innerHTML = Object.entries(data.probabilities)
    .map(([name, value]) => {
      const width = value * 100;
      return `
        <div class="bar-row">
          <div class="bar-label">
            <span>${titleCase(name)} probability</span>
            <span>${width.toFixed(1)}%</span>
          </div>
          <div class="bar"><span style="width:${width}%"></span></div>
        </div>
      `;
    })
    .join("");
  renderAnalysisProbabilityPlot();
  renderLiveClassDistribution(); // Refresh Class Dist to show prediction bars
  updateAnalysisSummary();
}

async function predict(event) {
  event.preventDefault();
  resultBox.className = "result-box";
  resultBox.innerHTML = "<span>Running model</span><strong>Predicting...</strong>";
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features: collectValues() }),
  });
  const payload = await response.json();
  if (!response.ok) {
    resultBox.innerHTML = `<span>Error</span><strong>${payload.error}</strong>`;
    return;
  }
  renderPrediction(payload);
}

async function reloadSampleData() {
  const response = await fetch(`/api/sample?t=${Date.now()}`);
  sampleData = await response.json();
  
  bestModel.textContent = sampleData.best_model;
  reportBestModel.textContent = sampleData.best_model;
  
  // 1. First create the UI elements (inputs/metrics)
  renderFeatureInputs(sampleData.feature_means);
  renderMetrics(sampleData.metrics);
  
  // 2. Then render charts that depend on them
  renderImportance();
  renderFeatureChart();
  renderLiveClassDistribution();
  renderLiveFeatureImportanceChart();
  renderLivePcaClusters();
}

async function trainModel() {
  trainButton.disabled = true;
  trainStatus.textContent = "Executing training cells...";
  trainConsole.hidden = false;
  trainLog.textContent = "[CELL 1] Loading Wisconsin Breast Cancer dataset...\n";
  
  try {
    // Simulated live log sequence for "Colab" feel
    setTimeout(() => trainLog.textContent += "[CELL 2] Preprocessing data: StandardScaler fit & transform...\n", 300);
    setTimeout(() => trainLog.textContent += "[CELL 3] Training Random Forest (n_estimators=250)...\n", 800);
    setTimeout(() => trainLog.textContent += "[CELL 4] Training SVM with RBF kernel...\n", 1400);
    setTimeout(() => trainLog.textContent += "[CELL 5] Performing PCA (n_components=2)...\n", 2000);
    setTimeout(() => trainLog.textContent += "[CELL 6] Running K-Means clustering...\n", 2500);

    const response = await fetch("/api/train", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Training failed");

    setTimeout(async () => {
      await reloadSampleData();
      trainLog.textContent += `[FINISH] Training complete. Best Accuracy: ${(payload.metrics.accuracy * 100).toFixed(1)}%\n`;
      trainStatus.textContent = `Execution complete. Best model: ${payload.best_model}.`;
    }, 3000);

  } catch (error) {
    trainLog.textContent += `\n[ERROR] ${error.message}\n`;
    trainStatus.textContent = `Execution failed: ${error.message}`;
  } finally {
    setTimeout(() => trainButton.disabled = false, 3000);
  }
}

async function generateDetailedReport() {
  if (!sampleData) {
    return;
  }
  generateReportButton.disabled = true;
  reportStatus.textContent = "Generating detailed report...";
  downloadReportLink.hidden = true;

  const snapshot = latestInputValues || collectValues();
  const response = await fetch("/api/report/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_values: snapshot,
      prediction: latestPrediction,
      average_deviation: Number(latestDeviation.toFixed(4)),
    }),
  });
  const payload = await response.json();

  if (!response.ok) {
    reportStatus.textContent = `Report generation failed: ${payload.error || "Unknown error"}`;
    generateReportButton.disabled = false;
    return;
  }

  reportStatus.textContent = `Report ready: ${payload.file_name}`;
  const cacheBustSeparator = payload.download_url.includes("?") ? "&" : "?";
  downloadReportLink.href = `${payload.download_url}${cacheBustSeparator}t=${Date.now()}`;
  downloadReportLink.hidden = false;
  generateReportButton.disabled = false;
}

async function init() {
  await reloadSampleData();

  document.querySelector("#loadBenign").addEventListener("click", () => loadValues(sampleData.benign_example));
  document.querySelector("#loadMalignant").addEventListener("click", () => loadValues(sampleData.malignant_example));
  document.querySelector("#loadMean").addEventListener("click", () => loadValues(sampleData.feature_means));
  document.querySelector("#predictionForm").addEventListener("submit", predict);
  trainButton.addEventListener("click", trainModel);
  generateReportButton.addEventListener("click", generateDetailedReport);
  window.addEventListener("hashchange", syncRoute);
  syncRoute();
}

init();
