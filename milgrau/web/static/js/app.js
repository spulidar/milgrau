"use strict";

/* UI-only module: scientific values are supplied by the read-only Python API. */
const byId = (id) => document.getElementById(id);
const state = { session: "", product: "" };
let requestVersion = 0;
let displayedResult = null;
let colorCache = null;

/** Percentiles are computed on the finite, display-reduced cells (not NetCDF). */
function percentile(sorted, pct) {
  if (!sorted.length) return NaN;
  const position = ((sorted.length - 1) * pct) / 100;
  const lower = Math.floor(position);
  const weight = position - lower;
  return sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * weight;
}

function colorData(useLog) {
  if (!displayedResult) return null;
  if (colorCache && colorCache.source === displayedResult && colorCache.useLog === useLog) {
    return colorCache;
  }
  let nonpositive = 0;
  const matrix = useLog
    ? displayedResult.rcs.map((row) => row.map((value) => {
      if (typeof value !== "number" || !Number.isFinite(value)) return null;
      if (value <= 0) { nonpositive += 1; return null; }
      return Math.log10(value);
    }))
    : displayedResult.rcs;
  const finite = [];
  for (const row of matrix) {
    for (const value of row) {
      if (typeof value === "number" && Number.isFinite(value)) finite.push(value);
    }
  }
  finite.sort((a, b) => a - b);
  colorCache = { source: displayedResult, useLog, matrix, finite, nonpositive };
  return colorCache;
}

function formatScale(value) {
  return Number(value.toPrecision(5)).toString();
}

function scaleStatus(message, error = false) {
  byId("scale-status").textContent = message;
  byId("scale-status").classList.toggle("error", error);
}

function updateSettingAvailability() {
  const mode = byId("color-mode").value;
  for (const id of ["percentile-min", "percentile-max"]) {
    byId(id).disabled = mode !== "robust";
  }
  for (const id of ["color-min", "color-max"]) {
    byId(id).disabled = mode !== "manual";
  }
}

/** Recolor the existing Plotly trace without rereading the selected NetCDF. */
async function updateAppearance() {
  updateSettingAvailability();
  if (!displayedResult || !byId("rcs-chart").data) return;
  const useLog = byId("log10").checked;
  const values = colorData(useLog);
  if (!values || !values.finite.length) {
    const empty = values ? values.matrix : displayedResult.rcs.map((row) => row.map(() => null));
    await Plotly.restyle("rcs-chart", { z: [empty], zauto: [false], zmin: [0], zmax: [1] }, [0]);
    byId("edit-plot").textContent = "✎ Edit plot";
    scaleStatus("No finite " + (useLog ? "positive " : "") + "signal values to display.", true);
    return;
  }

  const mode = byId("color-mode").value;
  let low, high;
  let description;
  if (mode === "robust") {
    const pmin = Number(byId("percentile-min").value);
    const pmax = Number(byId("percentile-max").value);
    if (!Number.isFinite(pmin) || !Number.isFinite(pmax) ||
        pmin < 0 || pmax > 100 || pmin >= pmax ||
        byId("percentile-min").value.trim() === "" ||
        byId("percentile-max").value.trim() === "") {
      scaleStatus("Enter valid percentile bounds with 0 ≤ lower < upper ≤ 100.", true);
      return;
    }
    low = percentile(values.finite, pmin);
    high = percentile(values.finite, pmax);
    description = "Robust P" + pmin + "–P" + pmax;
  } else if (mode === "manual") {
    const a = byId("color-min").value.trim();
    const b = byId("color-max").value.trim();
    low = Number(a);
    high = Number(b);
    if (!a || !b || !Number.isFinite(low) || !Number.isFinite(high) || low >= high) {
      scaleStatus("Enter finite manual limits with minimum < maximum (e.g. 1e4 and 1e7).", true);
      return;
    }
    description = "Manual";
  } else {
    low = values.finite[0];
    high = values.finite[values.finite.length - 1];
    description = "Full-range auto";
  }
  if (low === high) {
    const epsilon = Math.max(Math.abs(low) * 0.01, 1e-12);
    low -= epsilon;
    high += epsilon;
    description += " (constant values; padded display range)";
  }

  const palette = byId("colorscale").value;
  const axisTitle = useLog ? "log10(" + displayedResult.units + ")" : displayedResult.units;
  const hoverValue = useLog ? "log10(signal)" : "Signal";
  const update = {
    zmin: [low],
    zmax: [high],
    zauto: [false],
    colorscale: [palette],
    "colorbar.title.text": [axisTitle],
    hovertemplate: ["%{x}<br>Altitude: %{y:.2f} km<br>" +
      hoverValue + ": %{z:.5g}<extra></extra>"],
  };
  if (!colorCache.applied || colorCache.lastLog !== useLog) {
    update.z = [values.matrix];
    colorCache.applied = true;
    colorCache.lastLog = useLog;
  }
  await Plotly.restyle("rcs-chart", update, [0]);
  byId("edit-plot").textContent = "✎ " + description +
    (useLog ? " · log10" : "");
  scaleStatus(
    description + " · colors [" + formatScale(low) + ", " + formatScale(high) + "]" +
    (useLog ? " log10 units" : " " + displayedResult.units) +
    " · based on finite display-reduced cells. Out-of-range cells are color-saturated, not removed." +
    (useLog && values.nonpositive ? " " + values.nonpositive + " nonpositive cells hidden in log10." : "")
  );
}

async function changeColorMode() {
  if (byId("color-mode").value === "manual" && displayedResult) {
    const data = colorData(byId("log10").checked);
    if (data && data.finite.length) {
      let suggestedLow = percentile(data.finite, 2);
      let suggestedHigh = percentile(data.finite, 98);
      if (suggestedLow === suggestedHigh) {
        const delta = Math.max(Math.abs(suggestedLow) * 0.01, 1e-12);
        suggestedLow -= delta;
        suggestedHigh += delta;
      }
      if (!byId("color-min").value.trim()) byId("color-min").value = formatScale(suggestedLow);
      if (!byId("color-max").value.trim()) byId("color-max").value = formatScale(suggestedHigh);
    }
  }
  await updateAppearance();
}


async function api(path, params = {}) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(path + (query ? "?" + query : ""));
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Server returned " + response.status);
  }
  return response.json();
}

function setMessage(value = "", error = false) {
  byId("message").textContent = value;
  byId("message").className = error ? "message error" : "message";
}

function populate(id, options, selected = "") {
  const select = byId(id);
  select.replaceChildren();
  options.forEach(({ value, label }) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  });
  if (options.some((option) => option.value === selected)) select.value = selected;
}

function resetPlots() {
  closePlotEditor();
  requestVersion += 1;
  displayedResult = null;
  colorCache = null;
  if (window.Plotly) {
    Plotly.purge("rcs-chart");
    Plotly.purge("profile-chart");
  }
}

async function loadSessions() {
  const { sessions } = await api("/api/sessions");
  populate("session", sessions.map((item) => ({
    value: item.id,
    label: item.start_utc.replace("+00:00", "Z").replace("T", " ") +
      " → " + item.end_utc.replace("+00:00", "Z").replace("T", " "),
  })), state.session);
  state.session = byId("session").value;
  await loadProducts();
}

async function loadProducts() {
  resetPlots();
  state.session = byId("session").value;
   if (!state.session) {
    populate("product", []);
    populate("channel", []);
    populate("variable", []);
    setMessage("No saved scientific sessions were found in the configured processed-data directory.");
    return;
  }
  const { products } = await api(
    "/api/sessions/" + encodeURIComponent(state.session) + "/products"
  );
  const l1 = products.filter((item) => item.level === 1);
  populate("product", l1.map((p) => ({ value: p.id, label: p.label })), state.product);
  state.product = byId("product").value;
  if (!state.product) {
    populate("channel", []);
    populate("variable", []);
    setMessage("This session has no saved L1 product. RCS requires Level 1.", true);
    return;
  }
  await loadOptions();
}

async function loadOptions() {
  requestVersion += 1;
  displayedResult = null;
  colorCache = null;
  state.product = byId("product").value;
  if (!state.session || !state.product) return;
  const { channels, variables } = await api(
    "/api/sessions/" + encodeURIComponent(state.session) + "/options",
    { product: state.product }
  );
  const currentChannel = byId("channel").value;
  populate("channel", channels.map((name) => ({ value: name, label: name })), currentChannel);
  const currentVariable = byId("variable").value;
  populate("variable", variables.map((name) => ({ value: name, label: name })),
    variables.includes(currentVariable) ? currentVariable : "range_corrected_signal");
  await loadRcs();
}

async function loadRcs() {
  if (!state.session || !state.product || !byId("channel").value || !byId("variable").value) {
    setMessage("No compatible L1 channel or variable.", true);
    resetPlots();
    return;
  }
  const version = ++requestVersion;
  setMessage("Reading selected L1 data…");
  const result = await api(
    "/api/sessions/" + encodeURIComponent(state.session) + "/rcs",
    {
      product: state.product,
      channel: byId("channel").value,
      variable: byId("variable").value,
      statistic: byId("statistic").value,
    }
  );
  if (version !== requestVersion) return;
  if (!window.Plotly) throw new Error("Local Plotly.js bundle is unavailable");
  byId("plot-title").textContent = result.variable + " · " + result.channel;
  const plotConfig = {
    responsive: true,
    editable: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: { format: "png", filename: "milgrau-rcs" },
  };
  const axisFont = { family: "Arial, sans-serif", size: 12, color: "#3c5867" };
  const common = { paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: axisFont, margin: { t: 25, b: 68, l: 68, r: 15 } };
  const heat = {
    type: "heatmap",
    x: result.time_utc,
    y: result.altitude_km_agl,
    z: result.rcs,
    colorscale: "Jet",
    connectgaps: false,
    colorbar: { title: { text: result.units, side: "right" }, thickness: 12 },
    hovertemplate: "%{x}<br>Altitude: %{y:.2f} km<br>Signal: %{z:.5g}<extra></extra>",
  };
  const profile = {
    type: "scatter", mode: "lines",
    x: result.mean_profile, y: result.altitude_km_agl,
    name: "Time mean", line: { color: "#0a8391", width: 2.5 },
    hovertemplate: "Mean signal: %{x:.5g}<br>Altitude: %{y:.2f} km<extra></extra>",
  };
  await Promise.all([
    Plotly.react("rcs-chart", [heat], {
      ...common,
      xaxis: { title: "Time (UTC)", type: "date", gridcolor: "#e8eff3" },
      yaxis: { title: "Altitude (km a.g.l.)", gridcolor: "#e8eff3" },
    }, plotConfig),
    Plotly.react("profile-chart", [profile], {
      ...common, margin: { t: 25, b: 68, l: 62, r: 15 },
      xaxis: { title: "Mean signal", gridcolor: "#e8eff3" },
      yaxis: { title: "Altitude (km a.g.l.)", gridcolor: "#e8eff3" },
    }, plotConfig),
  ]);
  if (version !== requestVersion) return;
  displayedResult = result;
  colorCache = null;
  await updateAppearance();
  byId("plot-info").textContent =
    result.samples.time + " profiles × " + result.samples.altitude + " altitude bins";
  byId("reduction-note").textContent =
    "Display-only " + (result.statistic === "max" ? "cell maximum" : "cell mean") +
    " (not native-resolution data). The vertical profile is the time mean. " +
    "Gaps and missing observations are preserved; original NetCDF is not modified.";
  setMessage("");
}

async function showTab(event) {
  const tab = event.currentTarget.dataset.tab;
  for (const button of document.querySelectorAll(".tab")) {
    button.classList.toggle("active", button.dataset.tab === tab);
    if (button.dataset.tab === tab) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  byId("rcs-panel").hidden = tab !== "rcs";
  byId("future-panel").hidden = tab === "rcs";
  if (tab !== "rcs") byId("future-title").textContent = event.currentTarget.textContent;
  else if (window.Plotly) {
    Plotly.Plots.resize("rcs-chart");
    Plotly.Plots.resize("profile-chart");
  }
}

/** Contextual editing lives over the plot and never changes the source NetCDF. */
function switchEditorTab(section) {
  for (const id of ["colors", "axes"]) {
    const active = id === section;
    byId("editor-" + id).hidden = !active;
    byId("editor-" + id + "-tab").classList.toggle("active", active);
    byId("editor-" + id + "-tab").setAttribute("aria-pressed", String(active));
  }
}

function axisTimeLabel(raw) {
  if (!raw) return "";
  // Plotly dates represent the UTC x coordinate; datetime-local is explicitly labeled UTC.
  const str = String(raw).replace(" ", "T");
  return str.slice(0, 16);
}

function fillAxisInputs() {
  const view = byId("rcs-chart")._fullLayout;
  if (!view) return;
  const xr = view.xaxis.range;
  const yr = view.yaxis.range;
  byId("axis-tmin").value = axisTimeLabel(xr[0]);
  byId("axis-tmax").value = axisTimeLabel(xr[1]);
  byId("axis-ymin").value = Number(yr[0]).toPrecision(6);
  byId("axis-ymax").value = Number(yr[1]).toPrecision(6);
}

function openPlotEditor(section = "colors") {
  const editor = byId("plot-editor");
  editor.hidden = false;
  byId("edit-plot").setAttribute("aria-expanded", "true");
  switchEditorTab(section);
  if (section === "axes") fillAxisInputs();
}

function closePlotEditor() {
  byId("plot-editor").hidden = true;
  byId("edit-plot").setAttribute("aria-expanded", "false");
}

async function applyAxisLimits() {
  const t0 = byId("axis-tmin").value;
  const t1 = byId("axis-tmax").value;
  const amin = byId("axis-ymin").value.trim();
  const amax = byId("axis-ymax").value.trim();
  const y0 = Number(amin);
  const y1 = Number(amax);
  if (!t0 || !t1 || !Number.isFinite(Date.parse(t0 + "Z")) ||
      !Number.isFinite(Date.parse(t1 + "Z")) || t0 >= t1 ||
      !amin || !amax || !Number.isFinite(y0) || !Number.isFinite(y1) || y0 >= y1) {
    scaleStatus("Invalid axes: start must precede end and altitude minimum must be below maximum.", true);
    return;
  }
  await Plotly.relayout("rcs-chart", {
    "xaxis.range": [t0 + ":00Z", t1 + ":00Z"],
    "yaxis.range": [y0, y1],
  });
  scaleStatus("Axis ranges updated for this plot only.");
}

function editClick(event) {
  if (!displayedResult || !event.target || typeof event.target.closest !== "function") return;
  if (event.target.closest(".colorbar, .cbaxis")) {
    openPlotEditor("colors");
  } else if (event.target.closest(".xaxislayer-above, .yaxislayer-above, .xtitle, .ytitle")) {
    openPlotEditor("axes");
  }
}

function safeTask(task) {
  return async () => {
    try { await task(); } catch (error) { setMessage(error.message, true); }
  };
}

async function initialize() {
  for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", showTab);
  byId("session").addEventListener("change", safeTask(loadProducts));
  byId("product").addEventListener("change", safeTask(loadOptions));
  byId("channel").addEventListener("change", safeTask(loadRcs));
  byId("variable").addEventListener("change", safeTask(loadRcs));
  byId("statistic").addEventListener("change", safeTask(loadRcs));
  byId("color-mode").addEventListener("change", safeTask(changeColorMode));
  for (const id of ["percentile-min", "percentile-max", "color-min", "color-max"]) {
    byId(id).addEventListener("change", safeTask(updateAppearance));
  }
  byId("colorscale").addEventListener("change", safeTask(updateAppearance));
  byId("log10").addEventListener("change", safeTask(async () => {
    // Limits are expressed in the active scale's units; refresh manual bounds.
    byId("color-min").value = "";
    byId("color-max").value = "";
    if (byId("color-mode").value === "manual") await changeColorMode();
    else await updateAppearance();
  }));
  updateSettingAvailability();
  byId("edit-plot").addEventListener("click", () => {
    if (byId("plot-editor").hidden) openPlotEditor("colors");
    else closePlotEditor();
  });
  byId("close-editor").addEventListener("click", closePlotEditor);
  byId("editor-colors-tab").addEventListener("click", () => switchEditorTab("colors"));
  byId("editor-axes-tab").addEventListener("click", () => {
    switchEditorTab("axes");
    fillAxisInputs();
  });
  byId("rcs-chart").addEventListener("click", editClick);
  byId("axes-apply").addEventListener("click", safeTask(applyAxisLimits));
  byId("axes-reset").addEventListener("click", safeTask(async () => {
    await Plotly.relayout("rcs-chart", {"xaxis.autorange": true, "yaxis.autorange": true});
    fillAxisInputs();
  }));
  byId("plot-editor").addEventListener("keydown", (event) => {
    if (event.key === "Escape") closePlotEditor();
  });
  try {
    const site = await api("/api/site");
    byId("station-name").textContent = site.name;
    byId("site-location").textContent =
      "LAT " + site.latitude + "° · LON " + site.longitude +
      "° · ELEV " + site.elevation_m + " m";
    const logoBox = byId("footer-logos");
    for (const logo of (site.logos || [])) {
      const img = document.createElement("img");
      img.src = logo.url;
      img.alt = logo.alt;
      img.loading = "lazy";
      logoBox.appendChild(img);
    }
    await loadSessions();
  } catch (error) { setMessage(error.message, true); }
}

initialize();
