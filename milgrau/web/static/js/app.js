"use strict";

/* UI-only module: scientific values are supplied by the read-only Python API. */
const byId = (id) => document.getElementById(id);
const state = { session: "", product: "", dates: [] };
let requestVersion = 0;

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
  if (window.Plotly) {
    Plotly.purge("rcs-chart");
    Plotly.purge("profile-chart");
  }
}

async function loadSessions() {
  const requested = byId("utc-date").value;
  const response = await api("/api/sessions", { date: requested });
  const entries = response.sessions;
  populate("session", entries.map((item) => ({
    value: item.id,
    label: item.start_utc.replace("+00:00", "Z") + " → " +
      item.end_utc.replace("+00:00", "Z"),
  })), state.session);
  state.session = byId("session").value;
  await loadProducts();
}

async function loadProducts() {
  resetPlots();
  state.session = byId("session").value;
  byId("session-title").textContent = state.session || "No sessions available";
  byId("session-subtitle").textContent =
    state.session ? "Scientific session identity is preserved across UTC dates." : "No NetCDF sessions intersect this date.";
  if (!state.session) {
    populate("product", []);
    populate("channel", []);
    populate("variable", []);
    setMessage("No scientific session for the selected date.");
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

function safeTask(task) {
  return async () => {
    try { await task(); } catch (error) { setMessage(error.message, true); }
  };
}

async function initialize() {
  for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", showTab);
  byId("utc-date").addEventListener("change", safeTask(loadSessions));
  byId("session").addEventListener("change", safeTask(loadProducts));
  byId("product").addEventListener("change", safeTask(loadOptions));
  byId("channel").addEventListener("change", safeTask(loadRcs));
  byId("variable").addEventListener("change", safeTask(loadRcs));
  byId("statistic").addEventListener("change", safeTask(loadRcs));
  try {
    const [site, available] = await Promise.all([api("/api/site"), api("/api/dates")]);
    byId("station-name").textContent = site.name;
    byId("site-location").textContent =
      "LAT " + site.latitude + "° · LON " + site.longitude +
      "° · ELEV " + site.elevation_m + " m";
    state.dates = available.dates;
    populate("utc-date", state.dates.map((value) => ({ value, label: value + " UTC" })));
    if (state.dates.length) await loadSessions();
    else {
      byId("session-title").textContent = "No processed sessions found";
      setMessage("Check the processed_data directory configured in config.yaml.");
    }
  } catch (error) { setMessage(error.message, true); }
}

initialize();
