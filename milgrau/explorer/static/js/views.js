"use strict";

/* Additional scientific views. Data come solely from read-only, bounded NetCDF
   endpoints. No scientific computations are performed in the browser. */
(function () {
  const $ = (id) => document.getElementById(id);
  let generation = 0;
  let currentTab = "";
  let currentSession = "";
  const preferences = {};

  async function get(path, query = {}) {
    const qs = new URLSearchParams(query);
    const response = await fetch(path + (qs.size ? "?" + qs.toString() : ""));
    if (!response.ok) {
      const message = await response.json().catch(() => ({}));
      throw new Error(message.detail || "HTTP " + response.status);
    }
    return response.json();
  }
  function node(type, text, cls = "") {
    const item = document.createElement(type);
    if (text !== undefined && text !== null) item.textContent = String(text);
    if (cls) item.className = cls;
    return item;
  }
  function wipe(element) { element.replaceChildren(); }
  function sectionTitle(text) { return node("h2", text, "detail-section-title"); }
  function item(label, value) {
    const card = node("div", null, "detail-stat");
    card.append(node("span", label, "detail-stat-label"), node("strong", value ?? "—", "detail-stat-value"));
    return card;
  }
  function grid(items) {
    const container = node("div", null, "detail-stat-grid");
    items.forEach(([label, value]) => container.appendChild(item(label, value)));
    return container;
  }
  function table(headers, rows) {
    const table = node("table", null, "detail-table");
    const thead = node("thead"); const headRow = node("tr");
    headers.forEach((title) => headRow.appendChild(node("th", title)));
    thead.appendChild(headRow); table.appendChild(thead);
    const tbody = node("tbody");
    rows.forEach((values) => {
      const tr = node("tr");
      values.forEach((value) => tr.appendChild(node("td", value ?? "—")));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    const wrap = node("div", null, "detail-table-scroll");
    wrap.appendChild(table);
    return wrap;
  }
  function addSelect(container, id, title, choices, current, onChange) {
    const group = node("div", null, "detail-choice");
    const label = node("label", title); label.htmlFor = id;
    const select = node("select"); select.id = id;
    choices.forEach(([value, title]) => {
      const option = node("option", title); option.value = String(value);
      select.appendChild(option);
    });
    if (choices.some(([value]) => String(value) === String(current))) select.value = String(current);
    select.addEventListener("change", onChange);
    group.append(label, select); container.appendChild(group);
    return select;
  }
  function setHeader(title, subtitle, eyebrow) {
    $("detail-title").textContent = title;
    $("detail-caption").textContent = subtitle;
    $("detail-eyebrow").textContent = eyebrow;
  }
  function message(value, isError = false) {
    $("detail-status").textContent = value;
    $("detail-status").className = isError ? "detail-status error" : "detail-status";
  }
  function unavailable(text) {
    $("detail-body").appendChild(node("p", text, "detail-unavailable"));
  }
  function labelCase(text) { return String(text).replaceAll("_", " "); }
  function section(name, content) {
    const block = node("section", null, "detail-section");
    block.appendChild(sectionTitle(name));
    block.appendChild(content);
    $("detail-body").appendChild(block);
  }
  function emptyForPlot() { $("detail-chart").hidden = true; }
  function plot(data, layout) {
    $("detail-chart").hidden = false;
    return Plotly.react("detail-chart", data, {
      paper_bgcolor: "#fff", plot_bgcolor: "#fff",
      font: { family: "Segoe UI, Arial", size: 12, color: "#385b6e" },
      margin: { t: 52, r: 28, b: 85, l: 100 },
      ...layout,
    }, { responsive: true, displaylogo: false,
      toImageButtonOptions: { format: "png", filename: "milgrau-explorer" } });
  }
  async function overview(base, token) {
    setHeader("Session overview", "Summary of one continuous scientific acquisition", "MILGRAU · INSPECT");
    const data = await get(base + "/overview");
    if (token !== generation) return;
    section("Acquisition", grid([
      ["Session ID", data.session_id],
      ["Station", data.station],
      ["UTC start", data.start_utc],
      ["UTC end", data.end_utc],
      ["Duration", data.duration_minutes + " minutes"],
      ["Timezone", data.timezone],
      ["Local start", data.local_start + " " + data.timezone],
      ["Local end", data.local_end + " " + data.timezone],
    ]));
    section("Processing levels", grid(data.levels.map((entry) => [
      entry.level, entry.available ? "Available" : "Not available",
    ])));
    section("Session context", grid([
      ["Solar regimes", data.regime_sequence],
      ["Segments", data.segments],
      ["Saved channels", data.channels?.join(", ") || "—"],
      ["Saved figures", data.figures_count === null ? "Unknown" : data.figures_count],
    ]));
    message("");
  }

  async function metadata(base, token) {
    setHeader("NetCDF metadata", "Inspect dimensions, coordinates, variables and provenance without loading large signals", "MILGRAU · INSPECT");
    const controls = $("detail-controls");
    const initial = preferences.metadataLevel || "1";
    const select = addSelect(controls, "metadata-level", "Level", [
      ["0", "L0"], ["1", "L1"], ["2", "L2"],
    ], initial, () => { preferences.metadataLevel = select.value; render("metadata", currentSession); });
    const data = await get(base + "/metadata", { level: select.value });
    if (token !== generation) return;
    if (!data.available) { unavailable(data.reason); message(""); return; }
    section("File", grid([["NetCDF", data.file], ["Level", data.level],
      ["Dimensions", data.dimensions.length], ["Data variables", data.variables.length]]));
    section("Dimensions", table(["Dimension", "Size"], data.dimensions.map(d => [d.name, d.size])));
    section("Coordinates", table(["Name", "Shape", "Dtype", "Units", "First / last"],
      data.coordinates.map(c => [c.name, c.dims.map((d,i) => d + "=" + c.shape[i]).join(" × "),
        c.dtype, c.units, (c.first_last || []).join(" → ")])));
    const details = node("details"); details.open = true;
    details.appendChild(node("summary", "Show " + data.variables.length + " data variables"));
    details.appendChild(table(["Name", "Dimensions", "Type", "Units", "Description"],
      data.variables.map(v => [v.name, v.dims.map((d,i)=>d+"="+v.shape[i]).join(" × "),
        v.dtype, v.units, v.long_name])));
    section("Variables", details);
    const attributes = node("details");
    attributes.appendChild(node("summary", "Show " + data.attributes.length + " global attributes"));
    attributes.appendChild(table(["Attribute", "Value"], data.attributes.map(a => [a.name, a.value])));
    section("Global metadata / provenance", attributes);
    message("");
  }

  async function profiles(base, kind, token) {
    const atmospheric = kind === "atmosphere";
    setHeader(atmospheric ? "Atmospheric profiles" : "L2 optical profiles",
      atmospheric ? "Saved L1 thermodynamic inputs, not newly calculated profiles" :
        "Saved Level 2 retrieval output; scientific values are unchanged",
      atmospheric ? "LEVEL 1 · ATMOSPHERE" : "LEVEL 2 · OPTICAL PROFILES");
    const options = await get(base + "/profiles/" + kind + "/options");
    if (token !== generation) return;
    if (!options.available) { unavailable(options.reason); message(""); return; }
    const controls = $("detail-controls");
    const selected = preferences[kind] || {};
    const variable = addSelect(controls, "profile-variable", "Variable",
      options.variables.map(v => [v.id, v.label]), selected.variable,
      () => { preferences[kind] = { variable: variable.value, index: "0" }; render(kind, currentSession); });
    const index = addSelect(controls, "profile-choice",
      atmospheric ? "Atmosphere time" : "Wavelength (nm)",
      options.choices.map((value,i) => [String(i), value]),
      selected.variable === variable.value ? selected.index || "0" : "0",
      () => { preferences[kind] = { variable: variable.value, index: index.value }; render(kind, currentSession); });
    const choice = { variable: variable.value, index: index.value };
    preferences[kind] = choice;
    const result = await get(base + "/profiles/" + kind, choice);
    if (token !== generation) return;
    if (!result.available) { unavailable(result.reason); message(""); return; }
    section("Selected profile", grid([
      ["Source file", result.level],
      ["Variable", result.variable],
      [atmospheric ? "UTC timestamp" : "Wavelength", result.choice],
      ["Units", result.units || "not specified"],
      ["Sampled altitude bins", result.values.length],
      ["Atmospheric input source", result.source || "—"],
    ].filter(([key]) => atmospheric || key !== "Atmospheric input source")));
    const units = result.units ? " (" + result.units + ")" : "";
    await plot([{
      type: "scatter", mode: "lines", x: result.values, y: result.altitude_km,
      line: { width: 2, color: "#0a8391" }, connectgaps: false,
      hovertemplate: "%{x:.5g}<br>%{y:.2f} km<extra></extra>",
    }], {
      title: { text: result.title + " · " + result.choice, font: { size: 15 } },
      xaxis: { title: { text: result.title + units, standoff: 15 }, automargin: true },
      yaxis: { title: { text: "Altitude above station (km)", standoff: 12 }, automargin: true },
    });
    message("");
  }

  async function qa(base, token) {
    setHeader("Scientific QA", "Saved instrument and retrieval diagnostics; flags are not pass/fail judgments", "MILGRAU · DIAGNOSTICS");
    const options = await get(base + "/qa/options");
    if (token !== generation) return;
    if (!options.available) { unavailable(options.reason); message(""); return; }
    const controls = $("detail-controls");
    const preferred = preferences.qa || {};
    const metric = addSelect(controls, "qa-metric", "Diagnostic",
      options.metrics.map(v => [String(v.level) + ":" + v.variable, v.label]),
      preferred.metric, () => {
        preferences.qa = { metric: metric.value, index: "0" };
        render("qa", currentSession);
      });
    const selected = options.metrics.find(v => String(v.level) + ":" + v.variable === metric.value);
    const index = addSelect(controls, "qa-choice",
      selected.choice_axis === "channel" ? "Channel" : "Wavelength (nm)",
      selected.choices.map((v,i)=>[String(i),v]),
      preferred.metric === metric.value ? preferred.index || "0" : "0",
      () => { preferences.qa = { metric: metric.value, index: index.value }; render("qa", currentSession); });
    preferences.qa = { metric: metric.value, index: index.value };
    const series = await get(base + "/qa", {
      level: selected.level, variable: selected.variable, index: index.value,
    });
    if (token !== generation) return;
    if (!series.available) { unavailable(series.reason); message(""); return; }
    section("Recorded diagnostic", grid([
      ["Source", series.level], ["Variable", series.variable],
      ["Channel / wavelength", series.channel],
      ["Samples displayed", series.values.length],
      ["Units", series.units || "not specified"],
    ]));
    await plot([{
      type: "scatter", mode: "lines+markers",
      x: series.time_utc, y: series.values,
      line: { width: 1.7, color: "#0a8391" }, marker: { size: 4 },
      connectgaps: false,
      hovertemplate: "%{x}<br>%{y:.5g}<extra></extra>",
    }], {
      title: { text: series.title + " · " + series.channel, font: { size: 15 } },
      xaxis: { title: { text: "Time (UTC)", standoff: 15 }, type: "date", automargin: true },
      yaxis: { title: { text: series.title + (series.units ? " (" + series.units + ")" : ""), standoff: 10 }, automargin: true },
    });
    message("");
  }

  async function render(tab, session) {
    currentTab = tab;
    currentSession = session;
    const token = ++generation;
    wipe($("detail-controls")); wipe($("detail-body")); emptyForPlot();
    message("Loading saved scientific information…");
    if (!session) { message("No session selected."); return; }
    const base = "/api/sessions/" + encodeURIComponent(session);
    try {
      if (tab === "overview") await overview(base, token);
      else if (tab === "metadata") await metadata(base, token);
      else if (tab === "atmosphere" || tab === "optical") await profiles(base, tab, token);
      else if (tab === "qa") await qa(base, token);
      else message("Unsupported view.", true);
    } catch (error) {
      if (token === generation) {
        message("Could not display " + tab + ": " + error.message, true);
        emptyForPlot();
      }
    }
  }

  window.MilgrauViews = { render };
})();
