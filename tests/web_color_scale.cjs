/* Browser-independent checks for the frontend's color-limit calculations.
 * Node is used only by CI; the shipped Explorer needs no Node/npm runtime. */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const code = fs.readFileSync(
  path.join(__dirname, "../milgrau/web/static/js/app.js"), "utf8"
);
const html = fs.readFileSync(
  path.join(__dirname, "../milgrau/web/static/index.html"), "utf8"
);
for (const id of ["session", "plot-editor", "edit-plot", "editor-colors",
                   "editor-axes", "color-mode", "percentile-min", "percentile-max",
                   "color-min", "color-max", "colorscale", "log10", "axis-ymin", "axis-ymax",
                   "footer-logos"]) {
  assert(html.includes('id="' + id + '"'), "Missing plot control: " + id);
}
const nodes = new Map();
const events = [];
function element(id) {
  if (!nodes.has(id)) nodes.set(id, {
    value: "", checked: false, disabled: false, textContent: "", data: [{}],
    classList: { toggle() {} },
    setAttribute() {},
    addEventListener() {},
    replaceChildren() {},
    appendChild() {},
  });
  return nodes.get(id);
}
element("color-mode").value = "robust";
element("percentile-min").value = "2";
element("percentile-max").value = "98";
element("colorscale").value = "Jet";
element("rcs-chart")._fullLayout = {
  xaxis: { range: ["2026-05-28 14:00:00", "2026-05-28 21:00:00"] },
  yaxis: { range: [0, 30] },
};
let fetchCount = 0;
const axisEvents = [];
const plotly = {
  async restyle(id, change, index) {
    events.push({ id, change, index });
  },
  async relayout(id, update) {
    axisEvents.push({ id, update });
  },
  async react() {},
  purge() {},
  Plots: { resize() {} },
};
const context = vm.createContext({
  document: {
    getElementById: element,
    querySelectorAll: () => [],
    createElement: () => element("temporary"),
  },
  window: { Plotly: plotly },
  Plotly: plotly,
  URLSearchParams,
  fetch: async (url) => {
    fetchCount += 1;
    return {
      ok: true,
      json: async () => url.includes("/api/site")
        ? { name: "SPU", latitude: 0, longitude: 0, elevation_m: 0, logos: [] }
        : { sessions: [] },
    };
  },
});
vm.runInContext(code, context);

async function run() {
  // Let the initial catalog request settle, then work with synthetic saved RCS.
  await new Promise((resolve) => setImmediate(resolve));
  const readings = Array.from({ length: 100 }, (_, i) => i);
  readings.push(1e9);  // one extreme value that ruins Plotly's auto colors
  const source = { rcs: [[-4, 0, ...readings, null]], units: "a.u." };
  context.input = source;
  vm.runInContext("displayedResult = input; colorCache = null;", context);
  const beforeRecolor = fetchCount;

  await vm.runInContext("updateAppearance()", context);
  assert.equal(events.length, 1);
  assert.equal(events.at(-1).change.zauto[0], false);
  assert(events.at(-1).change.zmax[0] < 1000, "robust colors suppress single outlier");
  assert(events.at(-1).change.zmin[0] < events.at(-1).change.zmax[0]);
  assert.match(element("edit-plot").textContent, /Robust P2–P98/);

  element("color-mode").value = "auto";
  await vm.runInContext("updateAppearance()", context);
  assert.equal(events.at(-1).change.zmax[0], 1e9);

  element("color-mode").value = "manual";
  element("color-min").value = "1e4";
  element("color-max").value = "1e6";
  await vm.runInContext("updateAppearance()", context);
  assert.equal(events.at(-1).change.zmin[0], 10000);
  assert.equal(events.at(-1).change.zmax[0], 1000000);
  assert.equal(element("color-min").disabled, false);

  const count = events.length;
  element("color-max").value = "10";
  await vm.runInContext("updateAppearance()", context);
  assert.equal(events.length, count, "invalid limits keep existing plot");
  assert.match(element("scale-status").textContent, /minimum < maximum/);

  element("color-mode").value = "auto";
  element("log10").checked = true;
  await vm.runInContext("updateAppearance()", context);
  const transformed = events.at(-1).change.z[0];
  assert.equal(transformed[0][0], null, "negative values are missing in log10");
  assert.equal(transformed[0][1], null, "zero is missing in log10");
  assert.equal(transformed[0][3], 0, "log10(1) = 0");
  assert.equal(events.at(-1).change.zmax[0], 9);
  assert.match(element("edit-plot").textContent, /log10/);
  assert.equal(fetchCount, beforeRecolor, "appearance never rereads the NetCDF");

  const profileData = vm.runInContext("displayedResult.mean_profile", context);
  assert.equal(profileData, undefined, "appearance does not overwrite original profiles");

  vm.runInContext("editClick({target: {closest: (query) => query.includes('.colorbar') ? {} : null}})", context);
  assert.equal(element("plot-editor").hidden, false, "clicking colorbar opens the editor");
  assert.equal(element("editor-colors").hidden, false);
  vm.runInContext("openPlotEditor('axes')", context);
  assert.equal(element("editor-axes").hidden, false);
  assert.equal(element("axis-ymax").value, "30.0000");
  element("axis-ymin").value = "2";
  element("axis-ymax").value = "14";
  await vm.runInContext("applyAxisLimits()", context);
  assert.equal(axisEvents.at(-1).update["yaxis.range"][0], 2);
  assert.equal(axisEvents.at(-1).update["yaxis.range"][1], 14);
  vm.runInContext("closePlotEditor()", context);
  assert.equal(element("plot-editor").hidden, true);
  assert.equal(fetchCount, beforeRecolor, "in-chart edits do not reread NetCDF");

  console.log("Web Explorer appearance and inline editing tests passed");
}
run().catch((error) => { console.error(error); process.exitCode = 1; });
