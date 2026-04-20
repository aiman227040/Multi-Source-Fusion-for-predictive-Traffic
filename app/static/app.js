const map = L.map("map").setView([37.7749, -122.4194], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const markers = {};
const statusEl = document.getElementById("status");
const detailEl = document.getElementById("detail");

function riskClass(p) {
  if (p >= 0.55) return "risk";
  if (p >= 0.3) return "warn";
  return "ok";
}

function formatForecast(fc) {
  if (!fc || !fc.length) return "";
  return fc
    .map(
      (f) =>
        `${f.horizon_minutes}m: ${(f.standstill_probability * 100).toFixed(1)}% standstill`
    )
    .join("<br/>");
}

function renderDetail(z) {
  const fc = z.forecast || [];
  const p30 = fc.find((x) => x.horizon_minutes === 30)?.standstill_probability ?? 0;
  const fusion = z.fusion || {};
  const maps = z.maps || {};
  const vision = z.vision || {};

  detailEl.classList.remove("empty");
  detailEl.innerHTML = `
    <h2>${z.zone.name}</h2>
    <section>
      <span class="badge ${riskClass(p30)}">30m risk</span>
      <div>${formatForecast(fc)}</div>
    </section>
    <section>
      <strong>Fusion</strong><br/>
      Score: ${(fusion.fused_congestion_score * 100).toFixed(1)}%<br/>
      Maps FP filter: ${fusion.false_positive_maps ? "yes (softened)" : "no"}<br/>
      Visual escalation: ${fusion.escalation_visual ? "yes" : "no"}<br/>
      <em>${fusion.rationale || ""}</em>
    </section>
    <section>
      <strong>Maps</strong> alert L${maps.alert_level} · ratio ${maps.congestion_ratio?.toFixed(2) ?? "—"}
    </section>
    <section>
      <strong>CCTV</strong> vehicles ${vision.vehicle_count} · density ${(vision.density_score * 100).toFixed(0)}%<br/>
      ${(vision.anomaly_flags || []).join(", ") || "no anomalies"}
    </section>
    <section>
      <strong>Thumbnail</strong>
      <img class="thumb" src="${z.thumbnail_path}?t=${Date.now()}" alt="CCTV thumbnail" />
    </section>
  `;
}

function upsertMarker(z) {
  const lat = z.zone.camera_lat;
  const lng = z.zone.camera_lng;
  const fc = z.forecast || [];
  const p30 = fc.find((x) => x.horizon_minutes === 30)?.standstill_probability ?? 0;
  const r = 10 + p30 * 28;
  const color = p30 > 0.55 ? "#dc2626" : p30 > 0.3 ? "#d97706" : "#16a34a";

  if (markers[z.zone.id]) {
    map.removeLayer(markers[z.zone.id]);
  }
  const circle = L.circleMarker([lat, lng], {
    radius: r,
    color: "#0f172a",
    weight: 2,
    fillColor: color,
    fillOpacity: 0.55,
  });
  circle.addTo(map);
  circle.on("click", () => renderDetail(z));
  markers[z.zone.id] = circle;
}

async function refreshAll() {
  statusEl.textContent = "Loading…";
  try {
    const res = await fetch("/api/refresh_all", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    (data.zones || []).forEach(upsertMarker);
    statusEl.textContent = `Updated ${data.zones?.length ?? 0} zones`;
  } catch (e) {
    statusEl.textContent = "Error — see console";
    console.error(e);
  }
}

document.getElementById("btn-refresh").addEventListener("click", refreshAll);

refreshAll();
