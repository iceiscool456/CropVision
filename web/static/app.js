"use strict";

const STATUS_COLORS = {
  HEALTHY: "#2a9d57",
  DISEASED: "#e23b3b",
  UNKNOWN: "#8a8f98",
};
const LIVE_MAX_WIDTH = 800;

let speciesEnabled = false;
let deepEnabled = false;

/* ── helpers ─────────────────────────────────────────────── */
function el(id) {
  return document.getElementById(id);
}

function statusClass(status) {
  return status === "HEALTHY" ? "is-healthy" : status === "DISEASED" ? "is-diseased" : "is-unknown";
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/* Draw health-colored detection boxes onto a canvas aligned over the uploaded
   image. Scales from native pixels to rendered size. */
function drawOverlay(canvas, dispEl, result) {
  const srcW = dispEl.naturalWidth || dispEl.videoWidth;
  const srcH = dispEl.naturalHeight || dispEl.videoHeight;
  const dispW = dispEl.clientWidth;
  const dispH = dispEl.clientHeight;
  const ctx = canvas.getContext("2d");
  canvas.width = dispW;
  canvas.height = dispH;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!srcW || !srcH || !dispW || !dispH || !result) return;

  const sx = dispW / srcW;
  const sy = dispH / srcH;
  ctx.lineWidth = 3;
  ctx.font = "600 15px -apple-system, Segoe UI, Roboto, sans-serif";
  ctx.textBaseline = "top";

  (result.detections || []).forEach((d, i) => {
    const h = result.health[i];
    const color = h ? STATUS_COLORS[h.status] : STATUS_COLORS.UNKNOWN;
    const [x1, y1, x2, y2] = d.box;
    const rx = x1 * sx, ry = y1 * sy, rw = (x2 - x1) * sx, rh = (y2 - y1) * sy;
    ctx.strokeStyle = color;
    ctx.strokeRect(rx, ry, rw, rh);
    const label = h ? h.text : d.class_name;
    if (label) {
      const padX = 6, th = 22;
      const tw = ctx.measureText(label).width + padX * 2;
      const ly = Math.max(0, ry - th);
      ctx.fillStyle = color;
      ctx.fillRect(rx, ly, tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, rx + padX, ly + 3);
    }
  });
}

function deepBlockHtml(deep) {
  if (!deep) return "";
  if (deep.loading) {
    return `<div class="result-block deep-block"><h4>🔬 Accurate diagnosis (Plant.id)</h4><p class="muted"><span class="spinner"></span>Diagnosing…</p></div>`;
  }
  let html = `<div class="result-block deep-block"><h4>🔬 Accurate diagnosis (Plant.id)</h4>`;
  const cls = deep.is_healthy ? "is-healthy" : "is-diseased";
  const label = deep.is_healthy ? "Looks healthy" : "Disease / problem detected";
  html += `<div class="health-card ${cls}"><span class="dot ${cls}"></span><span class="label">${label}</span></div>`;
  if (deep.suggestions && deep.suggestions.length) {
    deep.suggestions.forEach((s) => {
      const pct = Math.round(s.probability * 100);
      html += `<div class="disease-item">
        <div class="name"><span>${escapeHtml(s.name)}</span><span>${pct}%</span></div>
        <div class="bar"><span style="width:${Math.max(pct, 2)}%"></span></div>
        ${s.treatment ? `<p class="treatment">${escapeHtml(s.treatment)}</p>` : ""}
      </div>`;
    });
  } else if (!deep.is_healthy) {
    html += `<p class="muted">No specific disease identified.</p>`;
  }
  html += `</div>`;
  return html;
}

function renderResults(container, result, { loading, showHealth = true, deep } = {}) {
  if (loading) {
    container.innerHTML = `<p class="muted"><span class="spinner"></span>Analyzing…</p>`;
    return;
  }

  let html = deepBlockHtml(deep);

  if (showHealth) {
    html += `<div class="result-block"><h4>Quick health (local)</h4>`;
    const health = ((result && result.health) || []).filter((h) => h);
    if (!health.length) {
      html += `<p class="muted">No plant analyzed yet.</p>`;
    } else {
      health.forEach((h) => {
        html += `<div class="health-card ${statusClass(h.status)}">
          <span class="dot ${statusClass(h.status)}"></span>
          <span class="label">${escapeHtml(h.text)}</span>
        </div>`;
      });
      if (deepEnabled && !deep) {
        html += `<p class="muted" style="margin-top:6px">For an accurate, specific diagnosis, run <strong>Deep diagnosis</strong>.</p>`;
      }
    }
    html += `</div>`;
  }

  html += `<div class="result-block"><h4>Species (Pl@ntNet)</h4>`;
  if (!speciesEnabled) {
    html += `<p class="muted">Species ID unavailable — no API key configured.</p>`;
  } else if (!result || !result.species.length) {
    html += `<p class="muted">Press “Identify species” to look up the plant.</p>`;
  } else {
    result.species.forEach((s) => {
      const pct = Math.round(s.score * 100);
      html += `<div class="species-row">
        <div class="name"><span>${escapeHtml(s.common_name || s.scientific_name)}</span><span>${pct}%</span></div>
        <div class="sci">${escapeHtml(s.scientific_name)}</div>
        <div class="bar"><span style="width:${pct}%"></span></div>
      </div>`;
    });
  }
  html += `</div>`;

  container.innerHTML = html;
}

async function postAnalyze(blob, { identifySpecies }) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  fd.append("identify_species", identifySpecies ? "true" : "false");
  const res = await fetch("/api/analyze", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  return res.json();
}

async function postDeep(blob) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  const res = await fetch("/api/deep-diagnose", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  return res.json();
}

async function updateCredits() {
  if (!deepEnabled) return;
  let text = "";
  let low = false;
  try {
    const res = await fetch("/api/credits");
    const d = await res.json();
    if (d.available && d.remaining != null) {
      const rem = Math.round(d.remaining);
      const total = d.total != null ? Math.round(d.total) : null;
      text = `🔬 ${rem}${total != null ? " / " + total : ""} deep-diagnosis credits left`;
      low = rem <= 5;
    }
  } catch (e) {
    /* leave badge hidden on failure */
  }
  document.querySelectorAll(".credit-badge").forEach((b) => {
    b.textContent = text;
    b.hidden = !text;
    b.classList.toggle("is-low", low);
  });
}

/* ── tabs ────────────────────────────────────────────────── */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("is-active"));
    tab.classList.add("is-active");
    el(`panel-${tab.dataset.tab}`).classList.add("is-active");
    if (tab.dataset.tab !== "live") stopCamera();
  });
});

/* ── API status / crops ──────────────────────────────────── */
async function loadStatus() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    speciesEnabled = data.species_enabled;
    const badge = el("apiStatus");
    if (data.species_enabled) {
      badge.textContent = "Models ready · Species ID enabled";
      badge.className = "badge badge--ok";
    } else {
      badge.textContent = "Models ready · Species ID off (no API key)";
      badge.className = "badge badge--warn";
    }
    const chips = el("cropChips");
    chips.innerHTML = (data.supported_crops || [])
      .map((c) => `<span class="chip">${escapeHtml(c)}</span>`)
      .join("");
    el("liveIdentify").disabled = !speciesEnabled;

    deepEnabled = data.deep_enabled;
    if (deepEnabled) {
      el("uploadDeep").hidden = false;
      el("liveDeep").hidden = false;
      updateCredits();
    }
  } catch (e) {
    const badge = el("apiStatus");
    badge.textContent = "Server unreachable";
    badge.className = "badge badge--warn";
  }
}

/* ── Upload flow ─────────────────────────────────────────── */
const dropzone = el("dropzone");
const fileInput = el("fileInput");
const uploadImg = el("uploadImg");
const uploadCanvas = el("uploadCanvas");
let uploadResult = null;
let uploadDeep = null;

function renderUpload(opts) {
  renderResults(el("uploadResults"), uploadResult, { ...opts, deep: uploadDeep });
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  uploadImg.onload = () => {
    dropzone.querySelector(".dropzone__hint").hidden = true;
    dropzone.querySelector(".canvas-wrap").hidden = false;
    el("uploadAnalyze").disabled = false;
    el("uploadReset").disabled = false;
    if (deepEnabled) el("uploadDeep").disabled = false;
    uploadResult = null;
    uploadDeep = null;
    renderUpload();
    drawOverlay(uploadCanvas, uploadImg, null);
  };
  uploadImg.src = url;
  dropzone._file = file;
}

dropzone.addEventListener("click", () => {
  if (dropzone.querySelector(".canvas-wrap").hidden) fileInput.click();
});
fileInput.addEventListener("change", (e) => loadFile(e.target.files[0]));
["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));

el("uploadAnalyze").addEventListener("click", async () => {
  if (!dropzone._file) return;
  renderResults(el("uploadResults"), null, { loading: true });
  el("uploadAnalyze").disabled = true;
  try {
    uploadResult = await postAnalyze(dropzone._file, { identifySpecies: speciesEnabled });
    drawOverlay(uploadCanvas, uploadImg, uploadResult);
    renderUpload();
  } catch (e) {
    el("uploadResults").innerHTML = `<p class="muted">Error: ${escapeHtml(e.message)}</p>`;
  } finally {
    el("uploadAnalyze").disabled = false;
  }
});

el("uploadDeep").addEventListener("click", async () => {
  if (!dropzone._file) return;
  uploadDeep = { loading: true };
  renderUpload();
  el("uploadDeep").disabled = true;
  try {
    uploadDeep = await postDeep(dropzone._file);
  } catch (e) {
    uploadDeep = null;
    el("uploadResults").innerHTML = `<p class="muted">Deep diagnosis error: ${escapeHtml(e.message)}</p>`;
    el("uploadDeep").disabled = false;
    return;
  }
  renderUpload();
  el("uploadDeep").disabled = false;
  updateCredits();
});

el("uploadReset").addEventListener("click", () => {
  dropzone._file = null;
  fileInput.value = "";
  dropzone.querySelector(".dropzone__hint").hidden = false;
  dropzone.querySelector(".canvas-wrap").hidden = true;
  el("uploadAnalyze").disabled = true;
  el("uploadReset").disabled = true;
  el("uploadDeep").disabled = true;
  uploadResult = null;
  uploadDeep = null;
  renderUpload();
});

/* ── Live camera flow ────────────────────────────────────── */
const video = el("video");
let stream = null;
let liveResult = null;
let liveDeep = null;

function renderLiveResults() {
  // Live focuses on species ID + on-demand Deep diagnosis.
  renderResults(el("liveResults"), liveResult, { showHealth: false, deep: liveDeep });
}

function captureBlob() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  const scale = Math.min(1, LIVE_MAX_WIDTH / vw);
  const c = document.createElement("canvas");
  c.width = Math.round(vw * scale);
  c.height = Math.round(vh * scale);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise((resolve) => c.toBlob(resolve, "image/jpeg", 0.8));
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 } },
      audio: false,
    });
  } catch (e) {
    el("liveResults").innerHTML = `<p class="muted">Camera access denied or unavailable.</p>`;
    return;
  }
  video.srcObject = stream;
  await video.play();
  el("livePlaceholder").hidden = true;
  el("camToggle").textContent = "Stop camera";
  el("camToggle").classList.remove("btn--primary");
  el("liveIdentify").disabled = !speciesEnabled;
  if (deepEnabled) el("liveDeep").disabled = false;
  liveResult = null;
  liveDeep = null;
  renderLiveResults();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  el("livePlaceholder").hidden = false;
  el("camToggle").textContent = "Start camera";
  el("camToggle").classList.add("btn--primary");
  el("liveIdentify").disabled = true;
  el("liveDeep").disabled = true;
}

el("camToggle").addEventListener("click", () => {
  if (stream) stopCamera();
  else startCamera();
});

el("liveIdentify").addEventListener("click", async () => {
  if (!stream) return;
  el("liveIdentify").disabled = true;
  el("liveIdentify").textContent = "Identifying…";
  try {
    const blob = await captureBlob();
    const r = await postAnalyze(blob, { identifySpecies: true });
    liveResult = r;
    renderLiveResults();
  } catch (e) {
    /* ignore */
  } finally {
    el("liveIdentify").disabled = !speciesEnabled;
    el("liveIdentify").textContent = "Identify species";
  }
});

el("liveDeep").addEventListener("click", async () => {
  if (!stream) return;
  el("liveDeep").disabled = true;
  liveDeep = { loading: true };
  renderLiveResults();
  try {
    const blob = await captureBlob();
    liveDeep = await postDeep(blob);
  } catch (e) {
    liveDeep = null;
  }
  renderLiveResults();
  el("liveDeep").disabled = false;
  updateCredits();
});

window.addEventListener("beforeunload", stopCamera);
loadStatus();
