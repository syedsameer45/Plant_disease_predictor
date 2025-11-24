// ================================================
// ADVANCED UI + FIXED VOICE SYSTEM
// ================================================

// DOM refs
const imageInput = document.getElementById('imageInput');
const previewImg = document.getElementById('previewImg');
const fileNameEl = document.getElementById('fileName');
const fileSizeEl = document.getElementById('fileSize');
const fileResEl = document.getElementById('fileRes');
const predictBtn = document.getElementById('predictBtn');
const spinner = document.getElementById('spinner');

const resultCard = document.getElementById('resultCard');
const diseaseTitle = document.getElementById('diseaseTitle');
const confidenceEl = document.getElementById('confidence');
const treatmentSummary = document.getElementById('treatmentSummary');
const treatmentSteps = document.getElementById('treatmentSteps');
const heatmapImg = document.getElementById('heatmapImg');
const lastConvEl = document.getElementById('lastConv');
const downloadOverlay = document.getElementById('downloadOverlay');
const historyList = document.getElementById('historyList');
const saveToHistoryBtn = document.getElementById('saveToHistoryBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

const speakBtn = document.getElementById('speakBtn');
const voiceSelect = document.getElementById('voiceSelect');
const voiceToggle = document.getElementById('voiceToggle');

let currentUpload = null;
let lastOverlayBase64 = null;
let lastResult = null;

const LS_KEY = "plant_ai_history_v1";
const VOICE_PREF_KEY = "voice_enabled";

// ======================================================
// IMAGE UPLOAD + PREVIEW
// ======================================================

const fileDrop = document.getElementById("fileDrop");
["dragenter", "dragover"].forEach(ev =>
  fileDrop.addEventListener(ev, e => {
    e.preventDefault();
    fileDrop.classList.add("drag");
  })
);

["dragleave", "drop"].forEach(ev =>
  fileDrop.addEventListener(ev, e => {
    e.preventDefault();
    fileDrop.classList.remove("drag");
  })
);

fileDrop.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

imageInput.addEventListener("change", e => {
  const f = e.target.files[0];
  if (f) setFile(f);
});

function setFile(file) {
  currentUpload = file;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = (file.size / 1024).toFixed(1) + " KB";

  const reader = new FileReader();
  reader.onload = () => (previewImg.src = reader.result);
  reader.readAsDataURL(file);

  getImageResolution(file)
    .then(res => {
      fileResEl.textContent = res.width + "x" + res.height;
    })
    .catch(() => (fileResEl.textContent = "—"));

  predictBtn.disabled = false;
}

function getImageResolution(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () =>
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

// ======================================================
// RUN PREDICTION
// ======================================================

predictBtn.addEventListener("click", async () => {
  if (!currentUpload) return alert("Choose an image first.");

  predictBtn.disabled = true;
  spinner.classList.remove("hidden");

  try {
    const form = new FormData();
    form.append("file", currentUpload);

    const res = await fetch("/predict", {
      method: "POST",
      body: form
    });

    if (!res.ok) {
      alert("Prediction failed.");
      return;
    }

    const data = await res.json();
    showResult(data);

    lastResult = data;
    lastOverlayBase64 = data.gradcam_base64;

    addToHistory({
      ts: Date.now(),
      name: currentUpload.name,
      disease: data.disease,
      confidence: data.confidence
    });
  } catch (err) {
    console.error(err);
    alert("Prediction error: " + err.message);
  }

  spinner.classList.add("hidden");
  predictBtn.disabled = false;
});

function showResult(data) {
  resultCard.classList.remove("hidden");
  diseaseTitle.textContent = data.disease;
  confidenceEl.textContent = data.confidence;

  treatmentSummary.textContent = data.treatment.summary || "";
  treatmentSteps.innerHTML = "";
  (data.treatment.steps || []).forEach(s => {
    const li = document.createElement("li");
    li.textContent = s;
    treatmentSteps.appendChild(li);
  });

  heatmapImg.src = "data:image/jpeg;base64," + data.gradcam_base64;
  lastConvEl.textContent = data.last_conv || "-";

  downloadOverlay.href = "data:image/jpeg;base64," + data.gradcam_base64;
  downloadOverlay.download = "gradcam.jpg";
}

// ======================================================
// =========== FIXED VOICE SYSTEM =======================
// ======================================================

// Load saved preference
voiceToggle.checked =
  localStorage.getItem(VOICE_PREF_KEY) === "1" || localStorage.getItem(VOICE_PREF_KEY) === null;

voiceToggle.addEventListener("change", () => {
  localStorage.setItem(VOICE_PREF_KEY, voiceToggle.checked ? "1" : "0");
});

// STOP all audio
function stopAllAudio() {
  try {
    window.speechSynthesis.cancel();
  } catch {}

  if (window.currentAudioObj) {
    window.currentAudioObj.pause();
    window.currentAudioObj.src = "";
    window.currentAudioObj = null;
  }
}

// Build text for speaking
function buildSpeechText(data) {
  let parts = [];

  parts.push(`Disease detected: ${data.disease}.`);
  parts.push(`Confidence: ${(parseFloat(data.confidence) * 100).toFixed(1)} percent.`);

  if (data.treatment.summary)
    parts.push(`Summary: ${data.treatment.summary}.`);

  if (data.treatment.steps) {
    parts.push("Recommended steps:");
    data.treatment.steps.forEach((s, idx) => {
      parts.push(`Step ${idx + 1}: ${s}.`);
    });
  }

  return parts.join(" ");
}

// Translate API
async function translateText(text, lang) {
  try {
    const params = new URLSearchParams();
    params.append("text", text);
    params.append("target", lang);

    const res = await fetch("/translate", {
      method: "POST",
      body: params,
      headers: { "Content-Type": "application/x-www-form-urlencoded" }
    });

    const js = await res.json();
    return js.translated || text;
  } catch {
    return text;
  }
}

// Server TTS
async function serverSpeak(text, lang) {
  stopAllAudio();

  const params = new URLSearchParams();
  params.append("text", text);
  params.append("lang", lang);

  const res = await fetch("/tts", {
    method: "POST",
    body: params,
    headers: { "Content-Type": "application/x-www-form-urlencoded" }
  });

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  const audio = new Audio(url);
  window.currentAudioObj = audio;
  audio.play();
}

// Client-side TTS for English only
function clientSpeak(text, lang) {
  stopAllAudio();

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = lang;
  utter.rate = 0.95;

  window.speechSynthesis.speak(utter);
}

// MASTER SPEAK FUNCTION (fixed)
async function speakResult(result) {
  if (!result) return;

  // FIX ✔ Completely turn off voice
  if (!voiceToggle.checked) {
    stopAllAudio();
    return;
  }

  stopAllAudio(); // FIX double audio

  const lang = voiceSelect.value;
  const englishText = buildSpeechText(result);

  // Telugu or Hindi → server TTS ONLY
  if (lang === "te" || lang === "hi") {
    const translated = await translateText(englishText, lang);
    return serverSpeak(translated, lang);
  }

  // English → browser TTS
  clientSpeak(englishText, lang);
}

speakBtn.addEventListener("click", () => {
  if (!lastResult) return alert("Analyze an image first.");
  speakResult(lastResult);
});

// ======================================================
// HISTORY (No Base64 Saved)
// ======================================================

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory(list) {
  localStorage.setItem(LS_KEY, JSON.stringify(list));
}

function addToHistory(entry) {
  const list = loadHistory();
  list.unshift(entry);
  if (list.length > 12) list.pop();
  saveHistory(list);
  renderHistory();
}

function renderHistory() {
  const list = loadHistory();
  historyList.innerHTML = "";

  if (!list.length) {
    historyList.innerHTML = "<div class='muted small'>No history yet</div>";
    return;
  }

  list.forEach(item => {
    const div = document.createElement("div");
    div.className = "history-item";

    div.innerHTML = `
      <div>
        <strong>${item.name}</strong>
        <div class="meta">${item.disease} • ${item.confidence}</div>
      </div>
      <div>
        <button class="btn small">View</button>
      </div>
    `;

    div.querySelector("button").onclick = () => {
      diseaseTitle.textContent = item.disease;
      confidenceEl.textContent = item.confidence;
      resultCard.classList.remove("hidden");
    };

    historyList.appendChild(div);
  });
}

clearHistoryBtn.addEventListener("click", () => {
  if (confirm("Clear history?")) {
    localStorage.removeItem(LS_KEY);
    renderHistory();
  }
});

renderHistory();
