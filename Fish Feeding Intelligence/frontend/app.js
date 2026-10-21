const $ = id => document.getElementById(id);
const get = (url, opts) => fetch(url, opts).then(async r => {
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || "Request failed");
  return data;
});

let chosenVideo = null;
let previewURL = null;
let cameraStream = null;
let mediaRecorder = null;
let mediaParts = [];
let recordTimer = null;
let recordRemaining = 30;
let currentResult = null;
let lastFeedAlert = "";

function toast(title, message, icon = "🔔") {
  $("toastTitle").textContent = title;
  $("toastMessage").textContent = message;
  $("toastIcon").textContent = icon;
  $("toast").hidden = false;
}
$("toastClose").onclick = () => $("toast").hidden = true;

function formatNumber(number, decimals = 2) {
  return typeof number === "number" && Number.isFinite(number)
    ? number.toFixed(decimals) : "—";
}

function chooseVideo(file) {
  if (!file) return;
  chosenVideo = file;
  $("videoSelected").hidden = false;
  $("filename").textContent = file.name;
  $("filesize").textContent = (file.size / 1048576).toFixed(1) + " MB";
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $("videoPreview").src = previewURL;
  $("analyzeButton").disabled = false;
  toast("Video ready", "Your recording is ready for analysis.", "🎥");
}

$("videoInput").onchange = e => chooseVideo(e.target.files[0]);

$("uploadTab").onclick = () => {
  $("recordSection").hidden = true;
  $("uploadSection").hidden = false;
  $("uploadTab").classList.add("active");
  $("recordTab").classList.remove("active");
};
$("recordTab").onclick = () => {
  $("recordSection").hidden = false;
  $("uploadSection").hidden = true;
  $("recordTab").classList.add("active");
  $("uploadTab").classList.remove("active");
};

const drop = $("dropZone");
drop.addEventListener("dragover", e => {
  e.preventDefault(); drop.classList.add("drag");
});
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", e => {
  e.preventDefault();
  drop.classList.remove("drag");
  if (e.dataTransfer.files.length) {
    chooseVideo(e.dataTransfer.files[0]);
  }
});

$("confidenceSlider").oninput = () => {
  $("confidenceValue").textContent =
    Number($("confidenceSlider").value).toFixed(2);
};

$("siteSearch").onsubmit = e => {
  e.preventDefault();
  const query = $("searchInput").value.toLowerCase();
  const mapping = [
    ["camera", "studio"],
    ["video", "studio"],
    ["yolo", "studio"],
    ["report", "results"],
    ["count", "results"],
    ["fish", "results"],
    ["speed", "results"],
    ["activity", "results"],
    ["feed", "feeding"],
    ["hungry", "feeding"],
    ["method", "insights"]
  ];
  const found = mapping.find(([key]) => query.includes(key));
  if (found) {
    document.getElementById(found[1]).scrollIntoView({
      behavior: "smooth"
    });
  } else {
    toast("Search", "Try video, feeding, fish, report or speed.", "⌕");
  }
};

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
}

function stopRecording() {
  if (recordTimer) clearInterval(recordTimer);
  recordTimer = null;
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}
$("recordStop").onclick = stopRecording;

$("recordStart").onclick = async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("Camera unsupported", "Use Chrome or Edge on localhost.", "⚠️");
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    $("camera").srcObject = cameraStream;
    $("cameraFallback").hidden = true;

    const options = [
      "video/webm;codecs=vp9", "video/webm;codecs=vp8",
      "video/webm", "video/mp4"
    ];
    const mime = options.find(t => MediaRecorder.isTypeSupported(t));
    mediaRecorder = new MediaRecorder(
      cameraStream, mime ? { mimeType: mime } : {}
    );
    mediaParts = [];

    mediaRecorder.ondataavailable = e => {
      if (e.data.size) mediaParts.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const mimeType = mediaRecorder.mimeType || "video/webm";
      const ext = mimeType.includes("mp4") ? "mp4" : "webm";
      const blob = new Blob(mediaParts, { type: mimeType });
      chooseVideo(new File(
        [blob], `goldfish_live_recording.${ext}`, { type: mimeType }
      ));
      stopCamera();
      $("recordStart").hidden = false;
      $("recordStop").hidden = true;
      $("clock").textContent = "00:30";
    };

    mediaRecorder.start(1000);
    $("recordStart").hidden = true;
    $("recordStop").hidden = false;
    recordRemaining = 30;
    $("clock").textContent = "00:30";

    recordTimer = setInterval(() => {
      recordRemaining -= 1;
      $("clock").textContent = "00:" +
        String(Math.max(0, recordRemaining)).padStart(2, "0");
      if (recordRemaining <= 0) stopRecording();
    }, 1000);
  } catch (error) {
    stopCamera();
    toast("Camera error", error.message, "⚠️");
  }
};

function progress(text, percentage) {
  $("progressPanel").hidden = false;
  $("progressText").textContent = text;
  $("progressNumber").textContent = percentage + "%";
  $("progressFill").style.width = percentage + "%";
}

function makeChart(series) {
  const svg = $("activityChart");
  svg.replaceChildren();
  const valid = series.filter(
    x => Number.isFinite(x.motion) && Number.isFinite(x.t)
  );
  if (!valid.length) return;
  const maxY = Math.max(...valid.map(x => x.motion), 0.0000001);
  const maxT = Math.max(...valid.map(x => x.t), 1);
  const coords = valid.map(x => [
    27 + x.t / maxT * 546,
    215 - x.motion / maxY * 172
  ]);
  const ns = "http://www.w3.org/2000/svg";
  for (let i = 0; i < 4; i++) {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", 25);
    line.setAttribute("x2", 575);
    line.setAttribute("y1", 40 + i * 58);
    line.setAttribute("y2", 40 + i * 58);
    line.setAttribute("stroke", "#dce7e4");
    line.setAttribute("stroke-dasharray", "5,6");
    svg.append(line);
  }
  const path = document.createElementNS(ns, "polyline");
  path.setAttribute(
    "points", coords.map(p => p.join(",")).join(" ")
  );
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#f6a021");
  path.setAttribute("stroke-width", "4");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  for (let i = 0; i < coords.length; i += Math.max(1, Math.floor(coords.length / 12))) {
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", coords[i][0]);
    circle.setAttribute("cy", coords[i][1]);
    circle.setAttribute("r", "4");
    circle.setAttribute("fill", "#0d9172");
    svg.append(circle);
  }
}

function showResults(result) {
  currentResult = result;
  $("resultsEmpty").hidden = true;
  $("resultContent").hidden = false;
  $("downloadReport").disabled = false;
  $("downloadJson").disabled = false;

  const fish = result.fish || {};
  const behaviour = result.behaviour || {};
  const index = behaviour.movement_index;

  $("fishCount").textContent = fish.available
    ? fish.count_median : "—";
  $("fishDescription").textContent = fish.available
    ? `Seen per frame: ${fish.count_min}–${fish.count_max}`
    : (fish.error || "Detector unavailable");

  $("quickness").textContent = formatNumber(index, 0);
  $("quicknessText").textContent = behaviour.movement_level ||
    "No movement reference";
  $("behaviourText").textContent = behaviour.anomaly_level ||
    "Not assessed";
  $("healthText").textContent = "Not assessed";

  $("detectionImage").hidden = !fish.image;
  $("detectionPlaceholder").hidden = !!fish.image;
  if (fish.image) {
    $("detectionImage").src =
      "/api/detection-image/" + result.job_id;
  }
  $("detectionFooter").textContent = fish.available
    ? `${fish.sampled_frames} detection frames analysed · ` +
      `Mean YOLO box confidence: ${formatNumber(fish.average_confidence, 2)} · ` +
      `Only visible detections are counted.`
    : (fish.error || "No YOLO detection image available");

  makeChart(result.timeline || []);

  $("activityExplanation").textContent =
    "This graph shows how whole-image movement changes during " +
    "the recording. Peaks may reflect active fish, water or camera motion.";

  $("quicknessBar").style.width =
    typeof index === "number" ? index + "%" : "0%";
  $("speedExplanation").textContent =
    `Relative activity: ${behaviour.movement_level || "unknown"}. ` +
    "This index compares visible scene movement with your no-feeding " +
    "reference clips. It is NOT fish speed in cm/s.";

  $("simpleExplanation").textContent =
    `YOLO sampled the video and measured visible fish. ` +
    `Behaviour was compared with your recorded reference data. ` +
    `${behaviour.anomaly_level || "Deviation unavailable"}. ` +
    "These measurements do not confirm hunger or medical health.";

  $("resultsSubtitle").textContent =
    `Video: ${result.video.duration_seconds}s · ` +
    `YOLO confidence ${result.settings.confidence} · ` +
    `${result.video.sampled_frames} motion frames`;

  toast(
    "Analysis complete",
    "Your goldfish behaviour report is ready.",
    "✅"
  );
  $("results").scrollIntoView({ behavior: "smooth" });
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

$("analyzeButton").onclick = async () => {
  if (!chosenVideo) return;
  $("analyzeButton").disabled = true;
  try {
    const form = new FormData();
    form.append("video", chosenVideo);
    form.append("confidence", $("confidenceSlider").value);
    form.append("sample_count", $("sampleCount").value);

    progress("Uploading video to AquaFeed AI", 2);
    const task = await get("/api/analyse", {
      method: "POST", body: form
    });

    while (true) {
      const job = await get("/api/jobs/" + task.job_id);
      progress(job.message, job.progress);
      if (job.state === "done") {
        showResults(job.result);
        break;
      }
      if (job.state === "failed") {
        throw new Error(job.message);
      }
      await delay(1200);
    }
  } catch (error) {
    toast("Analysis could not finish", error.message, "⚠️");
    progress("Analysis failed", 0);
  } finally {
    $("analyzeButton").disabled = false;
  }
};

$("downloadReport").onclick = () => {
  if (!currentResult) return;
  window.location.href = "/api/report/" + currentResult.job_id;
};

$("downloadJson").onclick = () => {
  if (!currentResult) return;
  const blob = new Blob(
    [JSON.stringify(currentResult, null, 2)],
    { type: "application/json" }
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "AquaFeed_Analysis_Data.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

async function updateFeeding() {
  try {
    const feed = await get("/api/feeding");
    $("feedingStatus").textContent = feed.label;
    $("feedingIndicator").textContent = feed.is_due
      ? "CHECK DUE" : feed.state === "RECENTLY_FED"
      ? "UPDATED" : "NO HISTORY";

    $("intervalSelect").value = String(feed.interval_hours);

    if (feed.last_fed_at) {
      const last = new Date(feed.last_fed_at);
      const next = new Date(feed.next_check_at);
      $("feedingDetails").textContent =
        `Last recorded feeding: ${last.toLocaleString()}. ` +
        `Next check: ${next.toLocaleString()}.`;
    } else {
      $("feedingDetails").textContent =
        "No feeding has been recorded. When you actually feed " +
        "the fish, use the button to save the time.";
    }

    $("feedingAlert").hidden = !feed.is_due;

    if (feed.is_due && lastFeedAlert !== feed.next_check_at) {
      lastFeedAlert = feed.next_check_at;
      toast(
        "Feeding check due · කෑම දෙන වේලාව!",
        "Review your aquarium and feeding schedule now.",
        "🐠"
      );
    }
    if (!feed.is_due) lastFeedAlert = "";
  } catch (error) {
    $("feedingStatus").textContent = "Planner unavailable";
  }
}

$("saveSchedule").onclick = async () => {
  try {
    await get("/api/feeding/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        interval_hours: Number($("intervalSelect").value)
      })
    });
    await updateFeeding();
    toast("Schedule saved", "Your reminder interval has been updated.", "⏰");
  } catch (error) {
    toast("Could not save schedule", error.message, "⚠️");
  }
};

$("feedNow").onclick = async () => {
  const confirmed = window.confirm(
    "Have you actually fed the fish? " +
    "This will record the feeding time. " +
    "It does not activate a physical feeder."
  );
  if (!confirmed) return;
  try {
    await get("/api/feeding/mark-fed", {
      method: "POST"
    });
    await updateFeeding();
    toast(
      "Feeding recorded",
      "Status updated to Recently Fed. ✅",
      "🍽️"
    );
  } catch (error) {
    toast("Could not record feeding", error.message, "⚠️");
  }
};

async function startSystem() {
  try {
    const data = await get("/api/system");
    $("statusYolo").textContent = data.yolo ? "CONNECTED" : "NOT FOUND";
    $("statusMotion").textContent = data.motion ? "CONNECTED" : "NOT FOUND";
    $("statusAE").textContent =
      data.autoencoder && data.preprocessor
        ? "FILES FOUND" : "NOT FOUND";
    $("modelConnected").textContent =
      data.yolo ? "✓ CONNECTED" : "⚠ CHECK PATH";
  } catch {
    $("modelConnected").textContent = "SERVER OFFLINE";
  }
  await updateFeeding();
}

startSystem();
setInterval(updateFeeding, 60000);
