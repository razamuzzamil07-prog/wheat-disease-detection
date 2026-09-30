/**
 * Wheat Leaf Disease Detection - Frontend logic
 *
 * The page calls the FastAPI backend:
 *   POST {API_BASE}/predict  (multipart/form-data, field name: "file")
 *
 * When the backend serves this page itself (uvicorn backend.main:app),
 * API_BASE resolves to the same origin. If you host the frontend
 * separately, set window.API_BASE before this script runs, e.g.:
 *   <script>window.API_BASE = "http://localhost:8000";</script>
 */

(function () {
  "use strict";

  var API_BASE = window.API_BASE || "";

  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var previewWrap = document.getElementById("preview-wrap");
  var preview = document.getElementById("preview");
  var predictBtn = document.getElementById("predict-btn");
  var clearBtn = document.getElementById("clear-btn");
  var status = document.getElementById("status");
  var resultCard = document.getElementById("result-card");

  var selectedFile = null;

  // ---------- helpers ----------

  function showStatus(message, type) {
    status.textContent = message;
    status.className = "status " + type;
  }

  function hideStatus() {
    status.className = "status hidden";
  }

  function setSelectedFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type)) {
      showStatus("Please choose an image file (JPG, PNG, BMP or WEBP).", "error");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showStatus("File is too large. Maximum size is 10 MB.", "error");
      return;
    }
    selectedFile = file;
    preview.src = URL.createObjectURL(file);
    previewWrap.classList.remove("hidden");
    resultCard.classList.add("hidden");
    hideStatus();
  }

  function clearAll() {
    selectedFile = null;
    fileInput.value = "";
    previewWrap.classList.add("hidden");
    resultCard.classList.add("hidden");
    hideStatus();
  }

  // ---------- upload interactions ----------

  dropzone.addEventListener("click", function () { fileInput.click(); });
  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", function () {
    setSelectedFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", function (e) {
    if (e.dataTransfer.files.length) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  });

  clearBtn.addEventListener("click", clearAll);

  // ---------- prediction ----------

  predictBtn.addEventListener("click", function () {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    showStatus("Analyzing image, please wait...", "loading");

    var formData = new FormData();
    formData.append("file", selectedFile);

    fetch(API_BASE + "/predict", { method: "POST", body: formData })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) {
            throw new Error(body.detail || "Prediction failed (HTTP " + res.status + ").");
          }
          return body;
        });
      })
      .then(function (result) {
        hideStatus();
        renderResult(result);
      })
      .catch(function (err) {
        showStatus(err.message || "Could not reach the prediction API.", "error");
      })
      .finally(function () {
        predictBtn.disabled = false;
      });
  });

  // ---------- rendering ----------

  function renderResult(result) {
    document.getElementById("result-name").textContent =
      result.display_name || result.label;

    var severityEl = document.getElementById("result-severity");
    var severity = (result.severity || "none").toLowerCase();
    severityEl.textContent =
      severity === "none" ? "No disease" : severity + " severity";
    severityEl.className = "badge " + severity;

    var pct = Math.round((result.confidence || 0) * 1000) / 10;
    document.getElementById("confidence-value").textContent = pct + "%";
    document.getElementById("confidence-fill").style.width = pct + "%";

    document.getElementById("result-description").textContent =
      result.description || "";
    document.getElementById("result-symptoms").textContent =
      result.symptoms || "—";
    document.getElementById("result-treatment").textContent =
      result.treatment || "—";

    var list = document.getElementById("prob-list");
    list.innerHTML = "";
    var entries = Object.entries(result.all_probabilities || {})
      .sort(function (a, b) { return b[1] - a[1]; });
    entries.forEach(function (entry) {
      var li = document.createElement("li");
      var name = document.createElement("span");
      name.className = "prob-name";
      name.textContent = entry[0].replace(/_/g, " ");
      var value = document.createElement("span");
      value.className = "prob-value";
      value.textContent = (entry[1] * 100).toFixed(1) + "%";
      li.appendChild(name);
      li.appendChild(value);
      list.appendChild(li);
    });

    document.getElementById("disclaimer").textContent = result.disclaimer || "";

    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
})();
