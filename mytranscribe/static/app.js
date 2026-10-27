/* The page: upload, watch, cancel, download. Plain DOM, no build step. */

const STATE_LABEL = {
  queued: "menunggu",
  running: "berjalan",
  finished: "selesai",
  cancelled: "dibatalkan",
  failed: "gagal",
};

const jobs = new Map(); // id -> { snapshot, source }
const streams = new Map(); // id -> EventSource

const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file");
const startButton = document.querySelector("#start");
const uploadError = document.querySelector("#upload-error");
const list = document.querySelector("#jobs");
const template = document.querySelector("#job-template");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  uploadError.hidden = true;
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    showUploadError("Pilih satu rekaman dulu.");
    return;
  }

  const format = new FormData(form).get("format") || "txt";
  const body = new FormData();
  body.append("file", file);
  body.append("format", format);

  startButton.disabled = true;
  startButton.textContent = "Mengunggah\u2026";
  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    if (!response.ok) {
      showUploadError(await readError(response));
      return;
    }
    const snapshot = await response.json();
    remember(snapshot);
    form.reset();
    render();
    subscribe(snapshot.id);
  } finally {
    startButton.disabled = false;
    startButton.textContent = "Mulai transkripsi";
  }
});

fileInput.addEventListener("change", () => {
  uploadError.hidden = true;
  inspectChosenFile();
});

// Story 4: know what you picked before committing. The file is probed on the
// server — the browser cannot read a duration for most containers — and the
// scratch copy is deleted immediately, so nothing is queued by looking.
async function inspectChosenFile() {
  const panel = document.querySelector("#inspect");
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    panel.hidden = true;
    return;
  }

  panel.hidden = false;
  document.querySelector("#inspect-name").textContent = file.name;
  document.querySelector("#inspect-duration").textContent = "membaca\u2026";
  document.querySelector("#inspect-kind").textContent = "\u2014";
  document.querySelector("#inspect-estimate").textContent = "membaca\u2026";

  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/inspect", { method: "POST", body });
    if (!response.ok) {
      const message = await readError(response);
      panel.hidden = true;
      showUploadError(message);
      return;
    }
    const facts = await response.json();
    document.querySelector("#inspect-duration").textContent = facts.duration_label;
    document.querySelector("#inspect-kind").textContent =
      `${facts.original_audio.codec} \u00b7 ${facts.original_audio.sample_rate} Hz \u00b7 ${facts.original_audio.channels} ch`;
    document.querySelector("#inspect-estimate").textContent = facts.estimate.range_label;
    if (facts.estimate.warning) showUploadError(facts.estimate.warning);
  } catch {
    // A failed probe must not block the upload; the Job will report properly.
    panel.hidden = true;
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    refreshAll();
  }
});

async function refreshAll() {
  try {
    const response = await fetch("/api/jobs");
    if (!response.ok) return;
    const payload = await response.json();
    for (const snapshot of payload.jobs || []) {
      remember(snapshot);
    }
    render();
  } catch {
    // A failed refresh must not clear the page; keep showing what we have.
  }
}

function subscribe(id) {
  if (streams.has(id)) return;
  const stream = new EventSource(`/api/jobs/${id}/events`);
  streams.set(id, stream);
  stream.onmessage = (message) => {
    try {
      remember(JSON.parse(message.data));
      render();
    } catch {
      // Ignore a malformed frame rather than tearing down the view.
    }
  };
  stream.addEventListener("end", () => closeStream(id));
  stream.onerror = () => {
    // On a finished Job the stream closing is expected, not an error.
    const entry = jobs.get(id);
    if (entry && isTerminal(entry.snapshot.state)) closeStream(id);
  };
}

function closeStream(id) {
  const stream = streams.get(id);
  if (stream) {
    stream.close();
    streams.delete(id);
  }
}

function remember(snapshot) {
  const entry = jobs.get(snapshot.id) || {};
  // The server's snapshot is the source of truth; never invent progress locally.
  entry.snapshot = snapshot;
  jobs.set(snapshot.id, entry);
  if (isTerminal(snapshot.state)) closeStream(snapshot.id);
}

function isTerminal(state) {
  return state === "finished" || state === "cancelled" || state === "failed";
}

function render() {
  const ordered = [...jobs.values()].sort((a, b) => {
    const left = a.snapshot.created_at || "";
    const right = b.snapshot.created_at || "";
    return left < right ? 1 : left > right ? -1 : 0;
  });

  for (const { snapshot } of ordered) {
    let card = document.getElementById(`job-${snapshot.id}`);
    if (!card) {
      card = template.content.firstElementChild.cloneNode(true);
      card.id = `job-${snapshot.id}`;
      wire(card, snapshot.id);
      list.prepend(card);
    }
    fill(card, snapshot);
  }
}

function wire(card, id) {
  card.querySelector(".cancel").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Membatalkan\u2026";
    try {
      const response = await fetch(`/api/jobs/${id}/cancel`, { method: "POST" });
      if (response.ok) {
        // The response is a snapshot, so it can be shown immediately — then the
        // stream carries the Job to its final state.
        remember(await response.json());
        render();
        subscribe(id);
      }
    } finally {
      button.disabled = false;
      button.textContent = "Batalkan";
    }
  });
  card.querySelector(".download").addEventListener("click", () => {
    // A deliberate click, so the browser cannot quietly drop a finished result.
    card.querySelector(".download").dataset.wasDownloaded = "1";
  });
}

function fill(card, snapshot) {
  card.querySelector(".job-name").textContent = snapshot.original_name;
  card.querySelector(".job-state").textContent = STATE_LABEL[snapshot.state] || snapshot.state;

  const source = snapshot.source || {};
  card.querySelector(".job-source").textContent = describeSource(snapshot, source);

  const queue = card.querySelector(".job-queue");
  if (snapshot.state === "queued" && snapshot.queue_position) {
    queue.hidden = false;
    queue.textContent = `Menunggu giliran, posisi ${snapshot.queue_position} di antrean.`;
  } else {
    queue.hidden = true;
  }

  const percent = Math.round((snapshot.progress || 0) * 100);
  const bar = card.querySelector(".bar");
  bar.setAttribute("aria-valuenow", String(percent));
  card.querySelector(".bar-fill").style.width = `${percent}%`;

  card.querySelector(".job-progress").textContent = describeProgress(snapshot);

  const estimate = snapshot.estimate || {};
  const estimateNote = card.querySelector(".job-estimate");
  estimateNote.textContent = describeEstimate(snapshot, estimate);

  const warning = card.querySelector(".job-warning");
  if (estimate.warning) {
    warning.hidden = false;
    warning.textContent = estimate.warning;
  } else {
    warning.hidden = true;
  }

  const error = card.querySelector(".job-error");
  if (snapshot.error) {
    error.hidden = false;
    error.textContent = snapshot.error;
  } else if (snapshot.expired) {
    error.hidden = false;
    error.textContent = "Hasil ini sudah kedaluwarsa, jadi tidak bisa diunduh lagi.";
  } else {
    error.hidden = true;
  }

  const download = card.querySelector(".download");
  if (snapshot.state === "finished" && !snapshot.expired) {
    download.hidden = false;
    download.href = `/api/jobs/${snapshot.id}/download`;
    download.setAttribute("download", "");
    download.textContent = `Unduh .${snapshot.format}`;
  } else {
    download.hidden = true;
    download.removeAttribute("href");
  }

  const cancel = card.querySelector(".cancel");
  cancel.hidden = isTerminal(snapshot.state);
  cancel.disabled = Boolean(snapshot.cancel_requested);
  cancel.textContent = snapshot.cancel_requested ? "Menghentikan\u2026" : "Batalkan";
  cancel.title = "Membatalkan bisa butuh sampai 30 detik bila sedang berjalan.";

  const partial = card.querySelector(".job-partial");
  if (snapshot.partial_text && !isTerminal(snapshot.state)) {
    partial.hidden = false;
    partial.querySelector(".job-partial-text").textContent = snapshot.partial_text;
  } else {
    partial.hidden = true;
  }
}

function describeSource(snapshot, source) {
  const parts = [];
  if (snapshot.duration_label) parts.push(snapshot.duration_label);
  if (source.codec) parts.push(source.codec);
  if (source.sample_rate) parts.push(`${source.sample_rate} Hz`);
  if (source.channels) parts.push(`${source.channels} ch`);
  parts.push(`format .${snapshot.format}`);
  parts.push(`mode ${snapshot.mode}`);
  return parts.join(" \u00b7 ");
}

function describeProgress(snapshot) {
  if (snapshot.cancel_requested) {
    // Stopping is not instant, and silence during the wait reads as a broken button.
    return "Sedang dihentikan. Berhenti di akhir bagian 30 detik yang sedang diproses.";
  }
  if (snapshot.state === "queued") return "Belum mulai.";
  if (snapshot.state === "finished") return "Selesai.";
  if (snapshot.state === "cancelled") return "Dibatalkan.";
  if (snapshot.state === "failed") return "Gagal.";
  const total = snapshot.pieces_total || 0;
  const done = snapshot.pieces_completed || 0;
  if (!total) return "Menyiapkan audio\u2026";
  return `${done} dari ${total} bagian selesai.`;
}

function describeEstimate(snapshot, estimate) {
  if (isTerminal(snapshot.state)) return "";
  if (snapshot.state === "queued" && snapshot.queue_position) {
    return `Perkiraan tunggu: ${estimate.range_label || "menghitung\u2026"} (termasuk antrean).`;
  }
  if (!estimate.seconds) return "Perkiraan tunggu: menghitung\u2026";
  return `Perkiraan sisa waktu: ${estimate.range_label}. Ini perkiraan, bukan janji.`;
}

async function readError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail) && payload.detail.length) {
      return payload.detail.map((item) => item.msg).join(" ");
    }
  } catch {
    // Fall through to the status line.
  }
  return `Unggahan ditolak (${response.status}).`;
}

function showUploadError(message) {
  uploadError.textContent = message;
  uploadError.hidden = false;
}

// A reload must not lose sight of a Job: fetch every known snapshot first, then
// subscribe to anything still moving.
refreshAll().then(() => {
  for (const [id, entry] of jobs) {
    if (!isTerminal(entry.snapshot.state)) subscribe(id);
  }
});