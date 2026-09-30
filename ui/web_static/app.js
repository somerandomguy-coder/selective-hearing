/**
 * AcousticLens Web Client
 * WebSocket telemetry consumer and REST command dispatcher
 */

(function () {
  let ws = null;
  let reconnectTimer = null;

  // DOM elements
  const wsStatusDot = document.getElementById("ws-status-dot");
  const wsStatusText = document.getElementById("ws-status-text");

  const valRms = document.getElementById("val-rms");
  const fillRms = document.getElementById("fill-rms");
  const valGain = document.getElementById("val-gain");
  const fillGain = document.getElementById("fill-gain");
  const badgeGateStatus = document.getElementById("badge-gate-status");

  const valSim = document.getElementById("val-sim");
  const fillSim = document.getElementById("fill-sim");
  const markerThreshold = document.getElementById("marker-threshold");
  const rangeThreshold = document.getElementById("range-threshold");
  const valThreshold = document.getElementById("val-threshold");

  const btnTriggerEnroll = document.getElementById("btn-trigger-enroll");
  const textEnrollProgress = document.getElementById("text-enroll-progress");

  const badgeOverlap = document.getElementById("badge-overlap");
  const speakerMatrix = document.getElementById("speaker-matrix");

  // Mode buttons
  const modeTabs = document.querySelectorAll(".mode-tab");

  // Initialize 8 speaker slots
  function initSpeakerSlots() {
    speakerMatrix.innerHTML = "";
    for (let i = 0; i < 8; i++) {
      const card = document.createElement("div");
      card.className = "speaker-card";
      card.id = `speaker-card-${i}`;
      card.innerHTML = `
        <div class="speaker-header">
          <span class="speaker-name">Speaker ${i}</span>
          <span class="speaker-led" id="spk-led-${i}"></span>
        </div>
        <div class="speaker-prob-bar">
          <div class="speaker-prob-fill" id="spk-prob-fill-${i}" style="width: 0%;"></div>
        </div>
        <div class="speaker-actions">
          <button class="btn-solo ${i === 0 ? 'active' : ''}" id="btn-solo-${i}">Solo</button>
          <button class="btn-mute" id="btn-mute-${i}">Mute</button>
        </div>
      `;
      speakerMatrix.appendChild(card);

      // Event listeners for solo and mute
      card.querySelector(`#btn-solo-${i}`).addEventListener("click", () => toggleSolo(i));
      card.querySelector(`#btn-mute-${i}`).addEventListener("click", () => toggleMute(i));
    }
  }

  // Connect WebSocket
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsStatusDot.className = "status-indicator connected";
      wsStatusText.textContent = "Connected";
      if (reconnectTimer) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        renderState(data);
      } catch (err) {
        console.error("Failed to parse telemetry:", err);
      }
    };

    ws.onclose = () => {
      wsStatusDot.className = "status-indicator";
      wsStatusText.textContent = "Reconnecting...";
      if (!reconnectTimer) {
        reconnectTimer = setInterval(connectWebSocket, 1500);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  // Render State
  function renderState(state) {
    // Active Mode
    modeTabs.forEach((tab) => {
      if (tab.dataset.mode === state.active_mode) {
        tab.classList.add("active");
      } else {
        tab.classList.remove("active");
      }
    });

    // Telemetry: RMS dBFS
    const rms = state.rms_level_db;
    valRms.textContent = `${rms.toFixed(1)} dBFS`;
    // Map -60dB to 0dB -> 0% to 100%
    const rmsPct = Math.max(0, Math.min(100, ((rms + 60) / 60) * 100));
    fillRms.style.width = `${rmsPct}%`;

    // Output Gain
    const gain = state.smoothed_gain;
    valGain.textContent = gain.toFixed(3);
    fillGain.style.width = `${Math.min(100, gain * 100)}%`;

    // Gate Status
    const status = state.gate_status.toLowerCase();
    badgeGateStatus.className = `status-badge ${status}`;
    badgeGateStatus.textContent = state.gate_status;

    // Mode 1: Cosine Similarity
    const sim = state.similarity_score;
    valSim.textContent = sim.toFixed(2);
    fillSim.style.width = `${Math.max(0, Math.min(100, sim * 100))}%`;

    // Threshold Marker
    const thresh = state.similarity_threshold;
    markerThreshold.style.left = `${thresh * 100}%`;
    valThreshold.textContent = thresh.toFixed(2);

    // Enrollment
    if (state.is_enrolling) {
      const pct = Math.round(state.enroll_progress * 100);
      textEnrollProgress.textContent = `Recording calibration... ${pct}%`;
      btnTriggerEnroll.disabled = true;
    } else {
      textEnrollProgress.textContent = "Ready";
      btnTriggerEnroll.disabled = false;
    }

    // Overlap badge
    if (state.overlap_detected) {
      badgeOverlap.classList.remove("hidden");
    } else {
      badgeOverlap.classList.add("hidden");
    }

    // 8 Speaker slots
    if (state.speaker_channels) {
      state.speaker_channels.forEach((ch) => {
        const card = document.getElementById(`speaker-card-${ch.id}`);
        const probFill = document.getElementById(`spk-prob-fill-${ch.id}`);
        const btnSolo = document.getElementById(`btn-solo-${ch.id}`);
        const btnMute = document.getElementById(`btn-mute-${ch.id}`);

        if (card && probFill) {
          probFill.style.width = `${Math.round(ch.prob * 100)}%`;

          if (ch.mute) {
            card.className = "speaker-card muted-voice";
          } else if (ch.active) {
            card.className = "speaker-card active-voice";
          } else {
            card.className = "speaker-card";
          }

          if (ch.solo) {
            btnSolo.classList.add("active");
          } else {
            btnSolo.classList.remove("active");
          }

          if (ch.mute) {
            btnMute.classList.add("active");
          } else {
            btnMute.classList.remove("active");
          }
        }
      });
    }
  }

  // REST API Actions
  async function setMode(mode) {
    await fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
  }

  async function toggleSolo(speakerId) {
    const btn = document.getElementById(`btn-solo-${speakerId}`);
    const isSolo = btn.classList.contains("active");
    await fetch("/api/solo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker_id: speakerId, solo: !isSolo }),
    });
  }

  async function toggleMute(speakerId) {
    const btn = document.getElementById(`btn-mute-${speakerId}`);
    const isMute = btn.classList.contains("active");
    await fetch("/api/mute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker_id: speakerId, mute: !isMute }),
    });
  }

  async function setThreshold(threshold) {
    await fetch("/api/threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold: parseFloat(threshold) }),
    });
  }

  async function triggerEnroll() {
    await fetch("/api/enroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ duration_s: 5.0 }),
    });
  }

  // Event Listeners
  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  });

  rangeThreshold.addEventListener("input", (e) => {
    valThreshold.textContent = parseFloat(e.target.value).toFixed(2);
    setThreshold(e.target.value);
  });

  btnTriggerEnroll.addEventListener("click", triggerEnroll);

  // Initialize
  initSpeakerSlots();
  connectWebSocket();
})();
