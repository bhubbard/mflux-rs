document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const modelSelect = document.getElementById("model-variant");
  const stepCountInput = document.getElementById("step-count");
  const stepCountVal = document.getElementById("step-count-val");
  const resSelect = document.getElementById("img-resolution");
  const scheduleTbody = document.getElementById("schedule-tbody");
  const canvas = document.getElementById("trajectory-canvas");
  const ctx = canvas.getContext("2d");

  // RoPE elements
  const posXInput = document.getElementById("pos-x");
  const posYInput = document.getElementById("pos-y");
  const posTInput = document.getElementById("pos-t");
  const posXVal = document.getElementById("pos-x-val");
  const posYVal = document.getElementById("pos-y-val");
  const posTVal = document.getElementById("pos-t-val");
  const axis0Bars = document.getElementById("axis0-bars");
  const axis1Bars = document.getElementById("axis1-bars");
  const axis2Bars = document.getElementById("axis2-bars");

  // Calc elements
  const calcW = document.getElementById("calc-w");
  const calcH = document.getElementById("calc-h");
  const calcPrec = document.getElementById("calc-prec");
  const statSpatial = document.getElementById("stat-spatial");
  const statPatches = document.getElementById("stat-patches");
  const statTokens = document.getElementById("stat-tokens");
  const statVram = document.getElementById("stat-vram");

  // Flow-matching math identical to Rust & upstream Python
  function computeEmpiricalMu(imageSeqLen, numSteps) {
    const a1 = 8.73809524e-5;
    const b1 = 1.89833333;
    const a2 = 0.00016927;
    const b2 = 0.45666666;
    if (imageSeqLen > 4300) {
      return a2 * imageSeqLen + b2;
    }
    const m_200 = a2 * imageSeqLen + b2;
    const m_10 = a1 * imageSeqLen + b1;
    const a = (m_200 - m_10) / 190.0;
    const b = m_200 - 200.0 * a;
    return a * numSteps + b;
  }

  function timeShiftExponential(mu, sigmaPower, t) {
    const expMu = Math.exp(mu);
    return expMu / (expMu + Math.pow(1.0 / t - 1.0, sigmaPower));
  }

  function stretchToTerminal(sigmas, shiftTerminal = 0.02) {
    const oneMinus = sigmas.map(s => 1.0 - s);
    const scaleFactor = oneMinus[oneMinus.length - 1] / (1.0 - shiftTerminal);
    return oneMinus.map(oms => 1.0 - oms / scaleFactor);
  }

  function computeSchedule(numSteps, seqLen = 4096) {
    if (numSteps <= 1) {
      return { sigmas: [1.0, 0.0], timesteps: [1000.0] };
    }

    const sigmaMin = 1.0 / 1000.0;
    const sigmaMax = 1.0;
    const timestepsLinear = [];
    for (let i = 0; i < numSteps; i++) {
      timestepsLinear.push(sigmaMax * 1000.0 - (i * (sigmaMax - sigmaMin) * 1000.0) / (numSteps - 1));
    }

    const sigmasLinear = timestepsLinear.map(t => t / 1000.0);
    const sigmasShifted = sigmasLinear.map(s => timeShiftExponential(1.0, 1.0, s));
    const sigmasFinal = stretchToTerminal(sigmasShifted, 0.02);
    const timesteps = sigmasFinal.map(s => s * 1000.0);

    const fullSigmas = [...sigmasFinal, 0.0];
    return { sigmas: fullSigmas, timesteps };
  }

  function updateSimulation() {
    const numSteps = parseInt(stepCountInput.value, 10);
    stepCountVal.textContent = numSteps;

    const res = parseInt(resSelect.value, 10);
    const seqLen = (res / 16) * (res / 16);

    const { sigmas, timesteps } = computeSchedule(numSteps, seqLen);

    // Update Table
    scheduleTbody.innerHTML = "";
    for (let i = 0; i < numSteps; i++) {
      const s1 = sigmas[i];
      const s2 = sigmas[i + 1];
      const dt = (s2 - s1).toFixed(4);
      const t = timesteps[i].toFixed(1);

      let phase = "Coarse Shapes";
      if (i > numSteps * 0.7) phase = "High-Frequency Details";
      else if (i > numSteps * 0.3) phase = "Semantic Structures";

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>Step ${i + 1}</strong></td>
        <td><code>${s1.toFixed(4)} → ${s2.toFixed(4)}</code></td>
        <td><code>${t}</code></td>
        <td style="color: #60a5fa">${dt}</td>
        <td><span class="badge">${phase}</span></td>
      `;
      scheduleTbody.appendChild(tr);
    }

    // Draw Canvas
    renderCanvas(sigmas);
  }

  function renderCanvas(sigmas) {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.beginPath();
    ctx.moveTo(40, 20);
    ctx.lineTo(40, h - 30);
    ctx.lineTo(w - 20, h - 30);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText("σ = 1.0 (Pure Noise)", 45, 30);
    ctx.fillText("σ = 0.0 (Clean Image)", 45, h - 35);
    ctx.fillText("Steps →", w - 70, h - 12);

    // Plot trajectory line
    const n = sigmas.length;
    const stepWidth = (w - 70) / (n - 1);
    const plotHeight = h - 60;

    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let i = 0; i < n; i++) {
      const x = 40 + i * stepWidth;
      const y = (h - 30) - (sigmas[i] * plotHeight);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Plot step points
    for (let i = 0; i < n; i++) {
      const x = 40 + i * stepWidth;
      const y = (h - 30) - (sigmas[i] * plotHeight);
      ctx.fillStyle = i === 0 ? "#ef4444" : (i === n - 1 ? "#10b981" : "#a855f7");
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // RoPE Frequency Bars
  function updateRoPE() {
    const x = parseInt(posXInput.value, 10);
    const y = parseInt(posYInput.value, 10);
    const t = parseInt(posTInput.value, 10);

    posXVal.textContent = x;
    posYVal.textContent = y;
    posTVal.textContent = t;

    renderAxisBars(axis0Bars, 8, t, 10000.0, 16);
    renderAxisBars(axis1Bars, 28, y, 10000.0, 56);
    renderAxisBars(axis2Bars, 28, x, 10000.0, 56);
  }

  function renderAxisBars(container, halfDim, pos, theta, totalDim) {
    container.innerHTML = "";
    for (let i = 0; i < halfDim; i++) {
      const scale = (2 * i) / totalDim;
      const omega = 1.0 / Math.pow(theta, scale);
      const angle = pos * omega;
      const cosVal = Math.cos(angle);
      const heightPercent = Math.max(10, Math.min(100, (cosVal * 0.5 + 0.5) * 100));

      const bar = document.createElement("div");
      bar.className = "freq-bar-col";
      bar.style.height = `${heightPercent}%`;
      bar.title = `Dim ${2*i}: ${cosVal.toFixed(3)}`;
      container.appendChild(bar);
    }
  }

  // Token & VRAM Calculator
  function updateCalc() {
    const w = parseInt(calcW.value, 10) || 1024;
    const h = parseInt(calcH.value, 10) || 1024;
    const prec = parseInt(calcPrec.value, 10) || 4;

    const latH = Math.floor(h / 8);
    const latW = Math.floor(w / 8);
    const patchH = Math.floor(h / 16);
    const patchW = Math.floor(w / 16);
    const tokens = patchH * patchW;

    statSpatial.textContent = `${latH} × ${latW}`;
    statPatches.textContent = `${patchH} × ${patchW}`;
    statTokens.textContent = tokens.toLocaleString();

    // 12B params: FP16 is 24GB, Q8 is 12GB, Q4 is 6GB base + activations
    let baseWeightGb = 12 * (prec / 8);
    let actGb = (tokens * 3072 * 4 * 57 * 2) / (1024 * 1024 * 1024);
    let totalGb = (baseWeightGb + Math.min(actGb, 3.5)).toFixed(1);

    statVram.textContent = `~${totalGb} GB`;
  }

  // Event Listeners
  modelSelect.addEventListener("change", () => {
    if (modelSelect.value === "schnell") {
      stepCountInput.value = 4;
    } else {
      stepCountInput.value = 25;
    }
    updateSimulation();
  });

  stepCountInput.addEventListener("input", updateSimulation);
  resSelect.addEventListener("change", updateSimulation);

  posXInput.addEventListener("input", updateRoPE);
  posYInput.addEventListener("input", updateRoPE);
  posTInput.addEventListener("input", updateRoPE);

  calcW.addEventListener("input", updateCalc);
  calcH.addEventListener("input", updateCalc);
  calcPrec.addEventListener("change", updateCalc);

  // Initial runs
  updateSimulation();
  updateRoPE();
  updateCalc();
});
