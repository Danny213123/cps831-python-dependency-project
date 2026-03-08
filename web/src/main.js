const state = {
  home: null,
  runs: [],
  selectedRunId: "",
  selectedRun: null,
  activeTab: "home",
  caseSearch: "",
  caseFilter: "all",
  caseDetails: new Map(),
  openCaseIds: new Set(),
  pollTimer: null,
};

const ui = {
  tabButtons: Array.from(document.querySelectorAll(".dashboard-tab")),
  tabHome: document.querySelector("#tab-home"),
  tabRuns: document.querySelector("#tab-runs"),
  runSelect: document.querySelector("#run-select"),
  runCount: document.querySelector("#run-count"),
  lastRefresh: document.querySelector("#last-refresh"),
  selectedRunId: document.querySelector("#selected-run-id"),
  homeTitle: document.querySelector("#home-title"),
  homeSubtitle: document.querySelector("#home-subtitle"),
  homeDescription: document.querySelector("#home-description"),
  homeScope: document.querySelector("#home-scope"),
  homeLocalUrl: document.querySelector("#home-local-url"),
  homeNetworkUrl: document.querySelector("#home-network-url"),
  homeInfoGrid: document.querySelector("#home-info-grid"),
  heroRunTitle: document.querySelector("#hero-run-title"),
  heroRunSubtitle: document.querySelector("#hero-run-subtitle"),
  runInfoGrid: document.querySelector("#run-info-grid"),
  progressLabel: document.querySelector("#progress-label"),
  progressPercent: document.querySelector("#progress-percent"),
  progressFill: document.querySelector("#progress-fill"),
  metricsGrid: document.querySelector("#metrics-grid"),
  perfLine: document.querySelector("#perf-line"),
  researchLine: document.querySelector("#research-line"),
  lastLlmLine: document.querySelector("#last-llm-line"),
  activeCases: document.querySelector("#active-cases"),
  recentActivity: document.querySelector("#recent-activity"),
  casesScroll: document.querySelector("#cases-scroll"),
  caseSearch: document.querySelector("#case-search"),
  caseFilter: document.querySelector("#case-filter"),
  caseRowTemplate: document.querySelector("#case-row-template"),
};

function switchTab(tabId) {
  state.activeTab = tabId;
  for (const button of ui.tabButtons) {
    const active = button.dataset.tab === tabId;
    button.classList.toggle("tab-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  }
  ui.tabHome.classList.toggle("tab-active", tabId === "home");
  ui.tabHome.classList.toggle("tab-hidden", tabId !== "home");
  ui.tabRuns.classList.toggle("tab-active", tabId === "runs");
  ui.tabRuns.classList.toggle("tab-hidden", tabId !== "runs");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatSeconds(value) {
  const seconds = Number(value || 0);
  return `${seconds.toFixed(1)}s`;
}

function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds || 0)));
  const hours = String(Math.floor(total / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const secs = String(total % 60).padStart(2, "0");
  return `${hours}:${minutes}:${secs}`;
}

function formatTimestamp(value) {
  if (!value) {
    return "unknown";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function dependencyPreview(list) {
  const deps = Array.isArray(list) ? list.filter(Boolean) : [];
  if (!deps.length) {
    return "-";
  }
  const preview = deps.slice(0, 3).join(", ");
  return deps.length > 3 ? `${preview}, +${deps.length - 3} more` : preview;
}

function runtimeSummary(config) {
  return [
    `profile=${config.effectiveModelProfile || "default"}`,
    `rag=${config.effectiveRagMode || "pypi"}`,
    `structured=${config.effectiveStructuredPrompting ? "on" : "off"}`,
    `repair=${config.effectiveRepairCycleLimit ?? 0}`,
    `fallback=${config.effectiveCandidateFallbackBeforeRepair ? "on" : "off"}`,
  ].join(" ");
}

function formatSuccessRate(successes, completed) {
  if (!completed) {
    return "n/a";
  }
  return `${((successes / completed) * 100).toFixed(1)}%`;
}

function secondsPerCase(run) {
  if (!run?.completed) {
    return null;
  }
  return Number(run.elapsedSeconds || 0) / Number(run.completed || 1);
}

function formatSecondsPerCase(run) {
  const value = secondsPerCase(run);
  return value == null ? "n/a" : `${value.toFixed(1)}s/case`;
}

function formatEta(run) {
  if (!run?.total || !run?.completed || run.completed >= run.total) {
    return run?.total && run?.completed >= run.total ? "00:00:00" : "n/a";
  }
  const value = secondsPerCase(run);
  if (value == null) {
    return "n/a";
  }
  return formatElapsed(value * (run.total - run.completed));
}

function tokensPerSecond(tokens, durationNs) {
  const count = Number(tokens || 0);
  const duration = Number(durationNs || 0);
  if (count <= 0 || duration <= 0) {
    return null;
  }
  return count / (duration / 1e9);
}

function formatTokensPerSecond(value) {
  return value == null ? "n/a" : `${value.toFixed(1)} tok/s`;
}

function formatOllamaSummary(stats) {
  if (!stats || !Number(stats.calls || 0)) {
    return "waiting for first response";
  }
  const parts = [`${stats.calls} calls`];
  const evalRate = tokensPerSecond(stats.eval_tokens, stats.eval_duration_ns);
  const promptRate = tokensPerSecond(stats.prompt_tokens, stats.prompt_duration_ns);
  if (Number(stats.eval_tokens || 0) > 0 || Number(stats.eval_duration_ns || 0) > 0) {
    parts.push(`out ${stats.eval_tokens || 0} tok @ ${formatTokensPerSecond(evalRate)}`);
  }
  if (Number(stats.prompt_tokens || 0) > 0 || Number(stats.prompt_duration_ns || 0) > 0) {
    parts.push(`prompt ${stats.prompt_tokens || 0} tok @ ${formatTokensPerSecond(promptRate)}`);
  }
  return parts.join(", ");
}

function formatLastLlm(stats) {
  if (!stats || !Number(stats.calls || 0)) {
    return "";
  }
  const model = stats.last_model || "unknown";
  const stage = stats.last_stage || "unknown";
  const rate = formatTokensPerSecond(tokensPerSecond(stats.last_eval_tokens, stats.last_eval_duration_ns));
  return `${stage} / ${model} @ ${rate}`;
}

function kvRows(fields) {
  return fields
    .map(
      ([label, value]) => `
        <div class="kv-row">
          <span class="kv-label">${escapeHtml(label)}</span>
          <span class="kv-value">${escapeHtml(value || "-")}</span>
        </div>
      `,
    )
    .join("");
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setLink(node, value) {
  if (!value || value === "-") {
    node.textContent = "-";
    node.removeAttribute("href");
    return;
  }
  node.textContent = value;
  node.href = value;
}

function renderHome() {
  const home = state.home;
  if (!home) {
    ui.homeTitle.textContent = "APDR Command Center";
    ui.homeSubtitle.textContent = "Run, report, and configure without memorizing commands.";
    ui.homeDescription.textContent = "The web dashboard mirrors the CLI command center on this host.";
    ui.homeScope.textContent = "-";
    setLink(ui.homeLocalUrl, "-");
    setLink(ui.homeNetworkUrl, "-");
    ui.homeInfoGrid.innerHTML = "";
    return;
  }
  ui.homeTitle.textContent = home.title;
  ui.homeSubtitle.textContent = home.subtitle;
  ui.homeDescription.textContent = home.description;
  ui.homeScope.textContent = home.server?.scope || "-";
  setLink(ui.homeLocalUrl, home.server?.localUrl || "-");
  setLink(ui.homeNetworkUrl, home.server?.networkUrl || "-");
  ui.homeInfoGrid.innerHTML = kvRows((home.fields || []).map((field) => [field.label, field.value]));
}

function setSelectedRun(runId) {
  state.selectedRunId = runId;
  state.selectedRun = null;
  state.caseDetails.clear();
  renderRuns();
  loadSelectedRun().catch(renderErrorState);
}

function renderRuns() {
  ui.runCount.textContent = String(state.runs.length);
  ui.runSelect.innerHTML = "";
  if (!state.runs.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No runs available";
    ui.runSelect.appendChild(option);
    ui.runSelect.disabled = true;
    ui.selectedRunId.textContent = "none";
    return;
  }
  ui.runSelect.disabled = false;
  const fragment = document.createDocumentFragment();
  for (const run of state.runs) {
    const option = document.createElement("option");
    option.value = run.runId;
    option.textContent = `${run.runId} [${run.status}] ${run.completed}/${run.total} ${run.successRate.toFixed(1)}%`;
    if (run.runId === state.selectedRunId) {
      option.selected = true;
    }
    fragment.appendChild(option);
  }
  ui.runSelect.appendChild(fragment);
  ui.selectedRunId.textContent = state.selectedRunId || "none";
}

function renderRunInfo(run) {
  const fields = [
    ["Run ID", run.runId],
    ["Version", run.appVersion || state.home?.version || "unknown"],
    ["Target", run.target],
    ["Resolver", run.resolver],
    ["Preset", run.preset],
    ["Research", run.researchBundle],
    ["Prompt", run.promptProfile],
    ["Source", run.benchmarkSource],
    ["Models", run.modelSummary || "unknown"],
    ["Effective", runtimeSummary(run.runtimeConfig || {})],
    ["Ollama", formatOllamaSummary(run.ollamaStats || {})],
    ["Jobs", String(run.jobs || 1)],
    ["Artifacts", run.artifactsDir || "-"],
  ];
  ui.runInfoGrid.innerHTML = kvRows(fields);
}

function renderProgress(run) {
  const percent = run.total ? (run.completed / run.total) * 100 : 0;
  ui.progressLabel.textContent = `${run.completed}/${run.total}`;
  ui.progressPercent.textContent = `( ${percent.toFixed(1)}% )`;
  ui.progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  ui.metricsGrid.innerHTML = `
    <span><span class="kv-label">Successes:</span> <span class="text-green">${escapeHtml(String(run.successes))}</span></span>
    <span><span class="kv-label">Failures:</span> <span class="text-red">${escapeHtml(String(run.failures))}</span></span>
    <span><span class="kv-label">Elapsed:</span> <span class="text-yellow">${escapeHtml(formatElapsed(run.elapsedSeconds))}</span></span>
    <span><span class="kv-label">Success rate:</span> <span class="text-yellow">${escapeHtml(formatSuccessRate(run.successes, run.completed))}</span></span>
    <span><span class="kv-label">Speed:</span> <span class="text-yellow">${escapeHtml(formatSecondsPerCase(run))}</span></span>
    <span><span class="kv-label">ETA:</span> <span class="text-yellow">${escapeHtml(formatEta(run))}</span></span>
  `;
  ui.perfLine.innerHTML = `
    <span class="kv-label">Perf</span>
    <span class="kv-value">
      build ${escapeHtml(formatSeconds(run.dockerBuildSeconds))},
      run ${escapeHtml(formatSeconds(run.dockerRunSeconds))},
      llm ${escapeHtml(formatSeconds(run.llmSeconds))},
      cache hits ${escapeHtml(String(run.imageCacheHits || 0))},
      build skips ${escapeHtml(String(run.buildSkips || 0))}
    </span>
  `;
  const researchFeatures = (run.researchFeatures || []).join(", ");
  ui.researchLine.innerHTML = researchFeatures
    ? `<span class="kv-label">Research feats</span><span class="kv-value">${escapeHtml(researchFeatures)}</span>`
    : "";
  const lastLlm = formatLastLlm(run.ollamaStats || {});
  ui.lastLlmLine.innerHTML = lastLlm
    ? `<span class="kv-label">Last LLM</span><span class="kv-value">${escapeHtml(lastLlm)}</span>`
    : "";
}

function renderActivityList(container, items, emptyText, formatter) {
  container.innerHTML = "";
  if (!items.length) {
    container.innerHTML = `<div class="empty-line">${escapeHtml(emptyText)}</div>`;
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const item of items) {
    const div = document.createElement("div");
    div.className = "bullet-item";
    div.innerHTML = formatter(item);
    fragment.appendChild(div);
  }
  container.appendChild(fragment);
}

function filteredCases() {
  if (!state.selectedRun) {
    return [];
  }
  const active = new Set(state.selectedRun.activeCases || []);
  return state.selectedRun.cases.filter((item) => {
    const haystack = [item.caseId, item.result, item.dependencyPreview, ...(item.dependencies || [])]
      .join(" ")
      .toLowerCase();
    const searchOk = !state.caseSearch || haystack.includes(state.caseSearch);
    if (!searchOk) {
      return false;
    }
    switch (state.caseFilter) {
      case "active":
        return active.has(item.caseId);
      case "pass":
        return item.success;
      case "fail":
        return !item.success;
      case "match":
        return item.pllmMatch === "MATCH";
      case "miss":
        return item.pllmMatch === "MISS";
      default:
        return true;
    }
  });
}

async function ensureCaseDetail(caseId) {
  if (state.caseDetails.has(caseId)) {
    return state.caseDetails.get(caseId);
  }
  const payload = await fetchJson(`/api/runs/${state.selectedRunId}/cases/${caseId}`);
  state.caseDetails.set(caseId, payload);
  return payload;
}

function caseMatchClass(value) {
  if (value === "MATCH") {
    return "text-yellow";
  }
  if (value === "MISS") {
    return "text-red";
  }
  return "text-muted";
}

function renderCaseDetails(container, payload) {
  const { case: summary, result, attempts, files, official } = payload;
  const critical = container.querySelector(".case-critical");
  critical.innerHTML = kvRows([
    ["Final result", summary.result],
    ["Attempts", String(summary.attempts)],
    ["Runtime profile", summary.runtimeProfile || "-"],
    ["PLLM", summary.pllmMatch || "-"],
    ["Classifier", summary.classifierOrigin || "-"],
    ["Root cause", summary.rootCauseBucket || "-"],
    ["Fallback", summary.pythonFallbackUsed ? "used" : "no"],
    ["Strategy", summary.candidatePlanStrategy || "-"],
    ["Official result", official.official_result || "-"],
    ["Official passed", official.official_passed || "-"],
    ["Official modules", official.official_python_modules || "-"],
    ["Dependencies", dependencyPreview(result.dependencies || [])],
  ]);

  const attemptList = container.querySelector(".attempt-list");
  attemptList.innerHTML = attempts
    .map(
      (attempt) => `
        <article class="attempt-card">
          <div class="attempt-header">
            <span class="section-title">Attempt ${escapeHtml(String(attempt.attemptNumber))}</span>
            <span class="${attempt.errorCategory || attempt.runSucceeded ? "text-yellow" : "text-muted"}">${escapeHtml(
              attempt.errorCategory || (attempt.runSucceeded ? "Success" : "No result"),
            )}</span>
          </div>
          <div class="attempt-badges">
            <span class="${attempt.buildSucceeded ? "text-green" : "text-red"}">build ${attempt.buildSucceeded ? "ok" : "fail"}</span>
            <span class="${attempt.runSucceeded ? "text-green" : "text-red"}">run ${attempt.runSucceeded ? "ok" : "fail"}</span>
            <span class="text-yellow">${escapeHtml(
              attempt.buildSkipped ? "build skipped" : formatSeconds(attempt.buildWallClockSeconds),
            )}</span>
            <span class="text-yellow">${escapeHtml(
              attempt.imageCacheHit ? "cache hit" : formatSeconds(attempt.runWallClockSeconds),
            )}</span>
          </div>
          <div class="attempt-grid">
            ${kvRows([
              ["Dependencies", dependencyPreview(attempt.dependencies || [])],
              ["Validation", attempt.validationCommand || "-"],
              ["Error details", attempt.errorDetails || "-"],
              ["Env cache key", attempt.environmentCacheKey || "-"],
            ])}
          </div>
          <div class="attempt-analysis">
            <div class="section-title">LLM analysis${
              attempt.llmFailureAnalysisModel ? ` (${escapeHtml(attempt.llmFailureAnalysisModel)})` : ""
            }</div>
            <div class="analysis-text">${escapeHtml(attempt.llmFailureAnalysis || "-")}</div>
          </div>
          <div class="attempt-activity">
            <div class="section-title">Activity</div>
            ${
              (attempt.activity || [])
                .map(
                  (event) => `
                    <div class="bullet-item">
                      • <span class="kv-value">${escapeHtml(event.kind)}</span> ${escapeHtml(event.detail || "")}
                    </div>
                  `,
                )
                .join("") || '<div class="empty-line">No attempt activity captured.</div>'
            }
          </div>
          <div class="artifact-links">
            ${(attempt.files || [])
              .map(
                (file) =>
                  `<a href="${escapeHtml(file.url)}" target="_blank" rel="noreferrer">${escapeHtml(file.label)}</a>`,
              )
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");

  const fileList = container.querySelector(".file-list");
  fileList.innerHTML = `
    <div class="section-title">Case artifacts</div>
    <div class="artifact-links">
      ${(files || [])
        .map((file) => `<a href="${escapeHtml(file.url)}" target="_blank" rel="noreferrer">${escapeHtml(file.label)}</a>`)
        .join("")}
    </div>
  `;
}

function renderCases() {
  const previousScrollTop = ui.casesScroll.scrollTop;
  ui.casesScroll.innerHTML = "";
  if (!state.selectedRun) {
    ui.casesScroll.innerHTML = `<div class="empty-line">Select a run to inspect cases.</div>`;
    return;
  }
  const activeCases = new Set(state.selectedRun.activeCases || []);
  const fragment = document.createDocumentFragment();
  for (const item of filteredCases()) {
    const node = ui.caseRowTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.caseId = item.caseId;
    node.dataset.status = item.status;
    if (activeCases.has(item.caseId)) {
      node.classList.add("is-active");
    }
    const stat = node.querySelector(".case-stat");
    stat.textContent = item.status;
    stat.classList.add(item.status === "PASS" ? "text-green" : "text-red");
    node.querySelector(".case-id").textContent = item.caseId;
    node.querySelector(".case-python").textContent = item.targetPython || "-";
    node.querySelector(".case-attempts").textContent = String(item.attempts || "-");
    node.querySelector(".case-seconds").textContent = Number(item.seconds || 0).toFixed(1);
    const pllm = node.querySelector(".case-pllm");
    pllm.textContent = item.pllmMatch || "-";
    pllm.classList.add(caseMatchClass(item.pllmMatch));
    node.querySelector(".case-result").textContent = item.result;
    node.querySelector(".case-dependencies").textContent = item.dependencyPreview;
    if (state.openCaseIds.has(item.caseId)) {
      node.open = true;
    }
    node.addEventListener("toggle", async () => {
      if (!node.open) {
        state.openCaseIds.delete(item.caseId);
        return;
      }
      state.openCaseIds.add(item.caseId);
      const detail = node.querySelector(".case-detail");
      if (detail.dataset.loaded === "true") {
        return;
      }
      detail.dataset.loaded = "loading";
      try {
        const payload = await ensureCaseDetail(item.caseId);
        renderCaseDetails(detail, payload);
        detail.dataset.loaded = "true";
      } catch (error) {
        detail.innerHTML = `<div class="empty-line">Failed to load case detail: ${escapeHtml(error.message)}</div>`;
        detail.dataset.loaded = "error";
      }
    });
    if (node.open && state.caseDetails.has(item.caseId)) {
      const detail = node.querySelector(".case-detail");
      renderCaseDetails(detail, state.caseDetails.get(item.caseId));
      detail.dataset.loaded = "true";
    }
    fragment.appendChild(node);
  }
  ui.casesScroll.appendChild(fragment);
  ui.casesScroll.scrollTop = previousScrollTop;
}

function renderSelectedRun() {
  const detail = state.selectedRun;
  if (!detail) {
    ui.heroRunTitle.textContent = "No run selected";
    ui.heroRunSubtitle.textContent = "Select a run to inspect benchmark progress and results.";
    ui.runInfoGrid.innerHTML = "";
    ui.progressLabel.textContent = "0/0";
    ui.progressPercent.textContent = "( 0.0% )";
    ui.progressFill.style.width = "0%";
    ui.metricsGrid.innerHTML = "";
    ui.perfLine.innerHTML = "";
    ui.researchLine.innerHTML = "";
    ui.lastLlmLine.innerHTML = "";
    renderActivityList(ui.activeCases, [], "No active cases.", () => "");
    renderActivityList(ui.recentActivity, [], "No recent activity.", () => "");
    renderCases();
    document.title = "APDR Benchmark Dashboard";
    return;
  }
  const run = detail.run;
  ui.heroRunTitle.textContent =
    run.status === "running" ? "APDR benchmark in progress" : `APDR benchmark ${run.status}`;
  ui.heroRunSubtitle.textContent =
    run.status === "running"
      ? "Use Ctrl-C only if you intend to stop the benchmark process itself."
      : `Run ${run.runId} is ${run.status}. Inspect results and attempt timelines below.`;
  renderRunInfo(run);
  renderProgress(run);
  renderActivityList(
    ui.activeCases,
    detail.currentCaseActivity || [],
    "No active cases.",
    (item) =>
      `• <span class="kv-value">${escapeHtml(item.case_id || item.caseId || "")}</span> [a${escapeHtml(
        String(item.attempt || 0),
      )} ${escapeHtml(item.kind || "activity")}] ${escapeHtml(item.detail || "")}`,
  );
  renderActivityList(
    ui.recentActivity,
    detail.recentCaseActivity || [],
    "No recent activity.",
    (item) =>
      `• <span class="kv-value">${escapeHtml(item.case_id || item.caseId || "")}</span> [${escapeHtml(
        item.kind || "activity",
      )}] ${escapeHtml(item.detail || "")}`,
  );
  renderCases();
  document.title = `APDR Benchmark Dashboard • ${run.runId}`;
}

function renderErrorState(error) {
  ui.heroRunTitle.textContent = "Dashboard error";
  ui.heroRunSubtitle.textContent = error.message;
}

async function loadHome() {
  const payload = await fetchJson("/api/home");
  state.home = payload.home || null;
  renderHome();
}

async function loadRuns() {
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs || [];
  if (!state.selectedRunId && state.runs.length) {
    state.selectedRunId = state.runs[0].runId;
  }
  if (state.selectedRunId && !state.runs.some((run) => run.runId === state.selectedRunId)) {
    state.selectedRunId = state.runs[0]?.runId || "";
  }
  renderRuns();
}

async function loadSelectedRun() {
  if (!state.selectedRunId) {
    state.selectedRun = null;
    renderSelectedRun();
    return;
  }
  const payload = await fetchJson(`/api/runs/${state.selectedRunId}`);
  state.selectedRun = payload;
  renderSelectedRun();
}

async function refresh() {
  try {
    await loadHome();
    await loadRuns();
    await loadSelectedRun();
    ui.lastRefresh.textContent = formatTimestamp(new Date().toISOString());
  } catch (error) {
    renderErrorState(error);
  }
}

ui.runSelect.addEventListener("change", (event) => {
  const nextRunId = event.target.value;
  if (nextRunId) {
    switchTab("runs");
    setSelectedRun(nextRunId);
  }
});

for (const button of ui.tabButtons) {
  button.addEventListener("click", () => switchTab(button.dataset.tab || "home"));
}

ui.caseSearch.addEventListener("input", (event) => {
  state.caseSearch = event.target.value.trim().toLowerCase();
  renderCases();
});

ui.caseFilter.addEventListener("change", (event) => {
  state.caseFilter = event.target.value;
  renderCases();
});

refresh();
switchTab(state.activeTab);
state.pollTimer = window.setInterval(refresh, 3000);
