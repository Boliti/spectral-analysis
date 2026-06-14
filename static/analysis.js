let selectedModel = null;
let datasetPreviewPayload = null;
let importedDatasetId = null;
let importedDatasetSummary = null;
let activeDatasetId = null;
let processedDatasetReady = false;
let lastResultPayload = null;
let lastComparisonRows = [];
let lastRenderedRunId = null;
let lastPredictionRows = [];
let selectedRunId = null;
let selectedSavedModel = null;
let lastDatasetPreviewData = null;
let lastResultPlots = [];
let loadingSlowTimer = null;
let selectedAnalysisTask = null;
let selectedAnalysisMethod = null;
let lastAnalysisSelectionKey = null;

const datasetFilesInput = document.getElementById("dataset-files");
const datasetPreviewBtn = document.getElementById("dataset-preview-btn");
const datasetValidateBtn = document.getElementById("dataset-validate-btn");
const datasetImportBtn = document.getElementById("dataset-import-btn");
const datasetConfigDownloadBtn = document.getElementById("dataset-config-download-btn");
const datasetLayoutInput = document.getElementById("dataset-layout");
const datasetSheetInput = document.getElementById("dataset-sheet");
const datasetSheetsInput = document.getElementById("dataset-sheets");
const datasetSheetsCheckboxes = document.getElementById("dataset-sheets-checkboxes");
const datasetSheetModeInput = document.getElementById("dataset-sheet-mode");
const datasetIdColumnInput = document.getElementById("dataset-id-column");
const datasetTargetColumnInput = document.getElementById("dataset-target-column");
const datasetAxisColumnInput = document.getElementById("dataset-axis-column");
const datasetTargetSourceInput = document.getElementById("dataset-target-source");
const datasetTargetRegexInput = document.getElementById("dataset-target-regex");
const datasetManualTargetInput = document.getElementById("dataset-manual-target");
const datasetGridModeInput = document.getElementById("dataset-grid-mode");
const datasetPreviewResult = document.getElementById("dataset-preview-result");
const datasetValidationResult = document.getElementById("dataset-validation-result");
const datasetSummaryEl = document.getElementById("dataset-summary");
const importResultCard = document.getElementById("import-result-card");
const datasetPreviewPlot = document.getElementById("dataset-preview-plot");
const datasetPreviewStatus = document.getElementById("dataset-preview-status");
const datasetPreviewVersionInput = document.getElementById("dataset-preview-version");
const datasetAndorHint = document.getElementById("dataset-andor-hint");
const datasetCheckSummary = document.getElementById("dataset-check-summary");
const datasetReadyCard = document.getElementById("dataset-ready-card");
const datasetReadySummary = document.getElementById("dataset-ready-summary");
const datasetUseBtn = document.getElementById("dataset-use-btn");
const datasetShowPlotBtn = document.getElementById("dataset-show-plot-btn");
const showPreprocessingSettingsBtn = document.getElementById("show-preprocessing-settings-btn");
const compactPreprocessingPlotBtn = document.getElementById("compact-preprocessing-plot-btn");
const compactPreprocessingRow = document.getElementById("compact-preprocessing-row");
const standardApplyPreprocessingInput = document.getElementById("standard-apply-preprocessing");
const standardPreprocessToggleWrap = document.getElementById("standard-preprocess-toggle-wrap");
const datasetExportCsvBtn = document.getElementById("dataset-export-csv-btn");
const datasetExportXlsxBtn = document.getElementById("dataset-export-xlsx-btn");
const datasetExportProcessedCsvBtn = document.getElementById("dataset-export-processed-csv-btn");
const datasetExportProcessedXlsxBtn = document.getElementById("dataset-export-processed-xlsx-btn");
const datasetExportZipBtn = document.getElementById("dataset-export-zip-btn");
const datasetExportProcessedZipBtn = document.getElementById("dataset-export-processed-zip-btn");
const datasetMetadataDownloadBtn = document.getElementById("dataset-metadata-download-btn");
const preprocessingConfigDownloadBtn = document.getElementById("preprocessing-config-download-btn");
const datasetResetBtn = document.getElementById("dataset-reset-btn");
const detectedMeasurementSummary = document.getElementById("detected-measurement-summary");
const measurementApplyDetectedBtn = document.getElementById("measurement-apply-detected-btn");
const measurementManualToggleBtn = document.getElementById("measurement-manual-toggle-btn");
const activeDatasetBadge = document.getElementById("active-dataset-badge");
const preprocessingCard = document.getElementById("preprocessing-card");
const preprocessingPresetInput = document.getElementById("preprocessing-preset");
const preprocessingResult = document.getElementById("preprocessing-result");
const trainDatasetVersionInput = document.getElementById("train-dataset-version");
const globalStatus = document.getElementById("global-status");
const globalStatusTitle = document.getElementById("global-status-title");
const globalStatusText = document.getElementById("global-status-text");
const globalProgressBar = document.getElementById("global-progress-bar");
const globalLoadingIndicator = document.getElementById("global-loading-indicator");
const globalLoadingText = document.getElementById("global-loading-text");
const toastStack = document.getElementById("toast-stack");
const preprocessingConfigView = document.getElementById("preprocessing-config-view");
const prePlotModeInput = document.getElementById("pre-plot-mode");
const prePlotScaleInput = document.getElementById("pre-plot-scale");
const prePlotLimitInput = document.getElementById("pre-plot-limit");
const prePlotStrategyInput = document.getElementById("pre-plot-strategy");
const prePlotVisibilityInput = document.getElementById("pre-plot-visibility");
const analysisRecommendations = document.getElementById("analysis-recommendations");
const analysisStepGuide = document.getElementById("analysis-step-guide");
const analysisRunSummary = document.getElementById("analysis-run-summary");
const modelConfigModeInput = document.getElementById("model-config-mode");
const analysisGoalInput = document.getElementById("analysis-goal");
const compareMethodsBtn = document.getElementById("compare-methods-btn");
const helpDialog = document.getElementById("help-dialog");
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const workflowModeInput = document.getElementById("workflow-mode");
const workflowModeHint = document.getElementById("workflow-mode-hint");
const standardDatasetFileInput = document.getElementById("standard-dataset-file");
const standardDatasetImportBtn = document.getElementById("standard-dataset-import-btn");
const standardDatasetResult = document.getElementById("standard-dataset-result");
const comparisonResult = document.getElementById("comparison-result");
const runsList = document.getElementById("runs-list");
const processingBlockIds = ["data-check", "baseline", "smoothing", "normalization", "peaks", "pca", "model-compare", "export"];
let lastPreprocessingPreview = null;
const plotRegistry = new Set();
let fullscreenPlotSourceId = null;

const trainFileInput = document.getElementById("train-file");
const manualTrainUploadRow = document.getElementById("manual-train-upload-row");
const inferFileInput = document.getElementById("infer-file");
const modelTypeInput = document.getElementById("model-type");
const targetValuesInput = document.getElementById("target-values");
const targetsFileInput = document.getElementById("targets-file");
const targetsFileHint = document.getElementById("targets-file-hint");
const modelNameInput = document.getElementById("model-name");
const nComponentsInput = document.getElementById("n-components");
const doValidationInput = document.getElementById("do-validation");
const testSizeInput = document.getElementById("test-size");
const randomStateInput = document.getElementById("random-state");

const uploadModelFileInput = document.getElementById("upload-model-file");
const uploadMetaFileInput = document.getElementById("upload-meta-file");
const uploadModelTypeInput = document.getElementById("upload-model-type");
const uploadModelNameInput = document.getElementById("upload-model-name");

const previewResult = document.getElementById("preview-result");
const textResult = document.getElementById("text-result");
const resultsSection = document.getElementById("results-section");
const rawJsonWrap = document.getElementById("raw-json-wrap");
const resultExplainer = document.getElementById("result-explainer");
const validationSummary = document.getElementById("validation-summary");
const validationPlotEl = document.getElementById("validation-plot");
const modelsList = document.getElementById("models-list");
const resultDownloads = document.getElementById("result-downloads");
const modelMetadataCard = document.getElementById("model-metadata-card");
const targetSourceNote = document.getElementById("target-source-note");
const targetsWrap = document.getElementById("targets-wrap");
const loadedModelBadge = document.getElementById("loaded-model-badge");
const inferFilesHint = document.getElementById("infer-files-hint");

const LAYOUT_LABELS = {
    excel_rows: "Спектры по строкам",
    excel_columns: "Спектры по столбцам",
    txt_folder: "Папка или ZIP",
    long_table: "Длинный формат",
    manual: "Ручной режим",
};

const TARGET_SOURCE_OPTIONS = {
    excel_rows: [
        ["column", "Колонка таблицы"],
        ["none", "Без target"],
    ],
    excel_columns: [
        ["sheet_name", "Имя листа Excel"],
        ["file_name", "Имя файла"],
        ["manual", "Один общий класс"],
        ["none", "Без target"],
    ],
    txt_folder: [
        ["none", "Без target"],
        ["folder_name", "Имя папки"],
        ["filename_regex", "Regex из имени файла"],
        ["file_name", "Имя файла"],
        ["manual", "Один общий класс"],
    ],
};

function humanLayout(layout) {
    return LAYOUT_LABELS[layout] || layout || "Не определён";
}

function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "н/д";
    }
    return Number(value).toFixed(2).replace(/\.00$/, "");
}

function uniqueValues(values = []) {
    const out = [];
    values.forEach((value) => {
        if (value !== null && value !== undefined && value !== "" && !out.includes(value)) out.push(value);
    });
    return out;
}

function valuesFromMetadata(summaryOrMeta = {}, key) {
    const meta = summaryOrMeta.metadata || summaryOrMeta || {};
    const measurement = summaryOrMeta.measurement_metadata || meta.measurement_metadata || {};
    const direct = meta[`${key}_values`] || measurement[`${key}_values`] || [];
    const preview = Array.isArray(meta.sample_metadata_preview) ? meta.sample_metadata_preview : [];
    const sampleValues = preview.map((item) => item?.[key]);
    const single = meta[key] ?? measurement[key];
    return uniqueValues([...(Array.isArray(direct) ? direct : [direct]), ...sampleValues, single]);
}

function formatDetectedValues(values, unit = "") {
    const clean = uniqueValues(values);
    if (!clean.length) return "";
    const numeric = clean.map(Number).filter((value) => Number.isFinite(value));
    if (numeric.length === clean.length) {
        numeric.sort((a, b) => a - b);
        if (numeric.length === 1) return `${formatNumber(numeric[0])}${unit}`;
        return `${formatNumber(numeric[0])}-${formatNumber(numeric[numeric.length - 1])}${unit}, ${numeric.length} values`;
    }
    return `${clean.slice(0, 4).join(", ")}${clean.length > 4 ? `, +${clean.length - 4}` : ""}${unit}`;
}

function detectedMeasurementCards(summaryOrMeta = {}) {
    return [
        ["Щель", formatDetectedValues(valuesFromMetadata(summaryOrMeta, "slit_width_um"), " мкм")],
        ["Решётка", formatDetectedValues(valuesFromMetadata(summaryOrMeta, "grating_lines_mm"), " штр./мм")],
        ["Экспозиция", formatDetectedValues(valuesFromMetadata(summaryOrMeta, "exposure_time_s"), " с")],
        ["Накопления", formatDetectedValues(valuesFromMetadata(summaryOrMeta, "accumulations"), "")],
        ["Мощность", formatDetectedValues(valuesFromMetadata(summaryOrMeta, "power_mw"), " мВт")],
        ["Образец", formatDetectedValues(valuesFromMetadata(summaryOrMeta, "sample_name"), "")],
    ].filter(([, value]) => value);
}

function renderDetectedMeasurementSummary(summaryOrMeta = {}) {
    if (!detectedMeasurementSummary) return;
    const cards = detectedMeasurementCards(summaryOrMeta);
    if (!cards.length) {
        detectedMeasurementSummary.style.display = "none";
        detectedMeasurementSummary.innerHTML = "";
        if (measurementApplyDetectedBtn) measurementApplyDetectedBtn.disabled = true;
        return;
    }
    detectedMeasurementSummary.style.display = "";
    detectedMeasurementSummary.innerHTML = `
        <strong>Обнаруженные параметры измерений</strong>
        <div class="measurement-card-grid">
            ${cards.map(([label, value]) => `<div class="measurement-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
        </div>
    `;
    if (measurementApplyDetectedBtn) measurementApplyDetectedBtn.disabled = false;
}

function applyDetectedMeasurementValues(summaryOrMeta = importedDatasetSummary) {
    const setIfSingle = (id, key, suffix = "") => {
        const values = valuesFromMetadata(summaryOrMeta, key);
        const el = document.getElementById(id);
        if (el && values.length === 1) el.value = `${values[0]}${suffix}`;
    };
    setIfSingle("meta-slit-width", "slit_width_um", " um");
    setIfSingle("meta-grating-lines", "grating_lines_mm");
    setIfSingle("meta-integration-time", "exposure_time_s", " s");
    setIfSingle("meta-accumulation-count", "accumulations");
}

function currentTheme() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function getPlotlyTheme() {
    const styles = getComputedStyle(document.documentElement);
    const css = (name) => styles.getPropertyValue(name).trim();
    return {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: css("--plot-bg") || css("--surface") || (currentTheme() === "dark" ? "rgb(17, 24, 39)" : "rgb(255, 255, 255)"),
        font: { color: css("--text") || "#1f2937" },
        xaxis: {
            gridcolor: css("--border") || "#dbe3ef",
            zerolinecolor: css("--border") || "#dbe3ef",
            linecolor: css("--muted") || "#64748b",
            tickfont: { color: css("--text") || "#1f2937" },
            titlefont: { color: css("--text") || "#1f2937" },
        },
        yaxis: {
            gridcolor: css("--border") || "#dbe3ef",
            zerolinecolor: css("--border") || "#dbe3ef",
            linecolor: css("--muted") || "#64748b",
            tickfont: { color: css("--text") || "#1f2937" },
            titlefont: { color: css("--text") || "#1f2937" },
        },
        legend: { font: { color: css("--text") || "#1f2937" } },
    };
}

function themedLayout(layout = {}) {
    const theme = getPlotlyTheme();
    const merged = {
        autosize: true,
        height: layout.height || 650,
        margin: { l: 62, r: 28, t: 58, b: 62, ...(layout.margin || {}) },
        ...theme,
        ...layout,
        xaxis: { ...theme.xaxis, ...(layout.xaxis || {}) },
        yaxis: { ...theme.yaxis, ...(layout.yaxis || {}) },
        legend: { ...theme.legend, ...(layout.legend || {}) },
    };
    if (layout.xaxis2) merged.xaxis2 = { ...theme.xaxis, ...layout.xaxis2 };
    if (layout.yaxis2) merged.yaxis2 = { ...theme.yaxis, ...layout.yaxis2 };
    return merged;
}

const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function newPlot(id, data, layout = {}, config = {}) {
    const element = typeof id === "string" ? document.getElementById(id) : id;
    if (!element) {
        console.warn(`Контейнер графика не найден: ${id}`);
        return Promise.resolve(null);
    }
    if (typeof Plotly === "undefined") {
        element.classList?.add("is-empty");
        element.textContent = "Библиотека графиков не загружена.";
        return Promise.resolve(null);
    }
    const plotId = element.id || (typeof id === "string" ? id : "");
    if (plotId) plotRegistry.add(plotId);
    ensureChartActions(plotId);
    return Plotly.newPlot(element, data, themedLayout(layout), { ...plotConfig, ...config });
}

function applyTheme(theme) {
    const normalized = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = normalized;
    localStorage.setItem("theme", normalized);
    if (themeToggleBtn) {
        themeToggleBtn.textContent = normalized === "dark" ? "Светлая тема" : "Тёмная тема";
    }
    updateAllAnalysisPlotsTheme();
}

function initTheme() {
    applyTheme(localStorage.getItem("theme") || "light");
}

function updateAllAnalysisPlotsTheme() {
    if (typeof Plotly === "undefined") return;
    const theme = getPlotlyTheme();
    plotRegistry.forEach((id) => {
        const el = document.getElementById(id);
        if (!el || !el.data) return;
        const relayout = {
            paper_bgcolor: theme.paper_bgcolor,
            plot_bgcolor: theme.plot_bgcolor,
            "font.color": theme.font.color,
            "legend.font.color": theme.legend.font.color,
            "xaxis.gridcolor": theme.xaxis.gridcolor,
            "xaxis.zerolinecolor": theme.xaxis.zerolinecolor,
            "xaxis.linecolor": theme.xaxis.linecolor,
            "yaxis.gridcolor": theme.yaxis.gridcolor,
            "yaxis.zerolinecolor": theme.yaxis.zerolinecolor,
            "yaxis.linecolor": theme.yaxis.linecolor,
            "xaxis2.gridcolor": theme.xaxis.gridcolor,
            "xaxis2.zerolinecolor": theme.xaxis.zerolinecolor,
            "xaxis2.linecolor": theme.xaxis.linecolor,
            "yaxis2.gridcolor": theme.yaxis.gridcolor,
            "yaxis2.zerolinecolor": theme.yaxis.zerolinecolor,
            "yaxis2.linecolor": theme.yaxis.linecolor,
        };
        Plotly.relayout(id, relayout);
    });
}

function ensureChartActions(plotId) {
    const plot = document.getElementById(plotId);
    if (!plot || plot.previousElementSibling?.classList?.contains("chart-actions")) return;
    const actions = document.createElement("div");
    actions.className = "chart-actions";
    actions.innerHTML = `
        <button type="button" class="btn btn-secondary" data-chart-fullscreen="${plotId}">Развернуть график</button>
        <button type="button" class="btn btn-ghost" data-chart-png="${plotId}">Скачать PNG</button>
    `;
    plot.parentNode.insertBefore(actions, plot);
}

function openFullscreenPlot(plotId) {
    const source = document.getElementById(plotId);
    if (!source || !source.data || typeof Plotly === "undefined") {
        showToast("warning", "График ещё не построен.");
        return;
    }
    fullscreenPlotSourceId = plotId;
    const dialog = document.getElementById("chart-modal") || createChartModal();
    dialog.showModal();
    const modalPlot = document.getElementById("chart-modal-plot");
    const layout = themedLayout({
        ...(source.layout || {}),
        width: Math.floor(window.innerWidth * 0.92),
        height: Math.max(600, Math.floor(window.innerHeight * 0.78)),
        autosize: true,
    });
    Plotly.newPlot(modalPlot, source.data, layout, plotConfig).then(() => {
        Plotly.Plots.resize(modalPlot);
    });
}

function createChartModal() {
    const dialog = document.createElement("dialog");
    dialog.id = "chart-modal";
    dialog.className = "chart-modal";
    dialog.innerHTML = `
        <div class="chart-modal-header">
            <strong>График</strong>
            <button type="button" class="btn" id="chart-modal-close">Закрыть</button>
        </div>
        <div class="chart-modal-body"><div id="chart-modal-plot" class="chart-modal-plot"></div></div>
    `;
    document.body.appendChild(dialog);
    document.getElementById("chart-modal-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
        const modalPlot = document.getElementById("chart-modal-plot");
        if (modalPlot && typeof Plotly !== "undefined") Plotly.purge(modalPlot);
    });
    return dialog;
}

function downloadPlotPng(plotId) {
    const plot = document.getElementById(plotId);
    if (!plot) return;
    Plotly.downloadImage(plot, { format: "png", filename: `${plotId}_${currentTheme()}`, width: 1600, height: 1000 });
}

function setGlobalStatus(title, text, progress = 10) {
    if (!globalStatus) return;
    globalStatus.style.display = "grid";
    globalStatusTitle.textContent = title;
    globalStatusText.textContent = text;
    globalProgressBar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
}

function clearGlobalStatus(successText = "Готово.") {
    if (!globalStatus) return;
    globalStatusTitle.textContent = successText;
    globalStatusText.textContent = "";
    globalProgressBar.style.width = "100%";
    setTimeout(() => {
        globalStatus.style.display = "none";
        globalProgressBar.style.width = "0%";
    }, 900);
}

function showGlobalLoading(message = "Выполняется...") {
    if (!globalLoadingIndicator || !globalLoadingText) return;
    globalLoadingText.textContent = message;
    globalLoadingIndicator.style.display = "flex";
    clearTimeout(loadingSlowTimer);
    loadingSlowTimer = setTimeout(() => {
        if (globalLoadingText) {
            globalLoadingText.textContent = "Операция занимает больше времени, чем обычно. Это нормально для больших датасетов.";
        }
    }, 5000);
}

function hideGlobalLoading() {
    clearTimeout(loadingSlowTimer);
    loadingSlowTimer = null;
    if (globalLoadingIndicator) globalLoadingIndicator.style.display = "none";
}

async function withLoading(button, message, asyncFn) {
    const doneBusy = setBusy(button, "Выполняется...");
    showGlobalLoading(message || "Выполняется...");
    try {
        return await asyncFn();
    } finally {
        doneBusy();
        hideGlobalLoading();
    }
}

function showToast(first, second = "success") {
    if (!toastStack) return;
    const knownTypes = new Set(["success", "warning", "error", "info"]);
    const type = knownTypes.has(first) ? first : (knownTypes.has(second) ? second : "success");
    const message = knownTypes.has(first) ? second : first;
    const item = document.createElement("div");
    item.className = `toast toast--${type}`;
    item.textContent = message || "";
    toastStack.appendChild(item);
    setTimeout(() => item.remove(), 5200);
}

function showError(userMessage, details = "") {
    const message = details ? `${userMessage}\n${details}` : userMessage;
    showToast("error", message || "Операция не выполнена.");
}

function humanError(data, fallback = "Операция не выполнена.") {
    const raw = String(data?.user_message || data?.detail || data?.message || fallback);
    if (raw.includes("window_length") || raw.includes("size of x")) {
        return "Размер окна сглаживания больше количества спектральных точек. Уменьшите window_length.";
    }
    if (raw.includes("polyorder")) {
        return "Порядок полинома должен быть меньше размера окна сглаживания.";
    }
    if (raw.includes("crop")) {
        return "Проверьте границы обрезки диапазона: min_axis должен быть меньше max_axis.";
    }
    return raw.replace(/^ValueError:\s*/i, "");
}

window.addEventListener("error", (event) => {
    keepDatasetUploadVisible();
    showToast("error", event.message || "Ошибка интерфейса. Блок загрузки оставлен доступным.");
});

window.addEventListener("unhandledrejection", (event) => {
    keepDatasetUploadVisible();
    const reason = event.reason?.message || event.reason || "Ошибка интерфейса. Блок загрузки оставлен доступным.";
    showToast("error", String(reason));
});

function renderErrorBox(element, data, fallback) {
    if (!element) return;
    const message = data?.user_message || data?.detail || fallback;
    const suggestions = Array.isArray(data?.suggestions) && data.suggestions.length
        ? `\n\nЧто проверить:\n- ${data.suggestions.join("\n- ")}`
        : "";
    const details = data?.details ? `\n\nТехнические подробности:\n${data.details}` : "";
    element.textContent = `${message}${suggestions}${details}`;
}

function setButtonLoading(button, isLoading, loadingText = "Выполняется...") {
    if (!button) return;
    if (isLoading) {
        if (!button.dataset.originalText) button.dataset.originalText = button.textContent || "";
        button.disabled = true;
        button.classList.add("is-loading");
        button.innerHTML = `<span class="mini-spinner"></span>${loadingText}`;
        return;
    }
    button.disabled = false;
    button.classList.remove("is-loading");
    button.textContent = button.dataset.originalText || button.textContent || "";
    delete button.dataset.originalText;
}

function setBusy(button, busyText) {
    if (!button) return () => {};
    const oldText = button.textContent;
    setButtonLoading(button, true, busyText);
    showGlobalLoading(busyText || "Выполняется...");
    return () => {
        button.dataset.originalText = oldText;
        setButtonLoading(button, false);
        hideGlobalLoading();
    };
}

function setSectionStatus(sectionId, status, message) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    section.dataset.status = status || "";
    let statusNode = section.querySelector(":scope > .section-status");
    if (!statusNode) {
        statusNode = document.createElement("div");
        statusNode.className = "section-status result-note";
        section.prepend(statusNode);
    }
    statusNode.textContent = message || "";
    statusNode.className = `section-status result-note section-status--${status || "info"}`;
}

function markStepDone(step) {
    const tab = document.querySelector(`.workflow-tab[data-step="${step}"]`);
    if (!tab) return;
    tab.classList.remove("workflow-tab--active", "workflow-tab--warning", "workflow-tab--error");
    tab.classList.add("workflow-tab--done");
    const state = tab.querySelector(".step-state");
    if (state) state.textContent = "done";
}

function setWorkflowStepStatus(step, status = "active") {
    const tab = document.querySelector(`.workflow-tab[data-step="${step}"]`);
    if (!tab) return;
    tab.classList.remove("workflow-tab--active", "workflow-tab--warning", "workflow-tab--error", "workflow-tab--done");
    if (status === "done") {
        markStepDone(step);
        return;
    }
    tab.classList.add(`workflow-tab--${status}`);
    const state = tab.querySelector(".step-state");
    if (state) state.textContent = status;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 120000, slowMessage = "Обработка занимает больше времени, чем обычно. Это нормально для больших датасетов.") {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const slow = setTimeout(() => showToast(slowMessage, "warning"), Math.min(5000, timeoutMs / 2));
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error("Сервер отвечает слишком долго. Попробуйте уменьшить объём данных или повторить запрос.");
        }
        throw error;
    } finally {
        clearTimeout(timeout);
        clearTimeout(slow);
    }
}

async function responseJsonOrError(response, fallback = "Операция не выполнена.") {
    const text = await response.text();
    if (!text) return {};
    try {
        return JSON.parse(text);
    } catch (_error) {
        return {
            detail: response.ok ? fallback : `${fallback} Сервер вернул не-JSON ответ: ${text.slice(0, 180)}`,
        };
    }
}

function updateTargetVisibility() {
    if (!modelTypeInput || !targetsWrap) return;
    const mt = modelTypeInput.value;
    const simple = (modelConfigModeInput?.value || "simple") === "simple";
    const supervised = ["pls", "plsda", "svm", "random_forest", "decision_tree", "svr", "compare_classification", "compare_regression"].includes(mt);
    const importedTarget = activeDatasetId && importedDatasetSummary?.target_name;
    targetsWrap.style.display = supervised && !importedTarget && !simple ? "flex" : "none";
    const importWrap = document.getElementById("targets-import-wrap");
    if (importWrap) {
        importWrap.style.display = supervised && !importedTarget && !simple ? "flex" : "none";
    }
    if (targetSourceNote) {
        targetSourceNote.style.display = importedTarget ? "block" : "none";
        targetSourceNote.textContent = importedTarget ? `Target будет взят из датасета: ${importedTarget}` : "";
    }
    if (importedTarget && previewResult) {
        const distribution = importedDatasetSummary?.class_distribution || {};
        if (previewResult) previewResult.textContent = [
            `Target будет взят из импортированного датасета: ${importedTarget}`,
            `Тип target: ${importedDatasetSummary?.target_type || "н/д"}`,
            `Распределение:\n${Object.entries(distribution).map(([k, v]) => `${k} — ${v}`).join("\n") || "н/д"}`,
        ].join("\n");
    }
    updateModelAvailability();
}

function updateModelAvailability() {
    if (!modelTypeInput) return;
    const targetType = importedDatasetSummary?.target_type || "none";
    const hasTarget = Boolean(activeDatasetId && importedDatasetSummary?.target_name);
    const classificationTypes = ["plsda", "svm", "random_forest", "decision_tree", "compare_classification"];
    const regressionTypes = ["pls", "svr", "compare_regression"];
    Array.from(modelTypeInput.options || []).forEach((option) => {
        let disabled = false;
        let reason = "";
        if (classificationTypes.includes(option.value)) {
            disabled = !hasTarget || targetType !== "categorical";
            reason = "Для классификации нужен categorical target.";
        }
        if (regressionTypes.includes(option.value)) {
            disabled = !hasTarget || targetType !== "numeric";
            reason = "Для регрессии нужен numeric target.";
        }
        option.disabled = disabled;
        option.title = disabled ? reason : "";
    });
    if (modelTypeInput.selectedOptions[0]?.disabled) {
        modelTypeInput.value = "pca";
        selectedAnalysisTask = "explore";
        selectedAnalysisMethod = "pca";
    }
    if (previewResult && activeDatasetId && targetType === "none") {
        previewResult.textContent = "Target отсутствует. Для классификации или регрессии нужен target; сейчас доступны PCA и кластеризация.";
    }
}

function updateModelModeUI() {
    const simple = (modelConfigModeInput?.value || "simple") === "simple";
    document.querySelectorAll(".model-advanced-field").forEach((node) => {
        node.style.display = "";
    });
    if (simple) autoTuneModelParams();
    updateModelAdvancedSettings(modelTypeInput?.value || "pca");
    updateTargetVisibility();
}

function updateModelAdvancedSettings(modelType) {
    const simple = (modelConfigModeInput?.value || "simple") === "simple";
    const map = {
        pca: ["n_components"],
        plsda: ["n_components", "validation", "test_size", "random_state"],
        pls: ["n_components", "validation", "test_size", "random_state"],
        svm: ["kernel", "C", "gamma", "class_weight", "validation", "test_size", "random_state"],
        random_forest: ["n_estimators", "max_depth", "class_weight", "validation", "test_size", "random_state"],
        decision_tree: ["max_depth", "class_weight", "validation", "test_size", "random_state"],
        kmeans: ["n_clusters", "random_state", "standardize_cluster"],
        hca: ["n_clusters", "linkage", "standardize_cluster"],
        svr: ["kernel", "C", "gamma", "epsilon", "validation", "test_size", "random_state"],
        compare_classification: ["validation", "test_size", "random_state", "class_weight"],
        compare_regression: ["validation", "test_size", "random_state"],
        compare_clustering: ["n_clusters", "random_state", "linkage"],
    };
    const allowed = new Set(map[modelType] || []);
    document.querySelectorAll("[data-model-param]").forEach((node) => {
        node.style.display = !simple && allowed.has(node.dataset.modelParam) ? "flex" : "none";
    });
    renderAnalysisWorkflowGuide(importedDatasetSummary);
    renderAnalysisRunSummary();
}

function autoTuneModelParams() {
    if (!modelTypeInput) return;
    const goal = analysisGoalInput?.value || "explore";
    const summary = importedDatasetSummary || {};
    const targetType = summary.target_type || "none";
    const nSamples = Number(summary.n_samples || 2);
    const nFeatures = Number(summary.n_features || 2);
    const nClasses = Object.keys(summary.class_distribution || {}).length;
    let modelType = "pca";
    if (goal === "classify") {
        modelType = targetType === "categorical" ? "plsda" : "pca";
        if (targetType !== "categorical") showToast("В датасете не задан категориальный target. Для классификации выберите target или стратегию извлечения класса.", "warning");
    }
    if (goal === "compare") {
        if (targetType === "categorical") modelType = "compare_classification";
        else if (targetType === "numeric") modelType = "compare_regression";
        else {
            modelType = "pca";
            showToast("Для сравнения моделей с target нужен target. Сейчас доступны PCA и кластеризация.", "warning");
        }
    }
    if (goal === "compare_regression") modelType = targetType === "numeric" ? "compare_regression" : "pca";
    if (goal === "cluster") modelType = "kmeans";
    if (goal === "regress") {
        modelType = targetType === "numeric" ? "pls" : "pca";
        if (targetType !== "numeric") showToast("Для регрессии нужен числовой target.", "warning");
    }
    modelTypeInput.value = modelType;
    selectedAnalysisTask = goalForModelType(modelType);
    selectedAnalysisMethod = modelType;
    syncAnalysisSelectionToInputs();
    if (modelType === "pca") {
        if (nComponentsInput) nComponentsInput.value = String(Math.max(1, Math.min(2, nSamples, nFeatures)));
        if (doValidationInput) doValidationInput.value = "false";
    } else if (modelType === "plsda") {
        if (nComponentsInput) nComponentsInput.value = String(Math.max(1, Math.min(3, Math.max(1, nClasses - 1), nSamples - 1, nFeatures)));
        if (doValidationInput) doValidationInput.value = "true";
        if (testSizeInput) testSizeInput.value = "0.3";
        if (randomStateInput) randomStateInput.value = "42";
    } else if (modelType === "pls") {
        if (nComponentsInput) nComponentsInput.value = String(Math.max(1, Math.min(5, nSamples - 1, nFeatures)));
        if (doValidationInput) doValidationInput.value = "true";
        if (testSizeInput) testSizeInput.value = "0.3";
        if (randomStateInput) randomStateInput.value = "42";
    } else if (["svm", "random_forest", "decision_tree", "svr", "compare_classification", "compare_regression"].includes(modelType)) {
        if (nComponentsInput) nComponentsInput.value = "";
        if (doValidationInput) doValidationInput.value = "true";
        if (testSizeInput) testSizeInput.value = "0.3";
        if (randomStateInput) randomStateInput.value = "42";
    } else if (["kmeans", "hca", "compare_clustering"].includes(modelType)) {
        if (nComponentsInput) nComponentsInput.value = "";
        if (doValidationInput) doValidationInput.value = "false";
        if (randomStateInput) randomStateInput.value = "42";
    }
    updateTargetVisibility();
    renderAnalysisWorkflowGuide(importedDatasetSummary);
    renderAnalysisRunSummary();
}

function setLoadedModelBadge() {
    if (!loadedModelBadge) return;
    if (!selectedModel) {
        loadedModelBadge.classList.remove("loaded-badge--ok");
        loadedModelBadge.classList.add("loaded-badge--empty");
        loadedModelBadge.textContent = "Модель не загружена";
        return;
    }
    loadedModelBadge.classList.remove("loaded-badge--empty");
    loadedModelBadge.classList.add("loaded-badge--ok");
    loadedModelBadge.textContent = `Загружена модель: ${selectedModel.modelId} [${selectedModel.modelType}]`;
}

function updateInferFilesHint() {
    if (!inferFileInput || !inferFilesHint) return;
    const files = Array.from(inferFileInput.files || []);
    if (files.length === 0) {
        inferFilesHint.textContent = "Файлы для применения модели не выбраны.";
        return;
    }
    inferFilesHint.textContent = `Выбрано файлов: ${files.length}. ${files.map((f) => f.name).join(", ")}`;
}

function renderText(obj) {
    if (resultsSection) resultsSection.style.display = "";
    if (rawJsonWrap) rawJsonWrap.style.display = "";
    if (!textResult) return;
    textResult.textContent = JSON.stringify(obj, null, 2);
}

function clearStructuredBlocks() {
    lastPredictionRows = [];
    if (resultExplainer) resultExplainer.textContent = "";
    if (validationSummary) validationSummary.innerHTML = "";
    if (comparisonResult) comparisonResult.innerHTML = "";
    if (validationPlotEl && typeof Plotly !== "undefined") {
        Plotly.purge("validation-plot");
        validationPlotEl.style.display = "none";
    }
    const plotEl = document.getElementById("plot");
    if (plotEl && typeof Plotly !== "undefined") {
        normalizePlotList(lastResultPlots).forEach((_, index) => {
            const plotNode = document.getElementById(`result-plot-${index}`);
            if (plotNode) Plotly.purge(plotNode);
        });
        Plotly.purge("plot");
        plotEl.innerHTML = "";
        plotEl.classList.remove("plot-stack");
    }
    if (rawJsonWrap && !textResult?.textContent.trim()) rawJsonWrap.style.display = "none";
    if (resultsSection) resultsSection.style.display = "none";
}

function metricCard(title, value) {
    const div = document.createElement("div");
    div.className = "metric-card";
    div.innerHTML = `<div class="metric-title">${title}</div><div class="metric-value">${value}</div>`;
    return div;
}

function renderValidationBlock(validation) {
    if (!validation || validation.status !== "ok") {
        if (validation && validation.reason) {
            resultExplainer.textContent += `\nВалидация: ${validation.reason}`;
        }
        return;
    }

    const metrics = validation.metrics || {};
    Object.entries(metrics).forEach(([key, val]) => {
        const num = typeof val === "number" ? val.toFixed(4) : String(val);
        validationSummary.appendChild(metricCard(key, num));
    });
    validationSummary.appendChild(metricCard("train_samples", validation.train_samples));
    validationSummary.appendChild(metricCard("test_samples", validation.test_samples));

    const kfold = validation.kfold;
    if (kfold && kfold.status === "ok" && kfold.mean_std) {
        Object.entries(kfold.mean_std).forEach(([metricName, stats]) => {
            validationSummary.appendChild(metricCard(`kfold ${metricName} mean`, Number(stats.mean).toFixed(4)));
            validationSummary.appendChild(metricCard(`kfold ${metricName} std`, Number(stats.std).toFixed(4)));
        });
    }

    const permutation = validation.permutation_test;
    if (permutation && permutation.status === "ok") {
        validationSummary.appendChild(metricCard("perm metric", permutation.metric));
        validationSummary.appendChild(metricCard("perm observed", Number(permutation.observed_score).toFixed(4)));
        validationSummary.appendChild(metricCard("perm p-value", Number(permutation.p_value).toFixed(4)));
    }

    const bootstrap = validation.bootstrap_ci;
    if (bootstrap && bootstrap.status === "ok") {
        Object.entries(bootstrap).forEach(([k, v]) => {
            if (!k.endsWith("_95ci") || !Array.isArray(v) || v.length !== 2) {
                return;
            }
            validationSummary.appendChild(metricCard(k, `[${Number(v[0]).toFixed(4)}, ${Number(v[1]).toFixed(4)}]`));
        });
    }

    if (Array.isArray(validation.limitations) && validation.limitations.length > 0) {
        resultExplainer.textContent += `\n\nОграничения:\n- ${validation.limitations.join("\n- ")}`;
    }

    if (Array.isArray(validation.confusion_matrix) && Array.isArray(validation.classes)) {
        validationPlotEl.style.display = "block";
        newPlot(
            "validation-plot",
            [
                {
                    type: "heatmap",
                    z: validation.confusion_matrix,
                    x: validation.classes,
                    y: validation.classes,
                    text: validation.confusion_matrix,
                    texttemplate: "%{text}",
                    hovertemplate: "Истинный класс: %{y}<br>Предсказанный класс: %{x}<br>Количество: %{z}<extra></extra>",
                    colorscale: "Blues",
                    showscale: true,
                },
            ],
            {
                title: "Confusion Matrix (test)",
                xaxis: { title: "Predicted class" },
                yaxis: { title: "True class" },
            },
            plotConfig
        );
    }
}

function renderInferenceExplanation(payload) {
    const result = payload.aggregate_result || payload.result || {};
    if (Array.isArray(result.predicted_classes)) {
        const counts = {};
        result.predicted_classes.forEach((cls) => {
            counts[cls] = (counts[cls] || 0) + 1;
        });
        const lines = Object.entries(counts).map(([cls, count]) => `Класс ${cls}: ${count}`);
        resultExplainer.textContent = `${resultExplainer.textContent}\nРаспределение предсказанных классов:\n${lines.join("\n")}`;
        return;
    }

    if (Array.isArray(result.predictions) && result.predictions.length > 0) {
        const values = result.predictions.map(Number);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        resultExplainer.textContent = `${resultExplainer.textContent}\nКоличество предсказаний: ${values.length}\nСреднее: ${mean.toFixed(4)}\nМин/Макс: ${min.toFixed(4)} / ${max.toFixed(4)}`;
        return;
    }

    resultExplainer.textContent = `${resultExplainer.textContent}\nПодробности доступны в JSON.`;
}

function collectPredictionRows(payload) {
    const rows = [];
    if (Array.isArray(payload.predictions) && payload.predictions.length) {
        return payload.predictions.map((row, index) => ({
            sample_id: row.sample_id || row.file_name || row.filename || `sample_${index + 1}`,
            source_file: row.source_file || row.file_name || row.filename || "",
            predicted: row.predicted ?? row.prediction ?? row.predicted_class ?? row.value ?? "",
            confidence: row.confidence ?? row.probability ?? row.proba ?? row.score ?? "",
        }));
    }
    const batch = Array.isArray(payload.batch_results) ? payload.batch_results : [];
    batch.forEach((item) => {
        const result = item.result || {};
        const classes = result.predicted_classes || result.predictions || result.scores || [];
        classes.forEach((value, index) => {
            rows.push({
                sample_id: result.sample_ids?.[index] || `${item.filename || "sample"}:${index + 1}`,
                source_file: item.filename || "",
                predicted: Array.isArray(value) ? value.join(";") : value,
                confidence: result.confidence?.[index] ?? result.probability?.[index] ?? result.proba?.[index] ?? "",
            });
        });
    });
    const aggregate = payload.aggregate_result || payload.result || {};
    if (!rows.length && Array.isArray(aggregate.predicted_classes)) {
        aggregate.predicted_classes.forEach((value, index) => rows.push({ sample_id: aggregate.sample_ids?.[index] || `sample_${index + 1}`, source_file: aggregate.source_files?.[index] || "", predicted: value, confidence: aggregate.confidence?.[index] ?? aggregate.probability?.[index] ?? "" }));
    }
    if (!rows.length && Array.isArray(aggregate.predictions)) {
        aggregate.predictions.forEach((value, index) => rows.push({ sample_id: aggregate.sample_ids?.[index] || `sample_${index + 1}`, source_file: aggregate.source_files?.[index] || "", predicted: value, confidence: aggregate.confidence?.[index] ?? aggregate.probability?.[index] ?? "" }));
    }
    return rows;
}

function renderPredictionsTable(payload) {
    if (!comparisonResult) return;
    lastPredictionRows = collectPredictionRows(payload);
    if (!lastPredictionRows.length) {
        comparisonResult.innerHTML = "<p class=\"muted\">Модель выполнена, но явные predictions в ответе не найдены. Подробности доступны в JSON.</p>";
        return;
    }
    const rowsHtml = lastPredictionRows.map((row) => `
        <tr>
            <td>${escapeHtml(row.source_file || row.sample_id)}</td>
            <td><strong>${escapeHtml(row.predicted)}</strong></td>
            <td>${escapeHtml(row.confidence)}</td>
        </tr>
    `).join("");
    comparisonResult.innerHTML = `
        <div class="line-controls wrap">
            <button type="button" class="btn" data-download-predictions-csv>Скачать CSV результатов</button>
            <button type="button" class="btn" data-download-result-json>Скачать JSON результата</button>
        </div>
        <table class="comparison-table">
            <thead><tr><th>Файл</th><th>Предсказанный класс / значение</th><th>Достоверность / вероятность</th></tr></thead>
            <tbody>${rowsHtml}</tbody>
        </table>
    `;
    const counts = {};
    lastPredictionRows.forEach((row) => {
        counts[row.predicted] = (counts[row.predicted] || 0) + 1;
    });
    const labels = Object.keys(counts);
    if (labels.length && labels.length <= 30) {
        newPlot("plot", [{ type: "bar", x: labels, y: labels.map((label) => counts[label]) }], {
            title: "Распределение предсказаний",
            xaxis: { title: "Предсказанное значение / класс" },
            yaxis: { title: "Количество спектров" },
            height: 560,
        });
    }
}

function renderStructuredResult(payload, mode) {
    lastResultPayload = payload;
    clearStructuredBlocks();
    if (resultsSection) resultsSection.style.display = "";
    if (mode === "train") {
        if (payload.task_type === "comparison" || payload.run?.run_type === "comparison") {
            renderComparisonResult(payload);
            scrollToResults();
            return;
        }
        const saved = payload.saved_model || {};
        if (resultExplainer) {
            const bestMetric = payload.metrics?.f1_macro ?? payload.metrics?.accuracy ?? payload.metrics?.r2 ?? payload.metrics?.silhouette_score ?? payload.metrics?.inertia ?? payload.validation?.metrics?.f1_macro ?? payload.validation?.metrics?.accuracy ?? "н/д";
            resultExplainer.textContent = [
                "Анализ завершён.",
                `метод: ${payload.model_type || saved.model_type || "н/д"}`,
                `задача: ${payload.task_type || "н/д"}`,
                `dataset_id: ${payload.dataset_id || activeDatasetId || "н/д"}`,
                `версия данных: ${payload.dataset_version || (trainDatasetVersionInput?.value || "raw")}`,
                `target: ${importedDatasetSummary?.target_name || payload.target_name || "н/д"}`,
                `статус: ${payload.status || "success"}`,
                `лучшая метрика: ${typeof bestMetric === "number" ? bestMetric.toFixed(4) : bestMetric}`,
                `run_id: ${payload.run_id || payload.run?.run_id || "н/д"}`,
                `Модель сохранена: ${saved.model_id || "unknown"} (${saved.model_type || "n/a"}).`,
                "",
                buildResultInterpretation(payload),
                payload.warnings?.length ? `ограничения/предупреждения:\n- ${payload.warnings.join("\n- ")}` : "",
            ].join("\n");
        }
        lastRenderedRunId = payload.run_id || payload.run?.run_id || null;
        renderValidationBlock(payload.validation);
        renderMetricsTable(payload.metrics ? [{ method: payload.model_type || saved.model_type || "model", status: "success", metrics: payload.metrics }] : []);
        if (payload.task_type === "regression" && Array.isArray(payload.predictions) && payload.predictions.length) {
            appendRegressionPredictionsTable(payload.predictions);
        }
        updateResultDownloads(lastRenderedRunId);
        showResultTab("summary");
        scrollToResults();
        return;
    }

    if (mode === "infer") {
        const meta = payload.model_metadata || {};
        const filesCount = Array.isArray(payload.batch_results) ? payload.batch_results.length : 1;
        if (resultExplainer) resultExplainer.textContent = `Инференс выполнен.\nМодель: ${meta.model_id || "n/a"} (${meta.model_type || "n/a"})\nОбработано файлов: ${filesCount}`;
        renderInferenceExplanation(payload);
        renderPredictionsTable(payload);
        updateResultDownloads(lastRenderedRunId);
        showResultTab("summary");
        scrollToResults();
    }
}

function buildResultInterpretation(payload = {}) {
    const model = payload.model_type || payload.saved_model?.model_type || "";
    const task = payload.task_type || "";
    const metrics = payload.metrics || {};
    const n = payload.summary?.n_samples || importedDatasetSummary?.n_samples || "";
    if (model === "pca") {
        const variance = metrics.explained_variance_total;
        return [
            "Как понимать результат:",
            `PCA показывает главные направления изменчивости спектров${typeof variance === "number" ? `; первые компоненты объясняют ${(variance * 100).toFixed(1)}% дисперсии.` : "."}`,
            "Если точки на PC1-PC2 образуют отдельные группы, в спектрах есть выраженная структура. Если группы перемешаны, различия слабые или требуют моделей с target.",
        ].join("\n");
    }
    if (["kmeans", "hca"].includes(model) || task === "clustering") {
        const clusters = metrics.n_clusters ?? payload.result?.n_clusters ?? "н/д";
        const silhouette = metrics.silhouette_score;
        const inertia = metrics.inertia;
        return [
            "Как понимать результат:",
            `Алгоритм разделил ${n || "набор"} спектров на ${clusters} кластер(а).`,
            silhouette !== undefined && silhouette !== null
                ? `Silhouette score = ${Number(silhouette).toFixed(4)}. Значения ближе к 1 означают более выраженное разделение, около 0 — перекрытие кластеров.`
                : "Silhouette score не рассчитан: вероятно, кластеров слишком мало или структура не подходит для метрики.",
            inertia !== undefined ? `Inertia = ${Number(inertia).toFixed(4)}. Её удобно сравнивать между разными значениями n_clusters.` : "",
            "Для физической интерпретации сравните средние спектры кластеров и распределение ширины щели/решётки по кластерам.",
        ].filter(Boolean).join("\n");
    }
    if (["plsda", "svm", "random_forest", "decision_tree"].includes(model) || task === "classification") {
        const f1 = metrics.f1_macro ?? payload.validation?.metrics?.f1_macro;
        return [
            "Как понимать результат:",
            f1 !== undefined ? `F1-score macro = ${Number(f1).toFixed(4)}.` : "Основные метрики смотрите в таблице.",
            "F1-score macro удобен при сравнении классов, потому что учитывает баланс precision и recall по каждому классу.",
            "Confusion matrix показывает, где модель путает реальные классы с предсказанными.",
        ].join("\n");
    }
    if (["pls", "svr"].includes(model) || task === "regression") {
        const r2 = metrics.r2;
        const rmse = metrics.rmse;
        return [
            "Как понимать результат:",
            r2 !== undefined ? `R² = ${Number(r2).toFixed(4)}.` : "R² показывает долю объяснённой изменчивости target.",
            rmse !== undefined ? `RMSE = ${Number(rmse).toFixed(4)}.` : "",
            "Чем ближе точки к диагонали на графике истинных и предсказанных значений, тем точнее модель. График остатков помогает увидеть систематические ошибки.",
        ].filter(Boolean).join("\n");
    }
    return "Как понимать результат:\nСмотрите таблицу метрик и графики. Если графики отсутствуют, метод не сформировал визуализацию для текущих данных.";
}

function appendRegressionPredictionsTable(predictions) {
    if (!comparisonResult || !Array.isArray(predictions) || !predictions.length) return;
    lastPredictionRows = predictions.map((row, index) => ({
        sample_id: row.sample_id || `sample_${index + 1}`,
        y_true: row.y_true ?? "",
        y_pred: row.y_pred ?? "",
        residual: row.residual ?? "",
    }));
    const rowsHtml = lastPredictionRows.map((row) => `
        <tr>
            <td>${escapeHtml(row.sample_id)}</td>
            <td>${typeof row.y_true === "number" ? row.y_true.toFixed(4) : escapeHtml(row.y_true)}</td>
            <td>${typeof row.y_pred === "number" ? row.y_pred.toFixed(4) : escapeHtml(row.y_pred)}</td>
            <td>${typeof row.residual === "number" ? row.residual.toFixed(4) : escapeHtml(row.residual)}</td>
        </tr>
    `).join("");
    comparisonResult.insertAdjacentHTML("beforeend", `
        <div class="line-controls wrap" style="margin-top:12px;">
            <button type="button" class="btn" data-download-predictions-csv>Скачать predictions CSV</button>
            <button type="button" class="btn" data-download-result-json>Скачать result JSON</button>
        </div>
        <table class="comparison-table">
            <thead><tr><th>sample_id</th><th>y_true</th><th>y_pred</th><th>residual</th></tr></thead>
            <tbody>${rowsHtml}</tbody>
        </table>
    `);
}

function renderComparisonResult(payload) {
    if (resultsSection) resultsSection.style.display = "";
    lastResultPayload = payload;
    const run = payload.run || {};
    const rows = payload.result?.method_results || payload.result?.comparison || payload.metrics?.comparison || run.results?.method_results || [];
    const best = payload.best_method || run.best_method;
    lastComparisonRows = Array.isArray(rows) ? rows : [];
    lastRenderedRunId = payload.run_id || run.run_id || null;
    if (resultExplainer) {
        const bestRow = lastComparisonRows.find((row) => row.method === best || row.model_type === best) || {};
        const bestMetrics = bestRow.metrics || bestRow;
        const bestMetric = bestMetrics.f1_macro ?? bestMetrics.accuracy ?? bestMetrics.r2 ?? bestMetrics.silhouette_score ?? bestMetrics.inertia ?? bestMetrics.adjusted_rand_score ?? "н/д";
        resultExplainer.textContent = [
            "Сравнение методов завершено.",
            `метод: ${best || "сравнение"}`,
            `задача: ${payload.task_type || run.run_type || "comparison"}`,
            `dataset_id: ${payload.dataset_id || run.dataset_id || activeDatasetId || "н/д"}`,
            `run_id: ${payload.run_id || run.run_id || "н/д"}`,
            `версия данных: ${payload.dataset_version || (run.used_processed_data ? "processed" : "raw")}`,
            `target: ${run.target_name || payload.summary?.target_name || "н/д"}`,
            `статус: ${payload.status || run.status || "success"}`,
            `лучшая модель: ${best || "н/д"}`,
            `лучшая метрика: ${typeof bestMetric === "number" ? bestMetric.toFixed(4) : bestMetric}`,
            "",
            "Как понимать результат:",
            "Сравните методы по основной метрике в таблице. Для классификации ориентируйтесь на F1 macro, для регрессии — на R²/RMSE, для кластеризации — на silhouette score и распределение кластеров.",
            payload.warnings?.length ? `предупреждения:\n- ${payload.warnings.join("\n- ")}` : "",
        ].filter(Boolean).join("\n");
    }
    renderMetricsTable(rows, best);
    const comparisonBar = buildComparisonChart(rows);
    renderResultPlots([comparisonBar, ...(payload.plots || run.plots || [])].filter(Boolean));
        updateResultDownloads(lastRenderedRunId);
        showResultTab("summary");
    const bestModel = (payload.saved_models || run.results?.method_results || rows || []).find((item) => item.method === best || item.model_type === best);
    const modelId = bestModel?.model_id;
    const modelType = bestModel?.model_type || best;
    if (comparisonResult && modelId && modelType) {
        comparisonResult.insertAdjacentHTML(
            "beforeend",
            `<div class="line-controls wrap" style="margin-top:12px;">
                <button type="button" class="btn btn-primary" data-load-best-model="${escapeHtml(modelType)}:${escapeHtml(modelId)}">Загрузить лучшую модель</button>
            </div>`
        );
    }
}

function buildComparisonChart(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return;
    const metric = rows.some((row) => row.metrics?.f1_macro !== undefined || row.f1_macro !== undefined)
        ? "f1_macro"
        : rows.some((row) => row.metrics?.accuracy !== undefined || row.accuracy !== undefined)
            ? "accuracy"
            : rows.some((row) => row.metrics?.r2 !== undefined || row.r2 !== undefined)
                ? "r2"
                        : rows.some((row) => row.metrics?.silhouette_score !== undefined || row.silhouette_score !== undefined)
                        ? "silhouette_score"
                        : rows.some((row) => row.metrics?.adjusted_rand_score !== undefined || row.adjusted_rand_score !== undefined)
                            ? "adjusted_rand_score"
                            : rows.some((row) => row.metrics?.inertia !== undefined || row.inertia !== undefined)
                                ? "inertia"
                                : null;
    if (!metric) return;
    const x = rows.map((row) => row.method || row.model_type || "model");
    const y = rows.map((row) => Number((row.metrics || row)[metric] || 0));
    return {
        plot_id: "comparison_bar",
        data: [{ type: "bar", x, y, marker: { opacity: 0.85 } }],
        layout: {
        title: `Сравнение методов по метрике ${metric}`,
        xaxis: { title: "Метод" },
        yaxis: { title: metric },
        height: 560,
        },
    };
}

function renderMetricsTable(rows, bestMethod = "") {
    if (resultsSection) resultsSection.style.display = "";
    if (!comparisonResult) return;
    if (!Array.isArray(rows) || rows.length === 0) {
        comparisonResult.innerHTML = "";
        return;
    }
    lastComparisonRows = rows;
    const metricKeys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "r2", "mae", "rmse", "n_clusters", "inertia", "silhouette_score", "cluster_distribution", "adjusted_rand_score", "normalized_mutual_info_score"];
    const activeKeys = metricKeys.filter((key) => rows.some((row) => row[key] !== undefined || row.metrics?.[key] !== undefined));
    const head = ["Метод", ...activeKeys, "Статус"].map((v) => `<th>${escapeHtml(v)}</th>`).join("");
    const body = rows.map((row) => {
        const metrics = row.metrics || row;
        const cells = activeKeys.map((key) => {
            const value = metrics[key];
            return `<td>${typeof value === "number" ? value.toFixed(4) : escapeHtml(typeof value === "object" && value !== null ? JSON.stringify(value) : (value ?? ""))}</td>`;
        }).join("");
        const cls = row.method === bestMethod ? "comparison-best-row" : "";
        return `<tr class="${cls}"><td><strong>${escapeHtml(row.method || row.model_type || "model")}</strong></td>${cells}<td>${escapeHtml(row.status || "success")}</td></tr>`;
    }).join("");
    comparisonResult.innerHTML = `
        <table class="comparison-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
        <div class="metric-help">${metricHelpText(activeKeys)}</div>
    `;
}

function metricHelpText(keys = []) {
    const lines = [];
    if (keys.includes("f1_macro")) lines.push("F1 macro учитывает баланс precision и recall по каждому классу и удобен при несбалансированных классах.");
    if (keys.includes("silhouette_score")) lines.push("Silhouette score показывает отделимость кластеров: ближе к 1 — лучше, около 0 — кластеры перекрываются.");
    if (keys.includes("inertia")) lines.push("Inertia — суммарная внутрикластерная ошибка; её полезно сравнивать при разных n_clusters.");
    if (keys.includes("r2")) lines.push("R² показывает качество аппроксимации: чем ближе к 1, тем лучше.");
    if (keys.includes("rmse")) lines.push("RMSE показывает типичный масштаб ошибки прогноза в единицах target.");
    return escapeHtml(lines.join(" "));
}

function renderPlot(plotPayload) {
    if (resultsSection && plotPayload?.data) resultsSection.style.display = "";
    if (!plotPayload || !plotPayload.data) {
        if (typeof Plotly !== "undefined") Plotly.purge("plot");
        const plot = document.getElementById("plot");
        if (plot) {
            plot.classList.add("is-empty");
            plot.textContent = "Для этого результата графики не сформированы";
        }
        return;
    }
    const plot = document.getElementById("plot");
    if (plot) {
        plot.classList.remove("is-empty");
        plot.classList.remove("plot-stack");
        plot.textContent = "";
    }
    newPlot("plot", plotPayload.data, plotPayload.layout || {});
    markStepDone("results");
}

function normalizePlotList(plots) {
    if (!plots) return [];
    if (Array.isArray(plots)) return plots.filter((plot) => plot?.data);
    if (typeof plots === "object") return Object.values(plots).filter((plot) => plot?.data);
    return [];
}

function renderResultPlots(plots = []) {
    const plotBox = document.getElementById("plot");
    if (!plotBox) return;
    const normalized = normalizePlotList(plots);
    lastResultPlots = normalized;
    if (typeof Plotly !== "undefined") {
        plotRegistry.forEach((id) => {
            if (id.startsWith("result-plot-")) {
                const node = document.getElementById(id);
                if (node) Plotly.purge(node);
            }
        });
    }
    plotBox.innerHTML = "";
    if (!normalized.length) {
        plotBox.classList.add("is-empty");
        plotBox.textContent = "Для этого результата графики не сформированы";
        return;
    }
    plotBox.classList.remove("is-empty");
    plotBox.classList.add("plot-stack");
    normalized.forEach((plot, index) => {
        const id = `result-plot-${index}`;
        const frame = document.createElement("div");
        frame.id = id;
        frame.className = "plot-frame";
        plotBox.appendChild(frame);
        newPlot(id, plot.data, plot.layout || { title: plot.title || `График ${index + 1}` }).catch((error) => {
            frame.classList.add("is-empty");
            frame.textContent = `График не построен: ${error.message || error}`;
        });
    });
    markStepDone("results");
}

function getSpectrumY(item) {
    return item?.y || item?.intensity || item?.raw || item?.processed || [];
}

function getSelectedSampleIds(dataset, linesCount = "5", selectionStrategy = "balanced_by_class") {
    const spectra = Array.isArray(dataset?.spectra) ? dataset.spectra : [];
    if (!spectra.length) return [];
    if (String(linesCount) === "all") return spectra.map((item) => item.sample_id);
    const limit = Math.max(1, Math.min(Number(linesCount || 5), spectra.length));
    const strategy = selectionStrategy === "first" ? "first_n" : selectionStrategy === "balanced" ? "balanced_by_class" : selectionStrategy;
    if (strategy === "first_n") return spectra.slice(0, limit).map((item) => item.sample_id);
    if (strategy === "balanced_by_class" && spectra.some((item) => item.target !== null && item.target !== undefined)) {
        const groups = {};
        spectra.forEach((item) => {
            const key = String(item.target ?? "без target");
            groups[key] = groups[key] || [];
            groups[key].push(item);
        });
        const selected = [];
        while (selected.length < limit && Object.values(groups).some((items) => items.length)) {
            Object.keys(groups).sort().forEach((key) => {
                if (groups[key].length && selected.length < limit) selected.push(groups[key].shift());
            });
        }
        return selected.map((item) => item.sample_id);
    }
    const copy = [...spectra];
    for (let i = copy.length - 1; i > 0; i -= 1) {
        const j = Math.floor((Math.sin(i * 9301 + 49297) * 233280 % 1 + 1) % 1 * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy.slice(0, limit).map((item) => item.sample_id);
}

function buildRawTraces(dataset, sampleIds = []) {
    const axis = dataset?.axis || dataset?.axis_raw || [];
    const allowed = new Set(sampleIds);
    return (dataset?.spectra || []).filter((item) => allowed.has(item.sample_id)).map((item) => ({
        x: axis,
        y: item.raw || getSpectrumY(item),
        type: "scatter",
        mode: "lines",
        name: item.target ? `${item.sample_id} [${item.target}]` : item.sample_id,
        legendgroup: String(item.target || item.sample_id),
        hovertemplate: "sample_id: %{text}<br>x: %{x}<br>интенсивность: %{y}<extra></extra>",
        text: item.sample_id,
    }));
}

function buildProcessedTraces(dataset, sampleIds = []) {
    const axis = dataset?.axis_processed || dataset?.axis || [];
    const allowed = new Set(sampleIds);
    return (dataset?.spectra || []).filter((item) => allowed.has(item.sample_id)).map((item) => ({
        x: axis,
        y: item.processed || getSpectrumY(item),
        type: "scatter",
        mode: "lines",
        name: item.target ? `${item.sample_id} обработанный [${item.target}]` : `${item.sample_id} обработанный`,
        legendgroup: String(item.target || item.sample_id),
    }));
}

function buildBeforeAfterTraces(dataset, sampleIds = []) {
    const raw = buildRawTraces(dataset, sampleIds).map((trace) => ({ ...trace, name: `${trace.name} raw`, opacity: 0.55 }));
    const processed = buildProcessedTraces(dataset, sampleIds).map((trace) => ({ ...trace, name: `${trace.name}`, line: { dash: "dash" } }));
    return [...raw, ...processed];
}

function buildClassMeanTraces(dataset, source = "raw") {
    const spectra = Array.isArray(dataset?.spectra) ? dataset.spectra : [];
    const axis = source === "processed" ? (dataset?.axis_processed || dataset?.axis || []) : (dataset?.axis_raw || dataset?.axis || []);
    const groups = {};
    spectra.forEach((item) => {
        const key = String(item.target || "без target");
        const values = source === "processed" ? (item.processed || getSpectrumY(item)) : (item.raw || getSpectrumY(item));
        groups[key] = groups[key] || [];
        groups[key].push(values.map(Number));
    });
    return Object.entries(groups).map(([cls, rows]) => {
        const n = rows[0]?.length || 0;
        const y = Array.from({ length: n }, (_, idx) => rows.reduce((sum, row) => sum + Number(row[idx] || 0), 0) / rows.length);
        return { x: axis, y, type: "scatter", mode: "lines", name: `Средний спектр: ${cls}` };
    });
}

function datasetPreviewContainerId() {
    return datasetPreviewPlot ? "dataset-preview-plot" : "dataset-plot";
}

function datasetHasTarget(summary = importedDatasetSummary) {
    return !!(summary?.target_name || Object.keys(summary?.class_distribution || {}).length || summary?.target_type);
}

function safeDatasetPreviewStrategy(strategy) {
    if ((strategy === "balanced_by_class" || strategy === "balanced") && !datasetHasTarget()) {
        return "first_n";
    }
    return strategy === "first" ? "first_n" : strategy;
}

function renderDatasetPreviewPlot(containerId, previewData, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const spectra = Array.isArray(previewData?.spectra) ? previewData.spectra : [];
    if (!previewData?.axis?.length || !spectra.length) {
        container.classList.add("is-empty");
        container.textContent = "Датасет импортирован, но график не построен: отсутствуют preview-данные.";
        return;
    }
    container.classList.remove("is-empty");
    container.classList.remove("plot-stack");
    container.textContent = "";
    const linesCount = options.linesCount || "10";
    const strategy = options.selectionStrategy || "balanced_by_class";
    const sampleIds = getSelectedSampleIds(previewData, linesCount, strategy);
    const traces = buildRawTraces(previewData, sampleIds);
    newPlot(containerId, traces, {
        title: options.title || "Предпросмотр импортированного датасета",
        xaxis: { title: "Спектральная ось" },
        yaxis: { title: "Интенсивность" },
        height: 640,
    });
}

function rerenderActivePreviewPlot() {
    if (lastPreprocessingPreview) {
        renderPreprocessingPlot(lastPreprocessingPreview);
        return;
    }
    if (lastDatasetPreviewData) {
        renderDatasetPreviewPlot(datasetPreviewContainerId(), lastDatasetPreviewData, {
            linesCount: prePlotLimitInput?.value || "10",
            selectionStrategy: safeDatasetPreviewStrategy(prePlotStrategyInput?.value || "balanced_by_class"),
        });
    }
}

function renderRunPlots(plots = []) {
    if (!Array.isArray(plots) || plots.length === 0) {
        if (resultExplainer) {
            resultExplainer.textContent = `${resultExplainer.textContent}\n\nДля этого запуска сохранены только числовые результаты. Графики отсутствуют.`;
        }
        renderResultPlots([]);
        return;
    }
    renderResultPlots(plots);
}

function datasetFilesFormData() {
    const files = Array.from(datasetFilesInput.files || []);
    if (files.length === 0) {
        throw new Error("Выберите файл, архив или набор файлов для импорта.");
    }
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    return formData;
}

function scrollToResults() {
    document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showResultTab(name = "summary") {
    document.querySelectorAll(".result-tab").forEach((tab) => {
        tab.classList.toggle("result-tab--active", tab.dataset.resultTab === name);
    });
    document.querySelectorAll(".result-panel").forEach((panel) => {
        panel.style.display = panel.dataset.resultPanel === name ? "" : "none";
    });
}

function updateResultDownloads(runId = lastRenderedRunId) {
    if (!resultDownloads) return;
    const buttons = [
        lastPredictionRows.length
            ? `<button type="button" class="btn" data-download-predictions-csv>Скачать CSV результатов</button>`
            : `<button type="button" class="btn" data-download-comparison-csv>Скачать CSV результатов</button>`,
        `<button type="button" class="btn" data-download-result-json>Скачать JSON результата</button>`,
    ];
    if (runId) {
        buttons.push(`<button type="button" class="btn" data-run-export="${escapeHtml(runId)}">Скачать ZIP эксперимента</button>`);
        buttons.push(`<a class="btn" href="/analysis/reports/from-run/${encodeURIComponent(runId)}">Скачать HTML-отчёт</a>`);
    }
    resultDownloads.innerHTML = buttons.join("");
}

function setWorkflowMode(mode) {
    if (workflowModeInput && mode && workflowModeInput.value !== mode) {
        workflowModeInput.value = mode;
    }
    applyWorkflowMode();
}

function keepDatasetUploadVisible() {
    const mode = workflowModeInput?.value || "full";
    const importSection = document.getElementById("import-section");
    if (importSection && !["infer", "history"].includes(mode)) {
        importSection.style.display = "";
    }
    if (importResultCard && ["full", "preprocess"].includes(mode)) {
        importResultCard.style.display = "";
    }
}

function applyWorkflowMode() {
    const mode = workflowModeInput?.value || "full";
    const importSection = document.getElementById("import-section");
    const wizardCards = importSection ? Array.from(importSection.querySelectorAll(".card")).filter((card) => card.id !== "standard-import-card" && card.id !== "dataset-ready-card") : [];
    const standardCard = document.getElementById("standard-import-card");
    const preprocessingSection = document.getElementById("preprocessing-card");
    const trainingSection = document.getElementById("training-section");
    const modelsSection = document.getElementById("models-section");
    const inferSection = document.getElementById("infer-section");
    const modelApplyCard = document.getElementById("model-apply-card");
    const resultsSectionEl = document.getElementById("results-section");
    const runsSection = document.getElementById("runs-section");

    const show = (el, visible) => { if (el) el.style.display = visible ? "" : "none"; };
    show(importSection, !["infer", "history"].includes(mode));
    show(standardCard, ["standard", "train", "preprocess"].includes(mode));
    wizardCards.forEach((card) => show(card, ["full"].includes(mode)));
    show(preprocessingSection, ["full", "preprocess"].includes(mode));
    if (standardPreprocessToggleWrap) show(standardPreprocessToggleWrap, mode === "standard");
    if (compactPreprocessingRow) show(compactPreprocessingRow, mode === "standard");
    show(trainingSection, ["full", "standard", "train"].includes(mode));
    show(modelsSection, ["full", "standard", "train", "infer", "history"].includes(mode));
    show(inferSection, ["full", "infer"].includes(mode));
    show(modelApplyCard, ["full", "infer"].includes(mode));
    show(runsSection, ["full", "history"].includes(mode));
    show(resultsSectionEl, ["full", "standard", "train", "infer", "history"].includes(mode) && Boolean(lastResultPayload));
    keepDatasetUploadVisible();
    if (mode === "standard") {
        if (datasetPreviewResult) datasetPreviewResult.textContent = "Полный мастер импорта скрыт для правильного датасета.";
        if (datasetCheckSummary) datasetCheckSummary.textContent = "";
        if (datasetValidationResult) datasetValidationResult.textContent = "";
    } else if (mode === "full") {
        if (standardDatasetResult) standardDatasetResult.textContent = "Если файл уже приведён к единому виду, переключитесь в режим правильного датасета.";
    }
    if (mode === "history") refreshRuns();
    const hints = {
        full: "Показываются все блоки: импорт, предобработка, обучение, модели и результаты.",
        standard: "Загрузите уже унифицированный CSV/XLSX без полного мастера импорта.",
        preprocess: "Оставлены загрузка датасета, предобработка и экспорт обработанной версии.",
        train: "Оставлены загрузка правильного датасета, обучение и результаты.",
        infer: "Оставлены сохранённые модели, применение модели и результаты.",
        history: "Показана история: открыть запуск, скачать ZIP или удалить run.",
    };
    if (workflowModeHint) workflowModeHint.textContent = hints[mode] || hints.full;
    setWorkflowStepStatus("import", "active");
}

function revealPreprocessingSettings() {
    if (preprocessingCard) preprocessingCard.style.display = "block";
    preprocessingCard?.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function importStandardDataset() {
    const file = standardDatasetFileInput?.files?.[0];
    if (!file) {
        showToast("Выберите правильный CSV/XLSX датасет.", "warning");
        return;
    }
    if (importedDatasetId) {
        const ok = confirm("Текущий датасет будет заменён новым файлом. Продолжить?");
        if (!ok) return;
    }
    const doneBusy = setBusy(standardDatasetImportBtn, "Загрузка...");
    try {
        const formData = new FormData();
        formData.append("file", file);
        setGlobalStatus("Загрузка правильного датасета...", "Файл приводится к внутреннему формату X + axis + target.", 20);
        const response = await fetchWithTimeout("/analysis/dataset/import-standard", { method: "POST", body: formData });
        const data = await responseJsonOrError(response, "Не удалось выполнить предобработку.");
        if (!response.ok || data.status === "error") throw new Error(humanError(data, "Не удалось загрузить правильный датасет."));
        importedDatasetId = data.dataset_id;
        activeDatasetId = data.dataset_id;
        importedDatasetSummary = data.summary;
        processedDatasetReady = false;
        renderDatasetSummary(data.summary);
        renderDatasetReady(data.summary);
        renderAnalysisRecommendations(data.summary);
        useImportedDatasetForTraining();
        if (data.preview) {
            lastDatasetPreviewData = data.preview;
            renderDatasetPreviewPlot(datasetPreviewContainerId(), data.preview, { linesCount: "10", selectionStrategy: safeDatasetPreviewStrategy("balanced_by_class") });
        } else {
            await renderImportedSpectraPreview(importedDatasetId);
        }
        if (standardDatasetResult) {
            standardDatasetResult.textContent = [
                "Правильный датасет загружен.",
                `dataset_id: ${data.dataset_id}`,
                `спектров: ${data.summary?.n_samples}`,
                `признаков: ${data.summary?.n_features}`,
                `target: ${data.summary?.target_name || "нет"}`,
            ].join("\n");
        }
        clearGlobalStatus("Датасет готов.");
        markStepDone("import");
        showToast("success", "Правильный датасет загружен.");
    } catch (error) {
        if (standardDatasetResult) standardDatasetResult.textContent = error.message;
        clearGlobalStatus("Ошибка.");
        showError(error.message);
    } finally {
        doneBusy();
    }
}

function optionList(select, values, selectedValue = "") {
    if (!select) return;
    select.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "Не выбрано";
    select.appendChild(empty);
    values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        if (value === selectedValue) {
            option.selected = true;
        }
        select.appendChild(option);
    });
}

function selectedMultiValues(select) {
    if (!select) return [];
    return Array.from(select.selectedOptions || []).map((option) => option.value).filter(Boolean);
}

function selectedSheetNames() {
    const checked = Array.from(document.querySelectorAll("#dataset-sheets-checkboxes input[type='checkbox']:checked"))
        .map((input) => input.value);
    if (checked.length) return checked;
    return selectedMultiValues(datasetSheetsInput);
}

function setAllSheetCheckboxes(checked) {
    document.querySelectorAll("#dataset-sheets-checkboxes input[type='checkbox']").forEach((input) => {
        input.checked = checked;
    });
    Array.from(datasetSheetsInput.options || []).forEach((option) => {
        option.selected = checked;
    });
    renderImportCheckSummary();
}

function renderSheetCheckboxes(sheetNames) {
    if (!datasetSheetsCheckboxes) return;
    datasetSheetsCheckboxes.innerHTML = "";
    sheetNames.forEach((name) => {
        const label = document.createElement("label");
        label.className = "checkbox-item";
        label.innerHTML = `<input type="checkbox" value="${name}" checked> <span>${name}</span>`;
        const input = label.querySelector("input");
        input.addEventListener("change", () => {
            Array.from(datasetSheetsInput.options || []).forEach((option) => {
                if (option.value === name) option.selected = input.checked;
            });
            renderImportCheckSummary();
        });
        datasetSheetsCheckboxes.appendChild(label);
    });
}

function setTargetSourceOptions(layout) {
    if (!datasetTargetSourceInput) return;
    const current = datasetTargetSourceInput.value;
    const options = TARGET_SOURCE_OPTIONS[layout] || TARGET_SOURCE_OPTIONS.excel_rows;
    datasetTargetSourceInput.innerHTML = "";
    options.forEach(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        datasetTargetSourceInput.appendChild(option);
    });
    if (options.some(([value]) => value === current)) {
        datasetTargetSourceInput.value = current;
    } else {
        datasetTargetSourceInput.value = options[0][0];
    }
}

function updateDatasetLayoutCards() {
    const layout = datasetLayoutInput?.value;
    document.querySelectorAll(".layout-card[data-layout]").forEach((card) => {
        card.classList.toggle("layout-card--active", card.dataset.layout === layout);
    });
}

function updateDatasetLayoutUI() {
    const layout = datasetLayoutInput?.value || "excel_rows";
    setTargetSourceOptions(layout);
    const targetSource = datasetTargetSourceInput?.value || "none";

    document.querySelectorAll(".dataset-field[data-visible-for]").forEach((field) => {
        const visibleFor = (field.getAttribute("data-visible-for") || "").split(/\s+/);
        const sourceConstraint = field.getAttribute("data-target-source-field");
        const visible = visibleFor.includes(layout) && (!sourceConstraint || sourceConstraint === targetSource);
        field.style.display = visible ? "flex" : "none";
    });

    const hint = document.getElementById("dataset-feature-hint");
    if (hint) {
        if (layout === "excel_rows") {
            hint.textContent = "Спектральная ось будет автоматически получена из числовых заголовков колонок. Эти колонки станут спектральными признаками.";
        } else if (layout === "excel_columns") {
            hint.textContent = "Ось берётся из выбранной колонки, остальные выбранные столбцы становятся отдельными спектрами.";
        } else if (layout === "txt_folder") {
            hint.textContent = "Каждый TXT/CSV-файл будет прочитан как один спектр; при разных осях используется общая сетка.";
        }
    }

    updateDatasetLayoutCards();
    if (layout === "txt_folder" && targetSource === "file_name") {
        showToast("Каждый файл может образовать отдельный класс. Такой target непригоден для классификации; лучше выбрать 'Без target' или regex для группировки.", "warning");
    }
    renderImportCheckSummary();
}

function firstExcelSheetPreview() {
    const sheets = datasetPreviewPayload?.excel?.sheets || [];
    const selected = datasetSheetInput.value || sheets[0]?.name;
    return sheets.find((sheet) => sheet.name === selected) || sheets[0] || null;
}

function fillDatasetMappingControls(payload) {
    const excelSheets = payload.excel?.sheets || [];
    const sheetNames = excelSheets.map((sheet) => sheet.name);
    const recommendedSheet = excelSheets.find((item) => item.sheet_role === "dataset") || excelSheets[0];
    optionList(datasetSheetInput, sheetNames, recommendedSheet?.name || sheetNames[0] || "");
    datasetSheetsInput.innerHTML = "";
    sheetNames.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        option.selected = true;
        datasetSheetsInput.appendChild(option);
    });
    renderSheetCheckboxes(sheetNames);

    const sheet = recommendedSheet;
    const columns = sheet?.columns || [];
    optionList(datasetIdColumnInput, columns, sheet?.id_candidates?.[0] || "");
    optionList(datasetTargetColumnInput, columns, sheet?.numeric_target_candidates?.[0] || sheet?.target_candidates?.[0] || "");
    optionList(datasetAxisColumnInput, columns, sheet?.axis_candidates?.[0] || columns[0] || "");

    const suggested = payload.suggested_layout?.layout;
    if (suggested && datasetLayoutInput.querySelector(`option[value="${suggested}"]`)) {
        datasetLayoutInput.value = suggested;
    }
    setTargetSourceOptions(datasetLayoutInput.value);
    if (suggested === "excel_columns") {
        datasetTargetSourceInput.value = "sheet_name";
        if (datasetSheetModeInput) datasetSheetModeInput.value = sheetNames.length > 1 ? "sheet_as_class" : "single_sheet";
    } else if (suggested === "excel_rows") {
        datasetTargetSourceInput.value = "column";
        if (datasetSheetModeInput) datasetSheetModeInput.value = "single_sheet";
    } else if (suggested === "txt_folder") {
        datasetTargetSourceInput.value = "none";
    }
    updateDatasetLayoutUI();
    updateProcessingBlockVisibility();
}

function renderDatasetPreview(payload) {
    const suggested = payload.suggested_layout || {};
    const lines = [
        `Файлов: ${payload.files?.length || 0}`,
        `Найден тип данных: ${humanLayout(suggested.layout)}`,
        `Уверенность автоопределения: ${Math.round((suggested.confidence || 0) * 100)}%`,
        ...(suggested.reasons || []).map((reason) => `- ${reason}`),
    ];

    if (payload.excel?.sheets) {
        lines.push("Листы Excel:");
        payload.excel.sheets.forEach((sheet) => {
            lines.push(`- ${sheet.name}: строк ${sheet.rows}, колонок ${sheet.columns.length}; ${sheet.recommendation || sheet.sheet_role || ""}`);
            if (sheet.numeric_target_candidates?.length) lines.push(`  numeric target: ${sheet.numeric_target_candidates.join(", ")}`);
            if (sheet.class_target_candidates?.length) lines.push(`  target для классификации: ${sheet.class_target_candidates.join(", ")}`);
        });
    }
    if (payload.zip) {
        lines.push(`Спектральных файлов в ZIP: ${payload.zip.spectral_file_count}`);
        lines.push("Target: не задан по умолчанию");
        lines.push("Датасет подходит для: PCA, k-means, HCA, анализ параметров спектрометра");
        lines.push("Имена файлов будут сохранены как sample_id/source_file");
        const detected = detectedMeasurementCards({ metadata: { ...(payload.zip.metadata || {}), sample_metadata_preview: payload.zip.sample_metadata_preview || [] } });
        if (detected.length) {
            lines.push("Обнаруженные параметры:");
            detected.forEach(([label, value]) => lines.push(`- ${label}: ${value}`));
        }
        if (payload.zip.nested_archives.length) lines.push(`Вложенные архивы: ${payload.zip.nested_archives.join(", ")}`);
        renderDetectedMeasurementSummary({ metadata: { ...(payload.zip.metadata || {}), sample_metadata_preview: payload.zip.sample_metadata_preview || [] } });
    }
    datasetPreviewResult.textContent = lines.join("\n");
    renderImportCheckSummary();
}

function sheetStatsForRows(sheet) {
    if (!sheet) return null;
    const idColumn = datasetIdColumnInput.value || sheet.id_candidates?.[0] || "";
    const targetColumn = datasetTargetColumnInput.value || sheet.target_candidates?.[0] || "";
    const featureCount = (sheet.columns || []).filter((column) => {
        if (column === idColumn || column === targetColumn) return false;
        return /^[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?$/.test(String(column).trim());
    }).length;
    return { idColumn, targetColumn, featureCount };
}

function renderImportCheckSummary(validationSummary = null, warnings = []) {
    if (!datasetCheckSummary) return;
    const layout = datasetLayoutInput?.value || datasetPreviewPayload?.suggested_layout?.layout || "manual";
    const lines = [`Найден тип данных:\n${humanLayout(layout)}`, "", "Будет импортировано:"];

    if (validationSummary) {
        const distribution = validationSummary.class_distribution || {};
        lines.push(`- спектров: ${validationSummary.n_samples}`);
        lines.push(`- спектральных точек: ${validationSummary.n_features}`);
        lines.push(`- диапазон оси: ${formatNumber(validationSummary.axis_min)}-${formatNumber(validationSummary.axis_max)}`);
        if (layout === "excel_rows") {
            lines.push(`- ID-колонка: ${datasetIdColumnInput.value || "не выбрана"}`);
            lines.push(`- target-колонка: ${validationSummary.target_name || "без target"}`);
        }
        if (validationSummary.target_name && Object.keys(distribution).length) {
            lines.push("- классы:");
            Object.entries(distribution).slice(0, 12).forEach(([cls, count]) => lines.push(`  ${cls} — ${count}`));
            if (Object.keys(distribution).length > 12) lines.push(`  ... ещё ${Object.keys(distribution).length - 12}`);
        } else {
            lines.push("- Target: не задан");
            lines.push("- Рекомендуемые задачи: PCA, кластеризация, анализ параметров спектрометра");
        }
        const detected = detectedMeasurementCards(validationSummary);
        if (detected.length) {
            lines.push("- обнаруженные параметры измерений:");
            detected.forEach(([label, value]) => lines.push(`  ${label}: ${value}`));
            renderDetectedMeasurementSummary(validationSummary);
        }
    } else if (datasetPreviewPayload?.excel?.sheets && layout === "excel_rows") {
        const sheet = firstExcelSheetPreview();
        const stats = sheetStatsForRows(sheet);
        lines.push(`- строк на выбранном листе: ${sheet?.rows ?? "н/д"}`);
        lines.push(`- спектральных признаков: ${stats?.featureCount ?? "н/д"}`);
        lines.push(`- ID-колонка: ${stats?.idColumn || "не найдена"}`);
        lines.push(`- target-колонка: ${stats?.targetColumn || "без target"}`);
    } else if (datasetPreviewPayload?.excel?.sheets && layout === "excel_columns") {
        const selectedSheets = selectedSheetNames();
        const sheet = firstExcelSheetPreview();
        lines.push(`- листы: ${(selectedSheets.length ? selectedSheets : (datasetPreviewPayload.excel.sheets || []).map((item) => item.name)).join(", ")}`);
        lines.push(`- режим листов: ${datasetSheetModeInput?.selectedOptions?.[0]?.textContent || "не выбран"}`);
        lines.push(`- колонка оси: ${datasetAxisColumnInput.value || sheet?.columns?.[0] || "не выбрана"}`);
        lines.push("- спектры: все числовые столбцы кроме оси");
    } else if (layout === "txt_folder" && datasetPreviewPayload?.zip) {
        lines.push(`- файлов спектров в ZIP: ${datasetPreviewPayload.zip.spectral_file_count}`);
        const targetSource = datasetTargetSourceInput?.value || "none";
        if (targetSource === "none") {
            lines.push("- Target: не задан");
            lines.push("- Доступны PCA, k-means, HCA и анализ параметров спектрометра");
        } else if (targetSource === "file_name") {
            lines.push("- target: имя файла");
            lines.push("- предупреждение: каждый файл может образовать отдельный класс; для классификации это непригодно");
        } else if (targetSource === "filename_regex") {
            lines.push(`- target: regex ${datasetTargetRegexInput?.value || "не задан"}`);
        } else {
            lines.push(`- target: ${targetSource}`);
        }
        const detected = detectedMeasurementCards({ metadata: { ...(datasetPreviewPayload.zip.metadata || {}), sample_metadata_preview: datasetPreviewPayload.zip.sample_metadata_preview || [] } });
        if (detected.length) {
            lines.push("- обнаруженные параметры:");
            detected.forEach(([label, value]) => lines.push(`  ${label}: ${value}`));
        }
        lines.push(`- общая сетка: ${datasetGridModeInput.value === "first_axis" ? "ось первого спектра" : "пересечение диапазонов"}`);
    } else {
        lines.push("- нажмите «Предпросмотр», чтобы увидеть найденные колонки и размеры.");
    }

    if (warnings.length) {
        lines.push("", "Предупреждения:");
        warnings.forEach((warning) => lines.push(`- ${friendlyWarning(warning)}`));
    }
    datasetCheckSummary.textContent = lines.join("\n");
}

function buildImportConfig() {
    if (!datasetLayoutInput || !datasetTargetSourceInput) {
        return { layout: "excel_rows", target_source: "none", measurement_metadata: buildMeasurementMetadata() };
    }
    const layout = datasetLayoutInput.value;
    const targetSource = datasetTargetSourceInput.value;
    const config = {
        layout,
        target_source: targetSource,
        filename_regex: targetSource === "filename_regex" ? (datasetTargetRegexInput?.value.trim() || undefined) : undefined,
        manual_target: datasetManualTargetInput.value.trim() || undefined,
        measurement_metadata: buildMeasurementMetadata(),
    };

    if (layout === "excel_columns") {
        config.sheets = datasetSheetModeInput?.value === "single_sheet" ? [datasetSheetInput.value] : selectedSheetNames();
        config.sheet_mode = datasetSheetModeInput?.value || "sheet_as_class";
        config.axis_column = datasetAxisColumnInput.value || undefined;
        config.sample_columns = "all_except_axis";
        config.interpolation = {
            enabled: true,
            grid_mode: datasetGridModeInput.value || "first_axis",
            step: "auto",
        };
    } else if (layout === "excel_rows") {
        config.sheet = datasetSheetInput.value || undefined;
        config.sheets = datasetSheetModeInput?.value === "single_sheet" ? [datasetSheetInput.value] : selectedSheetNames();
        config.sheet_mode = datasetSheetModeInput?.value || "single_sheet";
        config.id_column = datasetIdColumnInput.value || undefined;
        config.target_column = targetSource === "column" ? (datasetTargetColumnInput.value || undefined) : undefined;
        config.feature_columns = "numeric_headers";
    } else if (layout === "txt_folder") {
        config.interpolation = {
            enabled: true,
            grid_mode: datasetGridModeInput.value || "intersection",
            step: "auto",
        };
    } else if (layout === "long_table") {
        config.status = "planned";
    }
    return config;
}

function buildMeasurementMetadata() {
    return {
        slit_width: document.getElementById("meta-slit-width")?.value.trim() || undefined,
        grating_lines_per_mm: document.getElementById("meta-grating-lines")?.value || undefined,
        laser_wavelength: document.getElementById("meta-laser-wavelength")?.value.trim() || undefined,
        integration_time: document.getElementById("meta-integration-time")?.value.trim() || undefined,
        accumulation_count: document.getElementById("meta-accumulation-count")?.value || undefined,
        spectrometer_name: document.getElementById("meta-spectrometer-name")?.value.trim() || undefined,
    };
}

function configFormData() {
    const formData = datasetFilesFormData();
    formData.append("import_config", JSON.stringify(buildImportConfig()));
    return formData;
}

function friendlyWarning(message) {
    const text = String(message || "");
    const lower = text.toLowerCase();
    if (lower.includes("дубликат") || lower.includes("duplicate")) {
        return "Найдены повторяющиеся ID образцов. Это может быть нормально, если это повторные измерения одного образца.";
    }
    if (lower.includes("несбаланс")) {
        return "Классы несбалансированы. При обучении модели рекомендуется использовать стратифицированное разбиение.";
    }
    if (lower.includes("пропуск") || lower.includes("missing")) {
        return "Некоторые значения отсутствуют. Проверьте файл или используйте заполнение пропусков перед обучением.";
    }
    if (lower.includes("меньше 6")) {
        return "В датасете мало спектров, поэтому оценка качества модели может быть нестабильной.";
    }
    return text;
}

function renderDatasetValidation(validation) {
    if (!datasetValidationResult) return;
    if (importResultCard) importResultCard.style.display = "";
    const statusText = {
        ok: "Датасет готов к импорту.",
        warning: "Датасет можно импортировать, но есть предупреждения.",
        error: "Нужно исправить ошибки перед импортом.",
    }[validation.status] || `Статус: ${validation.status}`;
    const parts = [statusText];
    if (validation.errors?.length) parts.push(`Ошибки:\n- ${validation.errors.join("\n- ")}`);
    if (validation.warnings?.length) parts.push(`Предупреждения:\n- ${validation.warnings.map(friendlyWarning).join("\n- ")}`);
    datasetValidationResult.textContent = parts.join("\n\n");
    if (validation.summary) {
        renderDatasetSummary(validation.summary);
        renderImportCheckSummary(validation.summary, validation.warnings || []);
    }
}

function renderDatasetSummary(summary) {
    if (!datasetSummaryEl || !summary) return;
    if (importResultCard) importResultCard.style.display = "";
    const distribution = summary.class_distribution || {};
    const distributionLines = Object.keys(distribution).length
        ? Object.entries(distribution).map(([cls, count]) => `${cls} — ${count}`).join("\n")
        : "target не задан";
    const targetDetails = !summary.target_name
        ? "Target не задан. Доступны PCA, k-means, HCA и анализ параметров спектрометра."
        : summary.target_type === "numeric"
        ? [
            `Тип target: numeric`,
            `Диапазон target: ${formatNumber(summary.target_min)}-${formatNumber(summary.target_max)}`,
            `Уникальных уровней: ${summary.target_unique_count ?? "н/д"}`,
            Object.keys(distribution).length ? `Распределение уровней:\n${distributionLines}` : "",
            `Задача: регрессия`,
        ].filter(Boolean).join("\n")
        : `Классы:\n${distributionLines}`;
    const measurement = summary.measurement_metadata || {};
    const detectedMeasurementLines = detectedMeasurementCards(summary).map(([label, value]) => `${label}: ${value}`).join("\n");
    const measurementLines = Object.keys(measurement).length
        ? [
            `ширина щели: ${measurement.slit_width || "не указана"}`,
            `решётка: ${measurement.grating_lines_per_mm || "не указана"}`,
            `лазер: ${measurement.laser_wavelength || "не указан"}`,
            `время интеграции: ${measurement.integration_time || "не указано"}`,
            `накоплений: ${measurement.accumulation_count || "не указано"}`,
            `спектрометр: ${measurement.spectrometer_name || "не указан"}`,
            detectedMeasurementLines ? `обнаружено в файлах:\n${detectedMeasurementLines}` : "",
        ].join("\n")
        : (detectedMeasurementLines || "не указаны");
    renderDetectedMeasurementSummary(summary);
    datasetSummaryEl.textContent = [
        `dataset_id: ${summary.dataset_id || importedDatasetId || "ещё не сохранён"}`,
        `Версия: ${summary.version || "raw"}`,
        `Выбранные листы: ${(summary.selected_sheets || []).join(", ") || "н/д"}`,
        `Режим листов: ${summary.sheet_mode || "н/д"}`,
        `Спектров: ${summary.n_samples}`,
        `Спектральных точек: ${summary.n_features}`,
        `Диапазон оси: ${formatNumber(summary.axis_min)}-${formatNumber(summary.axis_max)}`,
        `Target: ${summary.target_name || "нет"}`,
        targetDetails,
        `NaN/Inf: ${summary.nan_count ?? 0}`,
        `Пропуски: ${formatPercent(summary.missing_fraction || 0)}`,
        `Пустых спектров: ${summary.empty_spectra_count ?? 0}`,
        `Отрицательные интенсивности: ${summary.has_negative_intensity ? "есть" : "нет"}`,
        `SNR средний/мин/макс: ${formatNumber(summary.snr?.mean)} / ${formatNumber(summary.snr?.min)} / ${formatNumber(summary.snr?.max)}`,
        `Технический статус: ${summary.technical_status || "н/д"}`,
        summary.technical_warnings?.length ? `Технические предупреждения:\n- ${summary.technical_warnings.join("\n- ")}` : "",
        `Параметры измерений:\n${measurementLines}`,
        `Первые sample_id: ${(summary.sample_ids_preview || []).join(", ")}`,
        `Источник: ${humanLayout(summary.source_layout)}`,
    ].join("\n");
}

function metadataHasSpectrometerParams(summary = {}) {
    const meta = summary.metadata || {};
    const preview = Array.isArray(meta.sample_metadata_preview) ? meta.sample_metadata_preview : [];
    return Boolean(
        meta.slit_width_um_values?.length ||
        meta.grating_lines_mm_values?.length ||
        preview.some((item) => item?.slit_width_um !== undefined || item?.grating_lines_mm !== undefined) ||
        summary.measurement_metadata?.slit_width ||
        summary.measurement_metadata?.grating_lines_per_mm
    );
}

function renderAndorHint(summary = {}) {
    if (!datasetAndorHint) return;
    if (!metadataHasSpectrometerParams(summary)) {
        datasetAndorHint.style.display = "none";
        datasetAndorHint.textContent = "";
        return;
    }
    datasetAndorHint.style.display = "";
    datasetAndorHint.innerHTML = `
        <p>Обнаружены параметры спектрометрической системы. Можно оценить влияние ширины щели и решётки на интенсивность, FWHM и SNR.</p>
        <button type="button" class="btn" disabled title="Будет доступно после добавления анализа параметров спектрометра">Анализ щели и решётки</button>
    `;
}

function renderDatasetReady(summary) {
    if (!datasetReadyCard || !datasetReadySummary || !summary) return;
    const distribution = summary.class_distribution || {};
    const classes = Object.keys(distribution).length
        ? Object.entries(distribution).map(([cls, count]) => `${cls} — ${count}`).join(", ")
        : "нет target";
    const targetInfo = summary.target_type === "numeric"
        ? `numeric, ${formatNumber(summary.target_min)}-${formatNumber(summary.target_max)}, уровней: ${summary.target_unique_count ?? "н/д"}`
        : classes;
    const tiles = [
        ["Спектров", summary.n_samples],
        ["Признаков", summary.n_features],
        ["Target", summary.target_name || "нет"],
        [summary.target_type === "numeric" ? "Тип target" : "Классы", targetInfo],
        ["Задача", summary.task_recommendation || "н/д"],
        ["Диапазон", `${formatNumber(summary.axis_min)}-${formatNumber(summary.axis_max)}`],
        ["NaN/Inf", summary.nan_count ?? 0],
        ["Статус", summary.technical_status || "ok"],
    ];
    const technical = [
        `dataset_id: ${summary.dataset_id || importedDatasetId || "н/д"}`,
        `Выбранные листы: ${(summary.selected_sheets || []).join(", ") || "н/д"}`,
        `Режим: ${summary.sheet_mode || "н/д"}`,
        `SNR средний/мин/макс: ${formatNumber(summary.snr?.mean)} / ${formatNumber(summary.snr?.min)} / ${formatNumber(summary.snr?.max)}`,
        summary.technical_warnings?.length ? `Предупреждения:\n- ${summary.technical_warnings.join("\n- ")}` : "",
    ].filter(Boolean).join("\n");
    datasetReadySummary.innerHTML = `
        <div class="dataset-summary-grid">
            ${tiles.map(([label, value]) => `
                <div class="summary-tile">
                    <div class="summary-label">${escapeHtml(label)}</div>
                    <div class="summary-value">${escapeHtml(value)}</div>
                </div>
            `).join("")}
        </div>
        <details>
            <summary>Показать технические сведения</summary>
            <pre>${escapeHtml(technical)}</pre>
        </details>
    `;
    datasetReadyCard.style.display = "block";
    if (datasetPreviewVersionInput) {
        Array.from(datasetPreviewVersionInput.options).forEach((option) => {
            if (option.value === "processed") option.disabled = !processedDatasetReady;
        });
        if (!processedDatasetReady && datasetPreviewVersionInput.value === "processed") {
            datasetPreviewVersionInput.value = "raw";
        }
    }
    if (preprocessingCard) {
        preprocessingCard.style.display = workflowModeInput?.value === "standard" ? "none" : "block";
    }
    renderAndorHint(summary);
    renderAnalysisRecommendations(summary);
    renderAnalysisWorkflowGuide(summary);
    updateModelModeUI();
    markStepDone("check");
}

function renderAnalysisRecommendations(summary) {
    if (!analysisRecommendations || !summary) return;
    const distribution = summary.class_distribution || {};
    const recommendations = summary.recommended_methods || [];
    const targetLine = summary.target_name && summary.target_type === "numeric"
        ? [
            `Найдена числовая целевая переменная: ${summary.target_name}.`,
            "По умолчанию используется как концентрация / регрессия.",
            summary.alternative_targets?.length ? `Альтернативный target для классификации: ${summary.alternative_targets.map((item) => item.name).join(", ")}.` : "",
            `Диапазон: ${formatNumber(summary.target_min)}-${formatNumber(summary.target_max)}; уникальных уровней: ${summary.target_unique_count ?? "н/д"}.`,
        ].filter(Boolean).join("\n")
        : summary.target_name
        ? `Для текущего датасета найден target: ${summary.target_name}. Классы/значения: ${Object.entries(distribution).map(([k, v]) => `${k} — ${v}`).join(", ") || "не указаны"}.`
        : "Target не задан. Рекомендуются методы разведочного анализа без учителя.";
    const lines = [targetLine, "", "Рекомендуемые методы:"];
    const buttons = [];
    recommendations.forEach((item, idx) => {
        const disabled = item.implemented === false || ["warning", "preprocessing"].includes(item.model_type);
        lines.push(`${idx + 1}. ${item.title || item.model_type} — ${item.reason || ""}${disabled ? " (информационно)" : ""}`);
        if (!disabled) {
            buttons.push(`<button type="button" class="btn btn-secondary" data-recommend-model="${item.model_type}">Выбрать ${item.title || item.model_type}</button>`);
        }
    });
    analysisRecommendations.innerHTML = `<pre>${escapeHtml(lines.join("\n"))}</pre>${buttons.length ? `<div class="line-controls wrap">${buttons.join("")}</div>` : ""}`;
}

const ANALYSIS_METHOD_INFO = {
    pca: ["PCA", "Сжимает спектры до нескольких главных компонент.", "Для визуального поиска групп, выбросов и общей структуры."],
    kmeans: ["k-means", "Делит спектры на заданное число кластеров.", "Для поиска групп без заранее заданных классов."],
    hca: ["HCA", "Строит иерархическую структуру сходства спектров.", "Для просмотра вложенных групп и близости образцов."],
    plsda: ["PLS-DA", "Классифицирует спектры через латентные переменные.", "Для категориального target и сравнения классов."],
    svm: ["SVM", "Классифицирует спектры в пространстве высокой размерности.", "Для устойчивого разделения сложных классов."],
    random_forest: ["Random Forest", "Использует ансамбль деревьев решений.", "Для устойчивой классификации и сравнения с линейными методами."],
    decision_tree: ["Decision Tree", "Строит простую интерпретируемую модель.", "Для быстрой базовой проверки."],
    pls: ["PLS Regression", "Прогнозирует числовой параметр по спектру.", "Для концентраций и других количественных величин."],
    svr: ["SVR", "Выполняет регрессию методом опорных векторов.", "Для нелинейной зависимости спектра и числового target."],
    compare_classification: ["Сравнить классификаторы", "Запускает несколько классификаторов.", "Для выбора лучшего метода по метрикам."],
    compare_regression: ["Сравнить регрессоры", "Запускает несколько регрессионных моделей.", "Для сравнения качества прогноза."],
    compare_clustering: ["Сравнить кластеризацию", "Сравнивает варианты кластеризации.", "Для подбора числа групп и метода."],
};

function targetState(summary = importedDatasetSummary) {
    const hasTarget = datasetHasTarget(summary);
    const targetType = summary?.target_type || "none";
    const numericTarget = hasTarget && targetType === "numeric";
    const categoricalTarget = hasTarget && !numericTarget;
    return { hasTarget, targetType, numericTarget, categoricalTarget };
}

function analysisTaskDefinitions(summary = importedDatasetSummary) {
    const { categoricalTarget, numericTarget } = targetState(summary);
    return [
        {
            goal: "explore",
            title: "Визуальный анализ",
            selectTitle: "Посмотреть структуру данных (PCA)",
            methods: "PCA",
            description: "Показать структуру спектров, группы и выбросы.",
            enabled: true,
            disabledReason: "",
        },
        {
            goal: "cluster",
            title: "Кластеризация",
            selectTitle: "Найти группы спектров (кластеризация)",
            methods: "k-means, HCA",
            description: "Найти группы спектров без заранее заданных классов.",
            enabled: true,
            disabledReason: "",
        },
        {
            goal: "classify",
            title: "Классификация",
            selectTitle: "Классифицировать спектры",
            methods: "PLS-DA, SVM, Random Forest, Decision Tree",
            description: "Обучить модель относить спектры к заданным классам.",
            enabled: categoricalTarget,
            disabledReason: "Нужен категориальный target.",
        },
        {
            goal: "regress",
            title: "Регрессия",
            selectTitle: "Предсказать числовой параметр",
            methods: "PLS Regression, SVR",
            description: "Предсказать числовой параметр, например концентрацию.",
            enabled: numericTarget,
            disabledReason: "Нужен числовой target.",
        },
        {
            goal: "compare",
            title: "Сравнение методов",
            selectTitle: "Сравнить методы",
            methods: categoricalTarget ? "классификаторы" : numericTarget ? "регрессионные модели" : "нужен target",
            description: "Запустить несколько моделей и сравнить их по метрикам.",
            enabled: categoricalTarget || numericTarget,
            disabledReason: "Для сравнения моделей с target нужен target.",
        },
    ];
}

function methodsForAnalysisGoal(goal, summary = importedDatasetSummary) {
    const { categoricalTarget, numericTarget } = targetState(summary);
    if (goal === "cluster") return ["kmeans", "hca"];
    if (goal === "classify") return categoricalTarget ? ["plsda", "svm", "random_forest", "decision_tree"] : [];
    if (goal === "regress") return numericTarget ? ["pls", "svr"] : [];
    if (goal === "compare") {
        if (categoricalTarget) return ["compare_classification"];
        if (numericTarget) return ["compare_regression"];
        return [];
    }
    return ["pca"];
}

function defaultTaskForSummary(summary = importedDatasetSummary) {
    const { categoricalTarget, numericTarget } = targetState(summary);
    if (categoricalTarget) return "classify";
    if (numericTarget) return "regress";
    return "cluster";
}

function defaultMethodForTask(goal, summary = importedDatasetSummary) {
    const methods = methodsForAnalysisGoal(goal, summary);
    if (goal === "compare") {
        const { categoricalTarget, numericTarget } = targetState(summary);
        if (categoricalTarget) return "compare_classification";
        if (numericTarget) return "compare_regression";
        return null;
    }
    return methods[0] || null;
}

function goalForModelType(modelType) {
    if (["compare_classification", "compare_regression", "compare_clustering"].includes(modelType)) return "compare";
    if (["kmeans", "hca", "compare_clustering"].includes(modelType)) return "cluster";
    if (["plsda", "svm", "random_forest", "decision_tree"].includes(modelType)) return "classify";
    if (["pls", "svr"].includes(modelType)) return "regress";
    return "explore";
}

function analysisSelectionKey(summary = importedDatasetSummary) {
    if (!summary) return "empty";
    return [
        summary.dataset_id || activeDatasetId || "no-id",
        summary.target_name || "no-target",
        summary.target_type || "none",
        summary.n_samples || 0,
        summary.n_features || 0,
    ].join("|");
}

function syncAnalysisSelectionToInputs() {
    if (analysisGoalInput && selectedAnalysisTask) analysisGoalInput.value = selectedAnalysisTask;
    if (modelTypeInput && selectedAnalysisMethod) modelTypeInput.value = selectedAnalysisMethod;
}

function ensureAnalysisSelection(summary = importedDatasetSummary) {
    const key = analysisSelectionKey(summary);
    const tasks = analysisTaskDefinitions(summary);
    const changedDataset = key !== lastAnalysisSelectionKey;
    if (changedDataset) {
        selectedAnalysisTask = defaultTaskForSummary(summary);
        selectedAnalysisMethod = defaultMethodForTask(selectedAnalysisTask, summary);
        lastAnalysisSelectionKey = key;
    }
    if (!selectedAnalysisTask) {
        selectedAnalysisTask = defaultTaskForSummary(summary);
    }
    const task = tasks.find((item) => item.goal === selectedAnalysisTask);
    if (!task?.enabled) {
        selectedAnalysisTask = defaultTaskForSummary(summary);
    }
    const methods = methodsForAnalysisGoal(selectedAnalysisTask, summary);
    if (selectedAnalysisTask === "compare") {
        selectedAnalysisMethod = defaultMethodForTask("compare", summary);
    } else if (!selectedAnalysisMethod || !methods.includes(selectedAnalysisMethod)) {
        selectedAnalysisMethod = defaultMethodForTask(selectedAnalysisTask, summary);
    }
    syncAnalysisSelectionToInputs();
    return { task: selectedAnalysisTask, method: selectedAnalysisMethod };
}

function renderAnalysisWorkflowGuide(summary = importedDatasetSummary) {
    if (!analysisStepGuide) return;
    const { hasTarget, numericTarget, categoricalTarget } = targetState(summary);
    const { task: selectedGoal, method: selectedModel } = ensureAnalysisSelection(summary);
    const tasks = analysisTaskDefinitions(summary);
    const taskOptions = tasks.map((task) => `
        <option value="${escapeHtml(task.goal)}" ${selectedGoal === task.goal ? "selected" : ""} ${task.enabled ? "" : "disabled"}>
            ${escapeHtml(task.selectTitle || task.title)}${task.enabled ? "" : ` — недоступно: ${task.disabledReason}`}
        </option>
    `).join("");
    const methods = methodsForAnalysisGoal(selectedGoal, summary);
    const methodCards = selectedGoal === "compare"
        ? `<div class="analysis-empty-note">Будет запущено сравнение доступных методов.</div>`
        : methods.length ? methods.map((method) => {
        const [title, description, usage] = ANALYSIS_METHOD_INFO[method] || [method, "", ""];
        const active = selectedModel === method;
        return `
            <button type="button" class="analysis-method-pill ${active ? "is-active" : ""}" data-recommend-model="${escapeHtml(method)}">
                <div class="method-card-header">
                    <div class="method-title">${escapeHtml(title)}</div>
                    ${active ? `<span class="method-badge selected">✓ Выбрано</span>` : ""}
                </div>
                <p class="method-description">${escapeHtml(description)}</p>
                <span class="method-use">${escapeHtml(usage)}</span>
            </button>
        `;
    }).join("") : `<div class="analysis-empty-note">Для выбранной задачи сейчас нет доступных методов: проверьте target в датасете.</div>`;
    const targetLine = hasTarget
        ? `Target: ${summary?.target_name || "target"} · тип: ${numericTarget ? "числовой" : "категориальный"} · доступны ${categoricalTarget ? "классификация и сравнение методов" : "регрессия и сравнение методов"}`
        : "Target не задан. Доступны PCA и кластеризация. Для классификации или регрессии нужен target";
    analysisStepGuide.innerHTML = `
        <div class="analysis-target-info ${hasTarget ? "analysis-target-info--ok" : "analysis-target-info--empty"}">${escapeHtml(targetLine)}</div>
        <section class="analysis-section analysis-section--compact">
            <h4>1. Что сделать?</h4>
            <label class="analysis-select-label">
                <span>Что сделать с датасетом?</span>
                <select id="analysis-task-select" class="input-field">
                    ${taskOptions}
                </select>
            </label>
        </section>
        <section class="analysis-section analysis-section--compact">
            <h4>2. Каким методом?</h4>
            <div class="analysis-method-list">${methodCards}</div>
            <div id="analysis-selection-summary" class="analysis-selection-summary"></div>
        </section>
        <section class="analysis-section analysis-section--compact">
            <h4>3. Параметры</h4>
            <p class="analysis-next-step">В простом режиме параметры подобраны автоматически. Для ручной настройки включите расширенный режим ниже.</p>
        </section>
    `;
    renderAnalysisRunSummary();
}

function renderAnalysisRunSummary() {
    if (!analysisRunSummary) return;
    ensureAnalysisSelection(importedDatasetSummary);
    const modelType = selectedAnalysisMethod || modelTypeInput?.value || "";
    const [modelTitle] = ANALYSIS_METHOD_INFO[modelType] || [modelType];
    const task = analysisTaskDefinitions(importedDatasetSummary).find((item) => item.goal === selectedAnalysisTask);
    const datasetVersion = trainDatasetVersionInput?.value === "processed" ? "processed" : "raw";
    const targetName = importedDatasetSummary?.target_name || "не задан";
    const simple = (modelConfigModeInput?.value || "simple") === "simple";
    const trainBtn = document.getElementById("train-btn");
    if (!selectedAnalysisTask || (!selectedAnalysisMethod && selectedAnalysisTask !== "compare")) {
        analysisRunSummary.innerHTML = `<div class="analysis-run-summary__main">Сначала выберите задачу анализа и метод.</div>`;
        const selectionSummary = document.getElementById("analysis-selection-summary");
        if (selectionSummary) selectionSummary.textContent = "Сначала выберите задачу анализа и метод.";
        if (trainBtn) {
            trainBtn.disabled = true;
            trainBtn.textContent = "Запустить анализ";
        }
        return;
    }
    const parts = [
        `Выбрано: ${task?.title || "задача"}${selectedAnalysisTask === "compare" ? "" : ` → ${modelTitle}`}`,
        `данные: ${datasetVersion}`,
        `target: ${targetName}`,
    ];
    const paramParts = [];
    if (["kmeans", "hca", "compare_clustering"].includes(modelType)) {
        paramParts.push(`число кластеров = ${document.getElementById("model-n-clusters")?.value || "2"}`);
    }
    if (["pca", "plsda", "pls"].includes(modelType)) {
        paramParts.push(`число компонент = ${nComponentsInput?.value || "авто"}`);
    }
    if (["kmeans", "svm", "random_forest", "decision_tree", "svr"].includes(modelType)) {
        paramParts.push(`случайное зерно = ${randomStateInput?.value || "42"}`);
    }
    if (paramParts.length) {
        parts.push(`Параметры: ${paramParts.join(", ")}`);
    }
    const selectionSummary = document.getElementById("analysis-selection-summary");
    const summaryText = parts.join(" · ");
    if (selectionSummary) {
        selectionSummary.textContent = summaryText;
    }
    if (trainBtn) {
        const buttonLabels = {
            pca: "Запустить PCA",
            kmeans: "Запустить k-means",
            hca: "Запустить HCA",
            plsda: "Запустить PLS-DA",
            svm: "Запустить SVM",
            random_forest: "Запустить Random Forest",
            decision_tree: "Запустить Decision Tree",
            pls: "Запустить PLS Regression",
            svr: "Запустить SVR",
            compare_classification: "Сравнить методы",
            compare_regression: "Сравнить методы",
            compare_clustering: "Сравнить методы",
        };
        trainBtn.disabled = false;
        trainBtn.textContent = selectedAnalysisTask === "compare" ? "Сравнить методы" : (buttonLabels[modelType] || "Запустить анализ");
    }
    analysisRunSummary.innerHTML = `
        <div class="analysis-run-summary__main">${escapeHtml(summaryText)}</div>
        ${simple ? `<div class="analysis-run-summary__hint">Параметры подобраны автоматически. Чтобы изменить их, включите расширенный режим.</div>` : ""}
    `;
    if (compareMethodsBtn) {
        compareMethodsBtn.style.display = "none";
        compareMethodsBtn.disabled = true;
    }
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function chooseRecommendedModel(modelType) {
    if (!modelType || ["warning", "preprocessing"].includes(modelType)) return;
    if (!modelTypeInput) return;
    selectedAnalysisMethod = modelType;
    selectedAnalysisTask = goalForModelType(modelType);
    modelTypeInput.value = modelType;
    if (analysisGoalInput) analysisGoalInput.value = selectedAnalysisTask;
    updateModelAdvancedSettings(modelType);
    updateTargetVisibility();
    renderAnalysisWorkflowGuide(importedDatasetSummary);
    renderAnalysisRunSummary();
}

async function previewDatasetImport() {
    const doneBusy = setBusy(datasetPreviewBtn, "Анализ...");
    try {
        setGlobalStatus("Файл загружается...", "Структура датасета анализируется на сервере.", 10);
        datasetPreviewResult.textContent = "Файл отправляется на сервер...";
        const response = await fetchWithTimeout("/analysis/dataset/preview", { method: "POST", body: datasetFilesFormData() });
        setGlobalStatus("Проверка структуры...", "Сервер прочитал файл, формируется preview.", 35);
        const data = await responseJsonOrError(response, "Не удалось применить предобработку.");
        if (!response.ok) throw new Error(humanError(data, "Не удалось выполнить предпросмотр."));
        datasetPreviewPayload = data;
        fillDatasetMappingControls(data);
        renderDatasetPreview(data);
        await validateDatasetImport(true);
        clearGlobalStatus("Предпросмотр готов.");
        showToast("Структура датасета проанализирована.", "success");
    } catch (error) {
        datasetPreviewResult.textContent = error.message;
        clearGlobalStatus("Ошибка.");
        showToast(error.message, "error");
    } finally {
        doneBusy();
    }
}

async function validateDatasetImport(fromPreview = false) {
    const doneBusy = setBusy(datasetValidateBtn, "Проверка...");
    try {
        setGlobalStatus("Проверка настроек...", "Данные приводятся к единому формату для проверки.", fromPreview ? 65 : 15);
        datasetValidationResult.textContent = "Проверяю настройки импорта...";
        const response = await fetchWithTimeout("/analysis/dataset/validate", { method: "POST", body: configFormData() });
        const data = await response.json();
        if (!response.ok) throw new Error(humanError(data, "Не удалось проверить датасет."));
        renderDatasetValidation(data);
        if (data.preview) {
            lastDatasetPreviewData = data.preview;
            renderDatasetPreviewPlot(datasetPreviewContainerId(), data.preview, { linesCount: "10", selectionStrategy: safeDatasetPreviewStrategy("balanced_by_class") });
        }
        if (!fromPreview) {
            renderText({ import_config: buildImportConfig(), validation: data });
            clearGlobalStatus("Проверка завершена.");
            showToast(data.status === "warning" ? "Датасет можно импортировать, но есть предупреждения." : "Настройки импорта проверены.", data.status === "warning" ? "warning" : "success");
        }
        return data;
    } catch (error) {
        datasetValidationResult.textContent = error.message;
        if (fromPreview) {
            renderImportCheckSummary();
        }
        if (!fromPreview) {
            clearGlobalStatus("Ошибка.");
            showToast(error.message, "error");
        }
        return null;
    } finally {
        doneBusy();
    }
}

async function importDataset() {
    const doneBusy = setBusy(datasetImportBtn, "Импорт...");
    try {
        setGlobalStatus("Данные приводятся к единому формату...", "Формируется матрица спектров и target-значения.", 15);
        datasetValidationResult.textContent = "Импортирую датасет...";
        const response = await fetchWithTimeout("/analysis/dataset/import", { method: "POST", body: configFormData() });
        setGlobalStatus("Формирование summary...", "Сервер проверяет полученный датасет.", 75);
        const data = await response.json();
        if (!response.ok) throw new Error(humanError(data, "Не удалось импортировать датасет."));
        if (data.status === "error") {
            renderDatasetValidation(data.validation);
            return;
        }
        importedDatasetId = data.dataset_id;
        importedDatasetSummary = data.summary;
        renderDatasetValidation(data.validation);
        renderDatasetReady(data.summary);
        if (data.preview) {
            lastDatasetPreviewData = data.preview;
            renderDatasetPreviewPlot(datasetPreviewContainerId(), data.preview, { linesCount: "10", selectionStrategy: safeDatasetPreviewStrategy("balanced_by_class") });
        } else {
            await renderImportedSpectraPreview(importedDatasetId);
        }
        renderText({ import_config: buildImportConfig(), import_result: data });
        markStepDone("import");
        clearGlobalStatus("Импорт завершён.");
        showToast("Датасет импортирован и готов к анализу.", "success");
    } catch (error) {
        datasetValidationResult.textContent = error.message;
        clearGlobalStatus("Ошибка.");
        showToast(error.message, "error");
    } finally {
        doneBusy();
    }
}

async function renderImportedSpectraPreviewLegacy(datasetId) {
    if (importResultCard) importResultCard.style.display = "";
    const limit = prePlotLimitInput?.value || "10";
    const strategy = prePlotStrategyInput?.value || "balanced_by_class";
    const response = await fetch(`/analysis/dataset/${datasetId}/spectra-preview?limit=${encodeURIComponent(limit)}&strategy=${encodeURIComponent(strategy)}&version=raw`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Не удалось построить предпросмотр спектров.");
    lastDatasetPreviewData = data;
    renderDatasetPreviewPlot("dataset-plot", data, { linesCount: limit, selectionStrategy: strategy });
}

async function downloadImportConfig() {
    if (importedDatasetId) {
        const response = await fetch(`/analysis/dataset/${importedDatasetId}/config`);
        const data = await response.json();
        if (!response.ok) {
            alert(data.detail || "Не удалось скачать config JSON");
            return;
        }
        const blob = new Blob([JSON.stringify(data.import_config || {}, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${importedDatasetId}_import_config.json`;
        link.click();
        URL.revokeObjectURL(url);
        return;
    }
    const blob = new Blob([JSON.stringify(buildImportConfig(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "import_config.json";
    link.click();
    URL.revokeObjectURL(url);
}

async function downloadDatasetMetadata() {
    if (!importedDatasetId) {
        showToast("Сначала импортируйте датасет.", "warning");
        return;
    }
    const version = processedDatasetReady ? "processed" : "raw";
    const response = await fetch(`/analysis/dataset/${importedDatasetId}/summary?version=${encodeURIComponent(version)}`);
    const data = await responseJsonOrError(response, "Не удалось скачать metadata JSON.");
    if (!response.ok) {
        showToast(humanError(data, "Не удалось скачать metadata JSON."), "error");
        return;
    }
    downloadTextFile(`${importedDatasetId}_metadata_${version}.json`, JSON.stringify(data.summary || data, null, 2), "application/json");
}

function exportImportedDataset(format, version = "raw") {
    if (!importedDatasetId) {
        showToast("Сначала импортируйте датасет.", "warning");
        return;
    }
    showToast(`Экспорт ${version === "processed" ? "предобработанного" : "исходного"} датасета начался.`, "success");
    showGlobalLoading("Подготовка файла к скачиванию...");
    window.location.href = `/analysis/dataset/${importedDatasetId}/export?format=${encodeURIComponent(format)}&version=${encodeURIComponent(version)}`;
    setTimeout(hideGlobalLoading, 1500);
}

async function downloadPreprocessingConfig() {
    if (!importedDatasetId || !processedDatasetReady) {
        alert("Сначала примените предобработку.");
        return;
    }
    const response = await fetch(`/analysis/dataset/${importedDatasetId}/preprocessing-config`);
    const data = await response.json();
    if (!response.ok) {
        alert(data.detail || "Не удалось скачать preprocessing JSON");
        return;
    }
    const blob = new Blob([JSON.stringify(data.preprocessing_config || {}, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${importedDatasetId}_preprocessing_config.json`;
    link.click();
    URL.revokeObjectURL(url);
}

function setActiveDataset(datasetId, summary = null) {
    activeDatasetId = datasetId;
    if (!activeDatasetBadge) return;
    if (!datasetId) {
        activeDatasetBadge.classList.remove("loaded-badge--ok");
        activeDatasetBadge.classList.add("loaded-badge--empty");
        activeDatasetBadge.textContent = "Импортированный датасет не выбран. Можно загрузить файл вручную.";
        if (trainFileInput) {
            trainFileInput.disabled = false;
        }
        if (manualTrainUploadRow) manualTrainUploadRow.style.display = "";
        return;
    }
    activeDatasetBadge.classList.remove("loaded-badge--empty");
    activeDatasetBadge.classList.add("loaded-badge--ok");
    const version = trainDatasetVersionInput?.value === "processed" ? "предобработанная версия" : "исходная версия";
    activeDatasetBadge.textContent = `Используется импортированный датасет: ${datasetId}, ${version}. Спектров: ${summary?.n_samples ?? "н/д"}, признаков: ${summary?.n_features ?? "н/д"}.`;
    if (trainFileInput) {
        trainFileInput.disabled = true;
    }
    if (manualTrainUploadRow) manualTrainUploadRow.style.display = "none";
}

function useImportedDatasetForTraining() {
    if (!importedDatasetId) {
        alert("Сначала импортируйте датасет.");
        return;
    }
    setActiveDataset(importedDatasetId, importedDatasetSummary);
    previewResult.textContent = "Для обучения будет использован импортированный датасет. Повторно загружать файл не нужно.";
    if (standardApplyPreprocessingInput?.checked && trainDatasetVersionInput) {
        trainDatasetVersionInput.value = processedDatasetReady ? "processed" : "raw";
        if (!processedDatasetReady) showToast("warning", "Стандартная предобработка ещё не применена. Для обучения выбран raw dataset.");
    }
}

function resetDatasetImport() {
    resetAnalysisWorkflow("manual");
}

function resetDatasetState() {
    datasetPreviewPayload = null;
    importedDatasetId = null;
    importedDatasetSummary = null;
    activeDatasetId = null;
    processedDatasetReady = false;
}

function resetPreprocessingState() {
    lastPreprocessingPreview = null;
    if (preprocessingCard) preprocessingCard.style.display = "none";
    if (preprocessingResult) preprocessingResult.textContent = "";
    if (preprocessingConfigView) preprocessingConfigView.textContent = "";
    if (datasetExportProcessedCsvBtn) datasetExportProcessedCsvBtn.disabled = true;
    if (datasetExportProcessedXlsxBtn) datasetExportProcessedXlsxBtn.disabled = true;
    if (datasetExportProcessedZipBtn) datasetExportProcessedZipBtn.disabled = true;
    if (datasetExportProcessedZipBtn) datasetExportProcessedZipBtn.disabled = true;
    if (preprocessingConfigDownloadBtn) preprocessingConfigDownloadBtn.disabled = true;
    if (standardApplyPreprocessingInput) standardApplyPreprocessingInput.checked = false;
    Plotly.purge("preprocessing-plot");
}

function resetModelState() {
    selectedModel = null;
    if (targetValuesInput) targetValuesInput.value = "";
    if (previewResult) previewResult.textContent = "Поддерживаются CSV, TXT, ESP и XLSX/XLS с числовыми парами x y или матрицей интенсивностей.";
    if (analysisRecommendations) analysisRecommendations.textContent = "После импорта здесь появятся рекомендуемые методы анализа.";
    clearStructuredBlocks();
    setLoadedModelBadge();
    setActiveDataset(null);
}

function resetCharts() {
    ["dataset-preview-plot", "dataset-plot", "preprocessing-plot", "validation-plot", "plot"].forEach((id) => Plotly.purge(id));
}

function resetMessages() {
    if (datasetPreviewResult) datasetPreviewResult.textContent = "Загрузите Excel, ZIP или набор TXT/CSV/ESP файлов.";
    if (datasetValidationResult) datasetValidationResult.textContent = "";
    if (datasetSummaryEl) datasetSummaryEl.textContent = "";
    if (datasetCheckSummary) datasetCheckSummary.textContent = "Проверка перед импортом появится после предпросмотра.";
    if (datasetReadySummary) datasetReadySummary.textContent = "";
    if (datasetPreviewStatus) datasetPreviewStatus.textContent = "";
    if (datasetAndorHint) {
        datasetAndorHint.style.display = "none";
        datasetAndorHint.textContent = "";
    }
    if (detectedMeasurementSummary) {
        detectedMeasurementSummary.style.display = "none";
        detectedMeasurementSummary.innerHTML = "";
    }
    if (measurementApplyDetectedBtn) measurementApplyDetectedBtn.disabled = true;
    if (textResult) textResult.textContent = "";
    keepDatasetUploadVisible();
    if (rawJsonWrap) rawJsonWrap.style.display = "none";
    if (resultsSection) resultsSection.style.display = "none";
}

function resetAnalysisWorkflow(reason = "new-file") {
    resetDatasetState();
    resetPreprocessingState();
    resetModelState();
    resetCharts();
    resetMessages();
    applyWorkflowMode();
    if (preprocessingCard) preprocessingCard.style.display = "none";
    if (trainDatasetVersionInput) {
        trainDatasetVersionInput.value = "raw";
        Array.from(trainDatasetVersionInput.options).forEach((option) => {
            if (option.value === "processed") option.disabled = true;
        });
    }
    if (datasetReadyCard) datasetReadyCard.style.display = "none";
    document.querySelectorAll(".workflow-tab--done").forEach((tab) => {
        tab.classList.remove("workflow-tab--done");
        const state = tab.querySelector(".step-state");
        if (state) state.textContent = "not-started";
    });
    if (reason === "new-file") {
        showToast("Выбран новый файл. Предыдущий датасет и результаты анализа сброшены.", "warning");
    }
}

function updateDatasetFileHint() {
    if (importedDatasetId) {
        const ok = confirm("Текущий импортированный датасет будет сброшен. Продолжить?");
        if (!ok) {
            datasetFilesInput.value = "";
            return;
        }
    }
    resetAnalysisWorkflow("new-file");
    keepDatasetUploadVisible();
    const files = Array.from(datasetFilesInput?.files || []);
    if (!datasetPreviewResult) {
        return;
    }
    if (files.length === 0) {
        datasetPreviewResult.textContent = "Загрузите Excel, ZIP или набор TXT/CSV/ESP файлов.";
        return;
    }
    datasetPreviewResult.textContent = `Выбрано файлов: ${files.length}\n${files.map((file) => `${file.name} (${Math.round(file.size / 1024)} KB)`).join("\n")}`;
}

async function previewFile() {
    const file = trainFileInput?.files?.[0];
    if (!file) {
        if (previewResult) previewResult.textContent = "Выберите файл для проверки.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/analysis/upload-preview", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
        if (previewResult) previewResult.textContent = data.detail || "Ошибка валидации файла.";
        return;
    }

    if (previewResult) previewResult.textContent = `Файл: ${data.filename}\nСпектров: ${data.sample_count}\nПризнаков: ${data.feature_count}\nФормат: ${data.source_format}`;
}

function numberValue(id, fallback) {
    const value = document.getElementById(id)?.value;
    return value === "" || value === undefined ? fallback : Number(value);
}

function collectProcessingBlocks() {
    const result = {};
    processingBlockIds.forEach((key) => {
        result[key.replaceAll("-", "_")] = !!document.getElementById(`block-${key}`)?.checked;
    });
    return result;
}

function setProcessingBlocks(blocks = {}) {
    processingBlockIds.forEach((key) => {
        const el = document.getElementById(`block-${key}`);
        if (!el) return;
        const normalized = key.replaceAll("-", "_");
        if (blocks[normalized] !== undefined) el.checked = !!blocks[normalized];
    });
    updateProcessingBlockVisibility();
}

function updateProcessingBlockVisibility() {
    const blocks = collectProcessingBlocks();
    const preset = preprocessingPresetInput?.value || "none";
    const showManualPanel = preset === "custom";
    const baselineWrap = document.getElementById("pre-baseline-method")?.closest("label");
    const smoothingWrap = document.getElementById("pre-smoothing-method")?.closest("label");
    const normalizationWrap = document.getElementById("pre-normalization-method")?.closest("label");
    if (baselineWrap) baselineWrap.style.display = (showManualPanel || blocks.baseline) ? "" : "none";
    if (smoothingWrap) smoothingWrap.style.display = (showManualPanel || blocks.smoothing) ? "" : "none";
    if (normalizationWrap) normalizationWrap.style.display = (showManualPanel || blocks.normalization) ? "" : "none";
    const peaks = document.getElementById("peaks-block-settings");
    if (peaks) peaks.style.display = blocks.peaks ? "" : "none";
    if (analysisGoalInput && blocks.model_compare) {
        analysisGoalInput.value = "compare";
        autoTuneModelParams();
    }
    updatePreprocessingParamVisibility();
    renderPreprocessingConfigView();
}

function buildPreprocessingConfig() {
    let blocks = collectProcessingBlocks();
    const preset = preprocessingPresetInput?.value || "none";
    if (preset !== "custom") {
        return { preset, blocks };
    }
    const baselineMethod = document.getElementById("pre-baseline-method")?.value || "off";
    const smoothingMethod = document.getElementById("pre-smoothing-method")?.value || "off";
    const normalizationMethod = document.getElementById("pre-normalization-method")?.value || "off";
    blocks = {
        ...blocks,
        baseline: baselineMethod !== "off",
        smoothing: smoothingMethod !== "off",
        normalization: normalizationMethod !== "off",
    };
    const config = {
        preset: "custom",
        blocks,
        baseline: { method: blocks.baseline ? baselineMethod : "off" },
        smoothing: { method: blocks.smoothing ? smoothingMethod : "off" },
        normalization: { method: blocks.normalization ? normalizationMethod : "off" },
        peaks: {
            enabled: !!blocks.peaks,
            method: document.getElementById("peak-method")?.value || "find_peaks",
            width: numberValue("peak-width", 1),
            prominence: numberValue("peak-prominence", 1),
        },
        pca: { enabled: !!blocks.pca },
        crop: {
            enabled: document.getElementById("pre-crop-enabled")?.value === "true",
            min_axis: numberValue("pre-crop-min", null),
            max_axis: numberValue("pre-crop-max", null),
        },
    };
    if (blocks.baseline && baselineMethod === "als") {
        config.baseline = {
            method: "als",
            lambda: numberValue("pre-als-lambda", 100000),
            p: numberValue("pre-als-p", 0.01),
            iterations: numberValue("pre-als-iterations", 10),
        };
    } else if (blocks.baseline && baselineMethod === "polynomial") {
        config.baseline = {
            method: "polynomial",
            degree: numberValue("pre-poly-degree", 3),
            iterations: numberValue("pre-poly-iterations", 10),
            asymmetry: numberValue("pre-poly-asymmetry", 0.01),
        };
    } else if (blocks.baseline && baselineMethod === "rolling_ball") {
        config.baseline = {
            method: "rolling_ball",
            radius: numberValue("pre-rolling-radius", 50),
            units: document.getElementById("pre-rolling-units")?.value || "x",
        };
    }
    if (blocks.smoothing && smoothingMethod === "savgol") {
        config.smoothing = { method: "savgol", window_length: numberValue("pre-sg-window", 15), polyorder: numberValue("pre-sg-polyorder", 3) };
    } else if (blocks.smoothing && smoothingMethod === "gaussian") {
        config.smoothing = { method: "gaussian", sigma: numberValue("pre-gaussian-sigma", 2) };
    } else if (blocks.smoothing && smoothingMethod === "moving_average") {
        config.smoothing = { method: "moving_average", window_size: numberValue("pre-ma-window", 5) };
    } else if (blocks.smoothing && smoothingMethod === "whittaker") {
        config.smoothing = { method: "whittaker", lambda: numberValue("pre-whittaker-lambda", 1000), order: numberValue("pre-whittaker-order", 2) };
    }
    return config;
}

function updatePreprocessingPresetCards() {
    const preset = preprocessingPresetInput?.value || "none";
    document.querySelectorAll(".preset-card[data-preset]").forEach((card) => {
        card.classList.toggle("preset-card--active", card.dataset.preset === preset);
    });
    const isCustom = preset === "custom";
    const details = document.getElementById("preprocessing-advanced");
    if (details) {
        details.style.display = "";
        if (isCustom) details.open = true;
    }
    updateProcessingBlockVisibility();
    updatePreprocessingParamVisibility();
    renderPreprocessingConfigView();
}

function updatePreprocessingParamVisibility() {
    const preset = preprocessingPresetInput?.value || "none";
    if (preset !== "custom") {
        document.querySelectorAll("[data-pre-param]").forEach((node) => {
            node.style.display = "none";
        });
        updatePreprocessingActiveSummary();
        return;
    }
    const baseline = document.getElementById("pre-baseline-method")?.value || "off";
    const smoothing = document.getElementById("pre-smoothing-method")?.value || "off";
    const crop = document.getElementById("pre-crop-enabled")?.value === "true";
    document.querySelectorAll("[data-pre-param]").forEach((node) => {
        const key = node.getAttribute("data-pre-param");
        const visible = (baseline !== "off" && key === baseline) || (smoothing !== "off" && key === smoothing) || (key === "crop" && crop);
        node.style.display = visible ? "flex" : "none";
    });
    updatePreprocessingActiveSummary();
    renderPreprocessingConfigView();
}

function updatePreprocessingActiveSummary() {
    const baseline = document.getElementById("pre-baseline-method")?.value || "off";
    const smoothing = document.getElementById("pre-smoothing-method")?.value || "off";
    const normalization = document.getElementById("pre-normalization-method")?.value || "off";
    const crop = document.getElementById("pre-crop-enabled")?.value === "true";
    const states = { baseline: baseline !== "off", smoothing: smoothing !== "off", normalization: normalization !== "off", crop };
    Object.entries(states).forEach(([key, enabled]) => {
        const badge = document.querySelector(`[data-pre-status="${key}"]`);
        if (!badge) return;
        badge.textContent = enabled ? "Включено" : "Выключено";
        badge.classList.toggle("pre-status-badge--on", enabled);
    });
    const steps = [];
    if (states.baseline) steps.push(methodLabel(baseline));
    if (states.smoothing) steps.push(methodLabel(smoothing));
    if (states.normalization) steps.push(methodLabel(normalization));
    if (states.crop) {
        const min = document.getElementById("pre-crop-min")?.value || "...";
        const max = document.getElementById("pre-crop-max")?.value || "...";
        steps.push(`Crop ${min}-${max}`);
    }
    const summary = document.getElementById("preprocessing-active-summary");
    if (summary) summary.textContent = `Активная обработка: ${steps.length ? steps.join(" → ") : "без предобработки"}`;
}

function resetPreprocessingToRaman() {
    if (preprocessingPresetInput) preprocessingPresetInput.value = "custom";
    const set = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.value = value;
    };
    set("pre-baseline-method", "als");
    set("pre-smoothing-method", "savgol");
    set("pre-normalization-method", "snv");
    set("pre-crop-enabled", "false");
    set("pre-als-lambda", "100000");
    set("pre-als-p", "0.01");
    set("pre-als-iterations", "10");
    set("pre-sg-window", "15");
    set("pre-sg-polyorder", "3");
    updatePreprocessingPresetCards();
    updatePreprocessingParamVisibility();
}

function renderPreprocessingPreview(payload) {
    lastPreprocessingPreview = payload;
    if (!preprocessingResult) return;
    const summary = payload.preprocessing_summary || {};
    const quality = payload.quality || {};
    const warnings = payload.warnings?.length ? `\nПредупреждения:\n- ${payload.warnings.join("\n- ")}` : "";
    preprocessingResult.textContent = [
        "Предпросмотр предобработки готов.",
        `Коррекция базовой линии: ${methodLabel(summary.baseline)}`,
        `Сглаживание: ${methodLabel(summary.smoothing)}`,
        `Нормализация: ${methodLabel(summary.normalization)}`,
        `признаков до: ${summary.raw_n_features || "н/д"}`,
        `признаков после обработки: ${summary.n_features || "н/д"}`,
        `NaN после обработки: ${quality.nan_count ?? 0}`,
        `доля отрицательных значений: ${formatPercent(quality.negative_fraction)}`,
        `диапазон оси: ${formatNumber(summary.axis_min)}-${formatNumber(summary.axis_max)}`,
        warnings,
    ].filter(Boolean).join("\n");
    renderPreprocessingConfigView();
    renderPreprocessingPlot(payload);
}

function methodLabel(method) {
    const labels = {
        off: "выключено",
        als: "ALS",
        savgol: "Савицкий — Голей",
        snv: "SNV",
        gaussian: "Gaussian",
        max: "по максимуму",
        polynomial: "Polynomial",
        rolling_ball: "Rolling Ball",
        moving_average: "Moving Average",
        whittaker: "Whittaker",
        vector: "векторная",
        area: "по площади",
    };
    return labels[method] || method || "выключено";
}

function formatPercent(value) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) return "н/д";
    return `${(Number(value) * 100).toFixed(1)}%`;
}

function viewNormalize(values) {
    const nums = values.map(Number);
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) return nums;
    return nums.map((v) => (v - min) / (max - min));
}

function classMeanTraces(payload, source) {
    const grouped = {};
    (payload.spectra || []).forEach((item) => {
        const key = item.target || "без target";
        grouped[key] = grouped[key] || [];
        grouped[key].push(item[source]);
    });
    return Object.entries(grouped).map(([cls, rows]) => {
        const n = rows[0]?.length || 0;
        const mean = Array.from({ length: n }, (_, idx) => rows.reduce((sum, row) => sum + Number(row[idx] || 0), 0) / rows.length);
        return { cls, mean };
    });
}

function renderPreprocessingPlot(payload = lastPreprocessingPreview) {
    if (!payload) return;
    const scale = prePlotScaleInput?.value || "autoscale";
    const showMode = prePlotVisibilityInput?.value || "before_after";
    const limitValue = prePlotLimitInput?.value || "5";
    const strategy = prePlotStrategyInput?.value || "balanced_by_class";
    const sampleIds = getSelectedSampleIds(payload, limitValue, strategy);
    const spectra = (payload.spectra || []).filter((item) => sampleIds.includes(item.sample_id));
    const traces = [];
    let layout = {
        title: "Предобработка: до/после",
        xaxis: { title: "Спектральная ось" },
        yaxis: { title: "Интенсивность" },
    };
    if (limitValue === "all" && (payload.spectra || []).length > 50) {
        showToast("warning", "Отображение всех спектров может замедлить браузер. Лучше использовать средние по классам.");
    }

    if (showMode === "class_means") {
        buildClassMeanTraces(payload, "raw").forEach((trace) => {
            traces.push({ ...trace, y: scale === "view_normalized" ? viewNormalize(trace.y) : trace.y, name: `${trace.name} raw` });
        });
        if (payload.axis_processed && payload.spectra?.some((item) => item.processed)) {
            buildClassMeanTraces(payload, "processed").forEach((trace) => {
                traces.push({ ...trace, y: scale === "view_normalized" ? viewNormalize(trace.y) : trace.y, name: `${trace.name} processed`, line: { dash: "dash" } });
            });
        }
        layout.title = "Предобработка: средние спектры по классам";
    } else if (showMode === "raw_only") {
        spectra.forEach((item) => {
            traces.push({ x: payload.axis_raw, y: scale === "view_normalized" ? viewNormalize(item.raw) : item.raw, type: "scatter", mode: "lines", name: `${item.sample_id} raw` });
        });
        layout.title = "Исходные спектры";
    } else if (showMode === "processed_only") {
        if (!payload.axis_processed || !payload.spectra?.some((item) => item.processed)) {
            showToast("warning", "Обработанная версия датасета ещё не создана.");
            return;
        }
        spectra.forEach((item) => {
            traces.push({ x: payload.axis_processed, y: scale === "view_normalized" ? viewNormalize(item.processed) : item.processed, type: "scatter", mode: "lines", name: `${item.sample_id} processed` });
        });
        layout.title = "Обработанные спектры";
    } else {
        if (!payload.axis_processed || !payload.spectra?.some((item) => item.processed)) {
            showToast("warning", "Обработанная версия датасета ещё не создана.");
            return;
        }
        spectra.forEach((item) => {
            traces.push({ x: payload.axis_raw, y: scale === "view_normalized" ? viewNormalize(item.raw) : item.raw, type: "scatter", mode: "lines", name: `${item.sample_id} raw`, xaxis: "x", yaxis: "y", opacity: 0.65 });
            traces.push({ x: payload.axis_processed, y: scale === "view_normalized" ? viewNormalize(item.processed) : item.processed, type: "scatter", mode: "lines", name: `${item.sample_id} processed`, xaxis: "x2", yaxis: "y2" });
        });
        layout = {
            title: "Предобработка: до/после",
            grid: { rows: 2, columns: 1, pattern: "independent" },
            xaxis: { title: "Исходные спектры" },
            yaxis: { title: "Интенсивность исходных" },
            xaxis2: { title: "Обработанные спектры" },
            yaxis2: { title: "Интенсивность обработанных" },
        };
        if (scale === "shared_y") {
            const all = traces.flatMap((trace) => trace.y || []);
            const min = Math.min(...all);
            const max = Math.max(...all);
            if (Number.isFinite(min) && Number.isFinite(max)) {
                layout.yaxis.range = [min, max];
                layout.yaxis2.range = [min, max];
            }
        }
    }
    if (!traces.length) {
        const el = document.getElementById("preprocessing-plot");
        if (el) {
            el.classList.add("is-empty");
            el.textContent = "Нет данных для построения графика.";
        }
        return;
    }
    const el = document.getElementById("preprocessing-plot");
    if (el) {
        el.classList.remove("is-empty");
        el.textContent = "";
    }
    newPlot("preprocessing-plot", traces, { ...layout, height: layout.grid ? 980 : 700 });
}

function renderPreprocessingConfigView() {
    if (!preprocessingConfigView) return;
    const config = buildPreprocessingConfig();
    updateCompactPreprocessingStatus(config);
    preprocessingConfigView.textContent = [
        "Конфигурация предобработки:",
        `1. Обрезка диапазона: ${config.crop?.enabled ? `${config.crop.min_axis}-${config.crop.max_axis}` : "выключено"}`,
        `2. Коррекция базовой линии: ${methodLabel(config.baseline?.method)}${config.baseline?.method === "als" ? `, lambda=${config.baseline.lambda}, p=${config.baseline.p}` : ""}`,
        `3. Сглаживание: ${methodLabel(config.smoothing?.method)}${config.smoothing?.method === "savgol" ? `, window=${config.smoothing.window_length}, polyorder=${config.smoothing.polyorder}` : ""}`,
        `4. Нормализация: ${methodLabel(config.normalization?.method)}`,
    ].join("\n");
}

function updateCompactPreprocessingStatus(config = buildPreprocessingConfig()) {
    const rowTitle = compactPreprocessingRow?.querySelector("strong");
    if (!rowTitle) return;
    const summary = config.preset && config.preset !== "custom"
        ? {
            none: "не применяется",
            raman_standard: "ALS + Savitzky-Golay + SNV",
            noisy: "ALS + Gaussian + Max",
            class_comparison: "ALS + Savitzky-Golay + SNV",
        }[config.preset] || config.preset
        : [methodLabel(config.baseline?.method), methodLabel(config.smoothing?.method), methodLabel(config.normalization?.method)]
            .filter((item) => item && item !== "выключено")
            .join(" + ") || "не применяется";
    rowTitle.textContent = `Предобработка: ${summary}`;
}

async function previewPreprocessing() {
    if (!importedDatasetId) {
        showToast("Сначала импортируйте датасет.", "warning");
        return;
    }
    const doneBusy = setBusy(document.getElementById("preprocessing-preview-btn"), "Предпросмотр...");
    try {
        setGlobalStatus("Предобработка выполняется...", "Сервер применяет настройки к нескольким спектрам.", 20);
        preprocessingResult.textContent = "Строю предпросмотр предобработки...";
        const limit = prePlotLimitInput?.value || "5";
        const strategy = prePlotStrategyInput?.value || "balanced";
        if (limit === "all") {
            showToast("Отображение всех спектров может замедлить работу браузера. Рекомендуется использовать средние спектры по классам.", "warning");
        }
        const response = await fetchWithTimeout(`/analysis/dataset/${importedDatasetId}/preprocess/preview?limit=${encodeURIComponent(limit)}&strategy=${encodeURIComponent(strategy)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(buildPreprocessingConfig()),
        });
        setGlobalStatus("График формируется...", "Готовим сравнение исходных и обработанных спектров.", 80);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(humanError(data, "Не удалось выполнить предобработку."));
        }
        renderPreprocessingPreview(data);
        markStepDone("visual");
        clearGlobalStatus("Предпросмотр готов.");
        showToast("Предпросмотр предобработки построен.", data.warnings?.length ? "warning" : "success");
    } catch (error) {
        preprocessingResult.textContent = error.message;
        clearGlobalStatus("Ошибка.");
        showToast(error.message, "error");
    } finally {
        doneBusy();
    }
}

async function applyPreprocessingToDataset() {
    if (!importedDatasetId) {
        showToast("Сначала импортируйте датасет.", "warning");
        return;
    }
    const doneBusy = setBusy(document.getElementById("preprocessing-apply-btn"), "Применение...");
    try {
        setGlobalStatus("Предобработка выполняется...", "Применяем pipeline ко всему датасету.", 20);
        preprocessingResult.textContent = "Применяю предобработку ко всему датасету...";
        const response = await fetchWithTimeout(`/analysis/dataset/${importedDatasetId}/preprocess/apply`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(buildPreprocessingConfig()),
        }, 180000);
        setGlobalStatus("Сохранение обработанной версии...", "Исходный датасет сохраняется без изменений.", 85);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(humanError(data, "Не удалось применить предобработку."));
        }
        processedDatasetReady = true;
        importedDatasetSummary = data.processed_summary;
        renderDatasetSummary(data.processed_summary);
        renderDatasetReady(data.processed_summary);
        preprocessingResult.textContent = [
            "Предобработка применена.",
            `Коррекция базовой линии: ${methodLabel(data.preprocessing_summary?.baseline)}`,
            `Сглаживание: ${methodLabel(data.preprocessing_summary?.smoothing)}`,
            `Нормализация: ${methodLabel(data.preprocessing_summary?.normalization)}`,
            `диапазон: ${formatNumber(data.preprocessing_summary?.axis_min)}-${formatNumber(data.preprocessing_summary?.axis_max)}`,
            `спектров: ${data.processed_summary?.n_samples || "н/д"}`,
            `признаков после обработки: ${data.preprocessing_summary?.n_features || "н/д"}`,
            data.warnings?.length ? `Предупреждения:\n- ${data.warnings.join("\n- ")}` : "",
        ].filter(Boolean).join("\n");
        if (datasetExportProcessedCsvBtn) datasetExportProcessedCsvBtn.disabled = false;
        if (datasetExportProcessedXlsxBtn) datasetExportProcessedXlsxBtn.disabled = false;
        if (datasetExportProcessedZipBtn) datasetExportProcessedZipBtn.disabled = false;
        if (preprocessingConfigDownloadBtn) preprocessingConfigDownloadBtn.disabled = false;
        if (trainDatasetVersionInput) {
            Array.from(trainDatasetVersionInput.options).forEach((option) => {
                if (option.value === "processed") option.disabled = false;
            });
            trainDatasetVersionInput.value = "processed";
        }
        if (datasetPreviewVersionInput) {
            Array.from(datasetPreviewVersionInput.options).forEach((option) => {
                if (option.value === "processed") option.disabled = false;
            });
            datasetPreviewVersionInput.value = "processed";
        }
        markStepDone("preprocess");
        clearGlobalStatus("Предобработка применена.");
        showToast("Обработанная версия датасета сохранена.", data.warnings?.length ? "warning" : "success");
    } catch (error) {
        preprocessingResult.textContent = error.message;
        clearGlobalStatus("Ошибка.");
        showToast(error.message, "error");
    } finally {
        doneBusy();
    }
}

async function trainModel() {
    if (activeDatasetId) {
        const useProcessed = trainDatasetVersionInput?.value === "processed";
        const payload = {
            dataset_id: activeDatasetId,
            use_processed: useProcessed,
            preprocessing_config: useProcessed ? buildPreprocessingConfig() : {},
            model_type: modelTypeInput.value,
            model_name: modelNameInput.value || "",
            analysis_goal: analysisGoalInput?.value || "",
            n_components: nComponentsInput.value ? Number(nComponentsInput.value) : null,
            params: getFrontendModelParams(modelTypeInput.value),
            model_params: getFrontendModelParams(modelTypeInput.value),
            validation: {
                enabled: doValidationInput.value === "true",
                test_size: Number(testSizeInput.value || "0.3"),
                random_state: Number(randomStateInput.value || "42"),
                stratified: true,
            },
        };
        const doneBusy = setBusy(document.getElementById("train-btn"), "Обучение...");
        try {
            setGlobalStatus("Модель обучается...", `Для обучения используется ${useProcessed ? "предобработанный" : "исходный"} датасет.`, 15);
            const response = await fetchWithTimeout("/analysis/model/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }, 180000);
            setGlobalStatus("Сохранение модели...", "Метаданные модели записываются вместе с preprocessing_config.", 85);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(humanError(data, "Не удалось обучить модель по импортированному датасету"));
            }
            renderText(data);
            renderStructuredResult(data, "train");
            if (!(data.task_type === "comparison" || data.run?.run_type === "comparison")) {
                renderResultPlots(data.plots || data.run?.plots || (data.plot ? [data.plot] : []));
            }
            await refreshModels();
            await refreshRuns();
            markStepDone("train");
            clearGlobalStatus("Модель обучена.");
            showToast("Модель обучена и сохранена.", "success");
        } catch (error) {
            clearGlobalStatus("Ошибка.");
            showToast(error.message, "error");
        } finally {
            doneBusy();
        }
        return;
    }

    const file = trainFileInput?.files?.[0];
    if (!file) {
        alert("Выберите обучающий файл или нажмите «Использовать для обучения модели» после импорта датасета.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_type", modelTypeInput.value);
    formData.append("model_name", modelNameInput.value || "");

    if (targetValuesInput.value.trim()) {
        formData.append("target_values", targetValuesInput.value.trim());
    }
    if (nComponentsInput.value) {
        formData.append("n_components", nComponentsInput.value);
    }

    formData.append("do_validation", doValidationInput.value);
    formData.append("test_size", testSizeInput.value || "0.3");
    formData.append("random_state", randomStateInput.value || "42");

    const doneBusy = setBusy(document.getElementById("train-btn"), "Обучение...");
    try {
        setGlobalStatus("Модель обучается...", "Файл отправлен на сервер.", 15);
        const response = await fetchWithTimeout("/analysis/train", { method: "POST", body: formData }, 180000);
        const data = await response.json();
        if (!response.ok) throw new Error(humanError(data, "Не удалось обучить модель"));
        renderText(data);
        renderStructuredResult(data, "train");
        if (!(data.task_type === "comparison" || data.run?.run_type === "comparison")) {
            renderResultPlots(data.plots || data.run?.plots || (data.plot ? [data.plot] : []));
        }
        await refreshModels();
        markStepDone("train");
        clearGlobalStatus("Модель обучена.");
        showToast("Модель обучена и сохранена.", "success");
    } catch (error) {
        clearGlobalStatus("Ошибка.");
        showToast(error.message, "error");
    } finally {
        doneBusy();
    }
}

function getFrontendModelParams(modelType) {
    const distribution = importedDatasetSummary?.class_distribution || {};
    const counts = Object.values(distribution).map(Number).filter(Number.isFinite);
    const imbalanced = counts.length >= 2 && Math.min(...counts) / Math.max(...counts) < 0.3;
    const classCount = Object.keys(distribution).length;
    const base = { random_state: Number(randomStateInput?.value || "42") };
    const kernel = document.getElementById("model-kernel")?.value || "rbf";
    const C = Number(document.getElementById("model-c")?.value || "1");
    const gamma = document.getElementById("model-gamma")?.value || "scale";
    const classWeight = document.getElementById("model-class-weight")?.value || (imbalanced ? "balanced" : null);
    const maxDepthValue = document.getElementById("model-max-depth")?.value;
    const maxDepth = maxDepthValue ? Number(maxDepthValue) : null;
    const nClusters = Number(document.getElementById("model-n-clusters")?.value || (classCount >= 2 ? classCount : 2));
    if (modelType === "svm") return { ...base, kernel, C, gamma, class_weight: classWeight };
    if (modelType === "random_forest") return { ...base, n_estimators: Number(document.getElementById("model-n-estimators")?.value || "200"), max_depth: maxDepth, class_weight: classWeight };
    if (modelType === "decision_tree") return { ...base, max_depth: maxDepth, class_weight: classWeight };
    if (modelType === "kmeans") return { ...base, n_clusters: nClusters };
    if (modelType === "hca") return { ...base, n_clusters: nClusters, linkage: document.getElementById("model-linkage")?.value || "ward" };
    if (modelType === "svr") return { ...base, kernel, C, epsilon: Number(document.getElementById("model-epsilon")?.value || "0.1"), gamma };
    return base;
}

function normalizeTargetsText(raw) {
    return raw
        .replace(/\r/g, "")
        .split(/[\n,;]+/)
        .map((v) => v.trim())
        .filter((v) => v.length > 0)
        .join(",");
}

async function loadTargetsFromFile() {
    const file = targetsFileInput.files[0];
    if (!file) {
        alert("Выберите файл с target-метками (.txt/.csv).");
        return;
    }

    const raw = await file.text();
    const normalized = normalizeTargetsText(raw);
    targetValuesInput.value = normalized;

    const count = normalized ? normalized.split(",").length : 0;
    targetsFileHint.textContent = `Загружено меток: ${count} (файл: ${file.name})`;
}

async function refreshModels() {
    const doneBusy = setBusy(document.getElementById("refresh-models-btn"), "Обновление...");
    try {
    const response = await fetch("/analysis/models");
    const data = await response.json();
    if (!response.ok) throw new Error(humanError(data, "Не удалось обновить список моделей."));
    modelsList.innerHTML = "";

    if (!Array.isArray(data) || data.length === 0) {
        modelsList.textContent = "Модели пока не сохранены.";
        return;
    }

    const rows = data.map((model) => {
        const metrics = model.metrics || {};
        const metricText = metrics.f1_macro !== undefined
            ? `F1=${Number(metrics.f1_macro).toFixed(3)}`
            : metrics.r2 !== undefined
                ? `R2=${Number(metrics.r2).toFixed(3)}`
                : metrics.explained_variance_total !== undefined
                    ? `EV=${Number(metrics.explained_variance_total).toFixed(3)}`
                    : metrics.silhouette_score !== undefined
                        ? `silhouette=${Number(metrics.silhouette_score).toFixed(3)}`
                        : metrics.inertia !== undefined
                            ? `inertia=${Number(metrics.inertia).toFixed(3)}`
                            : "";
        const key = `${model.model_type}:${model.model_id}`;
        return `
            <tr data-model-key="${escapeHtml(key)}">
                <td><strong>${escapeHtml(model.model_name || model.model_id)}</strong><br><span class="muted">${escapeHtml(model.model_id || "")}</span></td>
                <td>${escapeHtml(model.model_type || "н/д")}</td>
                <td>${escapeHtml(model.task_type || "analysis")}</td>
                <td>${escapeHtml(model.target_name || "нет")}</td>
                <td>${escapeHtml(metricText || "н/д")}</td>
                <td>${escapeHtml(model.created_at || "")}</td>
                <td class="actions">
                    <button type="button" class="btn btn-ghost" data-open-model="${escapeHtml(key)}">Открыть</button>
                    <button type="button" class="btn btn-secondary" data-load-model="${escapeHtml(key)}">Загрузить</button>
                    <button type="button" class="btn btn-ghost" data-download-model-zip="${escapeHtml(key)}">Скачать ZIP</button>
                    <button type="button" class="btn btn-danger" data-delete-model="${escapeHtml(key)}">Удалить</button>
                </td>
            </tr>
        `;
    }).join("");
    modelsList.innerHTML = `
        <table class="compact-table">
            <thead><tr><th>Название</th><th>Тип</th><th>Задача</th><th>Target</th><th>Метрика</th><th>Дата</th><th>Действия</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
    showToast("success", "Список моделей обновлён.");
    } catch (error) {
        showToast("error", error.message);
    } finally {
        doneBusy();
    }
}

async function refreshRuns() {
    if (!runsList) return;
    const doneBusy = setBusy(document.getElementById("refresh-runs-btn"), "Загружается история...");
    try {
        const response = await fetch("/analysis/runs");
        const data = await responseJsonOrError(response, "Не удалось загрузить историю.");
        if (!response.ok) throw new Error(humanError(data, "Не удалось загрузить историю."));
        const runs = data.runs || [];
        runsList.innerHTML = "";
        if (!runs.length) {
            runsList.textContent = "Запуски анализа пока не сохранены.";
            return;
        }
        const rows = runs.map((run) => {
            const metric = run.best_metric || run.metric || run.results?.best_metric || run.best_method || "н/д";
            return `
                <tr data-run-id="${escapeHtml(run.run_id)}">
                    <td>${escapeHtml(run.created_at || "")}<br><span class="muted">${escapeHtml(run.run_id)}</span></td>
                    <td>${escapeHtml(run.run_type || "analysis")}</td>
                    <td>${escapeHtml(run.best_method || run.method || "н/д")}</td>
                    <td>${escapeHtml(run.target_name || "нет")}</td>
                    <td>${escapeHtml(metric)}</td>
                    <td>${escapeHtml(run.status || "success")}</td>
                    <td class="actions">
                        <button type="button" class="btn btn-ghost" data-open-run="${escapeHtml(run.run_id)}">Открыть</button>
                        <button type="button" class="btn btn-ghost" data-run-export="${escapeHtml(run.run_id)}">Скачать ZIP</button>
                        <button type="button" class="btn btn-danger" data-delete-run="${escapeHtml(run.run_id)}">Удалить</button>
                    </td>
                </tr>
            `;
        }).join("");
        runsList.innerHTML = `
            <table class="compact-table">
                <thead><tr><th>Дата</th><th>Тип</th><th>Метод</th><th>Target</th><th>Метрика</th><th>Статус</th><th>Действия</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    } catch (error) {
        runsList.textContent = error.message;
        showToast("error", error.message);
    } finally {
        doneBusy();
    }
}

function getCheckedRunId() {
    return selectedRunId || document.querySelector("input[name='analysis-run']:checked")?.value || null;
}

function selectRun(runId) {
    selectedRunId = runId || null;
    document.querySelectorAll("[data-run-id]").forEach((row) => {
        row.classList.toggle("is-selected", row.dataset.runId === selectedRunId);
    });
}

async function openSelectedRun(runIdArg = null) {
    const runId = runIdArg || getCheckedRunId();
    if (!runId) {
        showToast("Выберите запуск анализа.", "warning");
        return;
    }
    showGlobalLoading("Открываю сохранённый запуск...");
    const response = await fetch(`/analysis/runs/${encodeURIComponent(runId)}`);
    const data = await response.json();
    hideGlobalLoading();
    if (!response.ok) {
        showToast(humanError(data, "Не удалось открыть запуск."), "error");
        return;
    }
    lastResultPayload = data.run;
    lastRenderedRunId = runId;
    renderText(data.run);
    clearStructuredBlocks();
    if (data.run?.run_type === "comparison") {
        renderComparisonResult({ run: data.run, result: data.run.results, run_id: data.run.run_id });
    } else {
        if (resultExplainer) resultExplainer.textContent = data.run?.summary || "Запуск анализа открыт.";
        renderMetricsTable(data.run?.results?.metrics ? [{ method: data.run.best_method || "model", metrics: data.run.results.metrics, status: data.run.status }] : []);
    }
    renderRunPlots(data.run?.plots || []);
    updateResultDownloads(runId);
    showResultTab("summary");
    scrollToResults();
}

async function deleteSelectedRun(runIdArg = null) {
    const runId = runIdArg || getCheckedRunId();
    if (!runId) {
        showToast("Выберите запуск анализа.", "warning");
        return;
    }
    if (!confirm(`Удалить запуск ${runId}?`)) return;
    const response = await fetch(`/analysis/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
        showToast(humanError(data, "Не удалось удалить запуск."), "error");
        return;
    }
    renderText(data);
    await refreshRuns();
    showToast("Запуск удалён.", "success");
}

function exportSelectedRun(runIdArg = null) {
    const runId = runIdArg || getCheckedRunId();
    if (!runId) {
        showToast("Выберите запуск анализа.", "warning");
        return;
    }
    window.location.href = `/analysis/runs/${encodeURIComponent(runId)}/export`;
}

function reportSelectedRun(runIdArg = null) {
    const runId = runIdArg || getCheckedRunId();
    if (!runId) {
        showToast("Выберите запуск анализа.", "warning");
        return;
    }
    showGlobalLoading("Формируется HTML-отчёт...");
    window.location.href = `/analysis/reports/from-run/${encodeURIComponent(runId)}`;
    setTimeout(hideGlobalLoading, 1500);
}

function getCheckedModel() {
    if (selectedSavedModel) return selectedSavedModel;
    const checked = document.querySelector("input[name='saved-model']:checked");
    if (!checked) {
        return null;
    }
    const [modelType, modelId] = checked.value.split(":");
    return { modelType, modelId };
}

function parseModelKey(key) {
    if (!key) return null;
    const [modelType, ...rest] = key.split(":");
    const modelId = rest.join(":");
    return modelType && modelId ? { modelType, modelId } : null;
}

function selectSavedModel(key) {
    selectedSavedModel = parseModelKey(key);
    document.querySelectorAll("[data-model-key]").forEach((row) => {
        row.classList.toggle("is-selected", row.dataset.modelKey === key);
    });
}

function downloadSelectedModel(kind = "model", modelArg = null) {
    const model = modelArg || getCheckedModel();
    if (!model) {
        showToast("warning", "Выберите модель из списка.");
        return;
    }
    const suffix = kind === "metadata" ? "metadata" : kind === "zip" ? "download-zip" : "download";
    showToast("success", "Скачивание выбранной модели началось.");
    showGlobalLoading("Подготовка модели к скачиванию...");
    window.location.href = `/analysis/models/${encodeURIComponent(model.modelId)}/${suffix}?model_type=${encodeURIComponent(model.modelType)}`;
    setTimeout(hideGlobalLoading, 1500);
}

async function loadSelectedModel(modelArg = null) {
    const model = modelArg || getCheckedModel();
    if (!model) {
        alert("Выберите модель из списка.");
        return;
    }

    const doneBusy = setBusy(document.getElementById("load-model-btn"), "Загрузка...");
    try {
        setGlobalStatus("Загрузка сохранённой модели...", "Читаем metadata модели.", 25);
        const response = await fetchWithTimeout(`/analysis/models/${model.modelType}/${model.modelId}/load`, { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(humanError(data, "Не удалось загрузить модель"));
        selectedModel = { modelType: model.modelType, modelId: model.modelId };
        setLoadedModelBadge();
        renderText({ message: "Модель загружена", selectedModel, metadata: data.model });
        renderModelMetadataCard(data.model);
        if (resultExplainer) resultExplainer.textContent = modelMetadataText(data.model);
        markStepDone("models");
        clearGlobalStatus("Модель загружена.");
        showToast("Сохранённая модель загружена.", "success");
    } catch (error) {
        clearGlobalStatus("Ошибка.");
        showToast(error.message, "error");
    } finally {
        doneBusy();
    }
}

async function openSavedModelMetadata(modelArg = null) {
    const model = modelArg || getCheckedModel();
    if (!model) {
        showToast("warning", "Выберите модель из списка.");
        return;
    }
    try {
        const response = await fetchWithTimeout(`/analysis/models/${model.modelType}/${model.modelId}/load`, { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(humanError(data, "Не удалось открыть metadata модели."));
        renderModelMetadataCard(data.model);
        showToast("Metadata модели открыта.", "success");
    } catch (error) {
        showToast("error", error.message);
    }
}

function modelMetadataText(meta = {}) {
    const preprocessing = meta.preprocessing || {};
    return [
        "Сведения о модели:",
        `название: ${meta.model_name || "н/д"}`,
        `тип: ${meta.model_type || "н/д"}`,
        `задача: ${meta.task_type || "н/д"}`,
        `run_id: ${meta.run_id || "н/д"}`,
        `создана: ${meta.created_at || "н/д"}`,
        `датасет: ${meta.dataset_id || meta.source_dataset_name || "н/д"}`,
        `target: ${meta.target_name || "н/д"} (${meta.target_type || "н/д"})`,
        `метрики: ${JSON.stringify(meta.metrics || {}, null, 2)}`,
        `использовались обработанные данные: ${meta.used_processed_data ? "да" : "нет"}`,
        `классы: ${(meta.classes || []).join(", ") || "н/д"}`,
        `диапазон оси: ${(meta.axis_range || []).join("–") || "н/д"}`,
        `ожидается признаков: ${meta.feature_count || meta.n_features || "н/д"}`,
        `предобработка: baseline=${methodLabel(preprocessing.baseline?.method)}, smoothing=${methodLabel(preprocessing.smoothing?.method)}, normalization=${methodLabel(preprocessing.normalization?.method)}`,
    ].join("\n");
}

function renderModelMetadataCard(meta = {}) {
    if (!modelMetadataCard) return;
    const preprocessing = meta.preprocessing_config || meta.preprocessing || {};
    const metrics = meta.metrics || {};
    const classes = Array.isArray(meta.classes) ? meta.classes.join(", ") : (meta.classes || "н/д");
    const tiles = [
        ["Тип", meta.model_type || "н/д"],
        ["Задача", meta.task_type || "н/д"],
        ["Target", meta.target_name || "н/д"],
        ["Классы", classes || "н/д"],
        ["Метрики", Object.keys(metrics).length ? Object.entries(metrics).map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(4) : v}`).join("; ") : "н/д"],
        ["Dataset", meta.dataset_id || meta.source_dataset_name || "н/д"],
        ["Предобработка", `baseline=${methodLabel(preprocessing.baseline?.method)}, smoothing=${methodLabel(preprocessing.smoothing?.method)}, normalization=${methodLabel(preprocessing.normalization?.method)}`],
        ["Ожидается признаков", meta.feature_count || meta.n_features || "н/д"],
        ["Диапазон оси", Array.isArray(meta.axis_range) ? meta.axis_range.join("-") : "н/д"],
    ];
    modelMetadataCard.innerHTML = `
        <h4>Metadata модели</h4>
        <div class="metadata-grid">
            ${tiles.map(([label, value]) => `
                <div class="metadata-tile">
                    <div class="metadata-label">${escapeHtml(label)}</div>
                    <div class="metadata-value">${escapeHtml(value)}</div>
                </div>
            `).join("")}
        </div>
        <details>
            <summary>Показать технические сведения</summary>
            <pre>${escapeHtml(JSON.stringify(meta, null, 2))}</pre>
        </details>
    `;
    modelMetadataCard.style.display = "block";
}

async function deleteSelectedModel(modelArg = null) {
    const model = modelArg || getCheckedModel();
    if (!model) {
        alert("Выберите модель для удаления.");
        return;
    }

    if (!confirm(`Удалить модель ${model.modelId} [${model.modelType}]?`)) {
        return;
    }

    const doneBusy = setBusy(document.getElementById("delete-model-btn"), "Удаление...");
    const response = await fetch(`/analysis/models/${model.modelType}/${model.modelId}`, { method: "DELETE" });
    const data = await response.json();

    if (!response.ok) {
        doneBusy();
        showToast("error", data.detail || "Не удалось удалить модель");
        return;
    }

    if (selectedModel && selectedModel.modelType === model.modelType && selectedModel.modelId === model.modelId) {
        selectedModel = null;
        setLoadedModelBadge();
    }

    renderText(data);
    await refreshModels();
    doneBusy();
    showToast("success", "Модель удалена.");
}

async function uploadOwnModel() {
    const modelFile = uploadModelFileInput.files[0];
    const metaFile = uploadMetaFileInput.files[0];
    if (!modelFile) {
        alert("Выберите .joblib файл модели.");
        return;
    }

    const formData = new FormData();
    formData.append("model_file", modelFile);
    if (metaFile) {
        formData.append("meta_file", metaFile);
    }
    formData.append("model_type", uploadModelTypeInput.value);
    formData.append("model_name", uploadModelNameInput.value || "");

    const response = await fetch("/analysis/models/upload", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
        alert(data.detail || "Не удалось загрузить модель");
        return;
    }

    renderText(data);
    resultExplainer.textContent = `Модель успешно загружена на сервер: ${data.saved_model.model_id}`;
    await refreshModels();
}

async function runInference() {
    if (!selectedModel) {
        alert("Сначала загрузите модель из списка.");
        return;
    }

    const files = Array.from(inferFileInput.files || []);
    if (files.length === 0) {
        alert("Выберите хотя бы один файл для инференса.");
        return;
    }

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    formData.append("model_type", selectedModel.modelType);
    formData.append("model_id", selectedModel.modelId);

    const doneBusy = setBusy(document.getElementById("infer-btn"), "Применение...");
    try {
        setGlobalStatus("Модель применяется...", "Файлы отправлены на сервер.", 20);
        const response = await fetchWithTimeout("/analysis/infer-batch", { method: "POST", body: formData }, 180000);
        const data = await response.json();
        if (!response.ok) throw new Error(humanError(data, "Ошибка применения модели"));
        renderText(data);
        renderStructuredResult(data, "infer");
        renderResultPlots(data.plots || (data.plot ? [data.plot] : []));
        markStepDone("infer");
        clearGlobalStatus("Применение модели завершено.");
        showToast("Модель применена к выбранным данным.", "success");
    } catch (error) {
        clearGlobalStatus("Ошибка.");
        showToast(error.message, "error");
    } finally {
        doneBusy();
    }
}

function bindClick(id, handler) {
    const element = document.getElementById(id);
    if (element) {
        element.addEventListener("click", handler);
    }
}

function downloadTextFile(filename, content, type = "text/plain") {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

function flattenComparisonRow(row) {
    const metrics = row?.metrics && typeof row.metrics === "object" ? row.metrics : {};
    const flat = {};
    Object.entries(row || {}).forEach(([key, value]) => {
        if (["metrics", "confusion_matrix", "params", "warnings"].includes(key)) return;
        flat[key] = value;
    });
    Object.assign(flat, metrics);
    return flat;
}

function toCsv(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return "";
    const flatRows = rows.map(flattenComparisonRow);
    const keys = Array.from(new Set(flatRows.flatMap((row) => Object.keys(row)))).sort();
    const escapeCsv = (value) => {
        const text = value === null || value === undefined ? "" : String(value);
        return /[",\n;]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    return [keys.join(","), ...flatRows.map((row) => keys.map((key) => escapeCsv(row[key])).join(","))].join("\n");
}

function downloadCurrentComparisonCsv() {
    if (!lastComparisonRows.length) {
        showToast("warning", "Нет таблицы сравнения для скачивания.");
        return;
    }
    downloadTextFile(`comparison_${lastRenderedRunId || Date.now()}.csv`, toCsv(lastComparisonRows), "text/csv");
}

function downloadCurrentResultJson() {
    if (!lastResultPayload) {
        showToast("warning", "Нет результата анализа для скачивания.");
        return;
    }
    downloadTextFile(`analysis_result_${lastRenderedRunId || Date.now()}.json`, JSON.stringify(lastResultPayload, null, 2), "application/json");
}

function downloadPredictionsCsv() {
    if (!lastPredictionRows.length) {
        showToast("warning", "Нет predictions для скачивания.");
        return;
    }
    downloadTextFile(`predictions_${Date.now()}.csv`, toCsv(lastPredictionRows), "text/csv");
}

function exportRunById(runId) {
    if (!runId) {
        showToast("warning", "run_id не найден.");
        return;
    }
    showGlobalLoading("Подготовка ZIP результата...");
    window.location.href = `/analysis/runs/${encodeURIComponent(runId)}/export`;
    setTimeout(hideGlobalLoading, 1500);
}

function downloadExampleDataset() {
    downloadTextFile(
        "example_correct_dataset.csv",
        "sample_id,target,source_sheet,source_file,469.34,470.38,471.42\nsample_1,group_a,Sheet1,example.xlsx,10.2,11.0,10.7\nsample_2,group_b,Sheet1,example.xlsx,15.1,14.8,15.4\n",
        "text/csv"
    );
}

function downloadExcelTemplate() {
    downloadTextFile(
        "excel_template_rows.csv",
        "ID,group,469.34,470.38,471.42\nsample_1,A,10.2,11.0,10.7\nsample_2,B,15.1,14.8,15.4\n",
        "text/csv"
    );
}

function collectAnalysisConfig() {
    return {
        blocks: collectProcessingBlocks(),
        preprocessing: buildPreprocessingConfig(),
        model: {
            goal: analysisGoalInput?.value || "explore",
            model_type: modelTypeInput?.value || "pca",
            model_name: modelNameInput?.value || "",
            dataset_version: trainDatasetVersionInput?.value || "raw",
            n_components: nComponentsInput?.value || "",
            validation: doValidationInput?.value || "true",
            test_size: testSizeInput?.value || "0.3",
            random_state: randomStateInput?.value || "42",
        },
    };
}

async function saveAnalysisConfig() {
    const config = collectAnalysisConfig();
    const response = await fetch("/analysis/configs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
    });
    const data = await response.json();
    if (!response.ok) {
        showToast(humanError(data, "Не удалось сохранить конфигурацию."), "error");
        return;
    }
    showToast(`Конфигурация сохранена: ${data.filename}`, "success");
    renderText(data);
}

function downloadAnalysisConfig() {
    const blob = new Blob([JSON.stringify(collectAnalysisConfig(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `analysis_config_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

async function loadAnalysisConfigFromFile() {
    const input = document.getElementById("load-analysis-config-file");
    const file = input?.files?.[0];
    if (!file) {
        showToast("Выберите JSON-файл конфигурации.", "warning");
        return;
    }
    try {
        const payload = JSON.parse(await file.text());
        applyAnalysisConfig(payload.config || payload);
        showToast("Конфигурация загружена.", "success");
    } catch (error) {
        showToast(`Не удалось прочитать конфигурацию: ${error.message}`, "error");
    }
}

function applyAnalysisConfig(config = {}) {
    if (config.blocks) setProcessingBlocks(config.blocks);
    const prep = config.preprocessing || {};
    if (prep.preset && preprocessingPresetInput) preprocessingPresetInput.value = prep.preset;
    if (prep.baseline?.method) document.getElementById("pre-baseline-method").value = prep.baseline.method;
    if (prep.smoothing?.method) document.getElementById("pre-smoothing-method").value = prep.smoothing.method;
    if (prep.normalization?.method) document.getElementById("pre-normalization-method").value = prep.normalization.method;
    if (config.model) {
        if (analysisGoalInput) analysisGoalInput.value = config.model.goal || analysisGoalInput.value;
        if (modelTypeInput) modelTypeInput.value = config.model.model_type || modelTypeInput.value;
        if (modelNameInput) modelNameInput.value = config.model.model_name || "";
        if (trainDatasetVersionInput) trainDatasetVersionInput.value = config.model.dataset_version || trainDatasetVersionInput.value;
        if (nComponentsInput) nComponentsInput.value = config.model.n_components || "";
        if (doValidationInput) doValidationInput.value = config.model.validation || doValidationInput.value;
        if (testSizeInput) testSizeInput.value = config.model.test_size || testSizeInput.value;
        if (randomStateInput) randomStateInput.value = config.model.random_state || randomStateInput.value;
    }
    updatePreprocessingPresetCards();
    updatePreprocessingParamVisibility();
    updateTargetVisibility();
}

async function renderImportedSpectraPreview(datasetId) {
    const limit = prePlotLimitInput?.value || "10";
    const strategy = safeDatasetPreviewStrategy(prePlotStrategyInput?.value || "balanced_by_class");
    const version = datasetPreviewVersionInput?.value || "raw";
    if (datasetPreviewStatus) datasetPreviewStatus.textContent = "Загрузка графика датасета...";
    try {
        const response = await fetch(`/analysis/dataset/${datasetId}/spectra-preview?limit=${encodeURIComponent(limit)}&strategy=${encodeURIComponent(strategy)}&version=${encodeURIComponent(version)}`);
        const data = await responseJsonOrError(response, "Не удалось построить предпросмотр спектров.");
        if (!response.ok) throw new Error(humanError(data, "Не удалось построить предпросмотр спектров."));
        lastDatasetPreviewData = data;
        renderDatasetPreviewPlot(datasetPreviewContainerId(), data, { linesCount: limit, selectionStrategy: strategy });
        if (datasetPreviewStatus) datasetPreviewStatus.textContent = `График обновлён (${version}).`;
    } catch (error) {
        if (datasetPreviewStatus) datasetPreviewStatus.textContent = error.message;
        showToast(error.message, "error");
        throw error;
    }
}

function initAnalysisPage() {
    initTheme();
    bindClick("preview-btn", previewFile);
    bindClick("train-btn", trainModel);
    bindClick("refresh-models-btn", refreshModels);
    bindClick("load-model-btn", loadSelectedModel);
    bindClick("download-model-btn", () => downloadSelectedModel("model"));
    bindClick("download-model-meta-btn", () => downloadSelectedModel("metadata"));
    bindClick("download-model-zip-btn", () => downloadSelectedModel("zip"));
    bindClick("delete-model-btn", deleteSelectedModel);
    bindClick("upload-model-btn", uploadOwnModel);
    bindClick("infer-btn", runInference);
    bindClick("load-targets-btn", loadTargetsFromFile);
    bindClick("dataset-preview-btn", previewDatasetImport);
    bindClick("dataset-validate-btn", () => validateDatasetImport(false));
    bindClick("dataset-import-btn", importDataset);
    bindClick("dataset-metadata-download-btn", downloadDatasetMetadata);
    bindClick("dataset-config-download-btn", downloadImportConfig);
    bindClick("dataset-use-btn", useImportedDatasetForTraining);
    bindClick("dataset-show-plot-btn", () => {
        if (!importedDatasetId) {
            showToast("warning", "Сначала загрузите или импортируйте датасет.");
            return;
        }
        renderImportedSpectraPreview(importedDatasetId);
    });
    bindClick("compact-preprocessing-plot-btn", () => {
        if (!importedDatasetId) {
            showToast("warning", "Сначала загрузите правильный датасет.");
            return;
        }
        renderImportedSpectraPreview(importedDatasetId);
    });
    bindClick("show-preprocessing-settings-btn", revealPreprocessingSettings);
    bindClick("dataset-export-csv-btn", () => exportImportedDataset("csv", "raw"));
    bindClick("dataset-export-xlsx-btn", () => exportImportedDataset("xlsx", "raw"));
    bindClick("dataset-export-zip-btn", () => exportImportedDataset("zip", "raw"));
    bindClick("dataset-export-processed-csv-btn", () => exportImportedDataset("csv", "processed"));
    bindClick("dataset-export-processed-xlsx-btn", () => exportImportedDataset("xlsx", "processed"));
    bindClick("dataset-export-processed-zip-btn", () => exportImportedDataset("zip", "processed"));
    bindClick("preprocessing-config-download-btn", downloadPreprocessingConfig);
    bindClick("preprocessing-reset-raman-btn", resetPreprocessingToRaman);
    bindClick("measurement-apply-detected-btn", () => {
        applyDetectedMeasurementValues(datasetPreviewPayload?.zip ? { metadata: { ...(datasetPreviewPayload.zip.metadata || {}), sample_metadata_preview: datasetPreviewPayload.zip.sample_metadata_preview || [] } } : importedDatasetSummary);
        showToast("Обнаруженные одиночные параметры подставлены в поля.", "success");
    });
    bindClick("measurement-manual-toggle-btn", () => {
        document.getElementById("meta-slit-width")?.focus();
    });
    bindClick("dataset-reset-btn", resetDatasetImport);
    bindClick("dataset-sheets-select-all", () => setAllSheetCheckboxes(true));
    bindClick("dataset-sheets-clear", () => setAllSheetCheckboxes(false));
    bindClick("preprocessing-preview-btn", previewPreprocessing);
    bindClick("preprocessing-apply-btn", applyPreprocessingToDataset);
    bindClick("help-open-btn", () => helpDialog?.showModal());
    bindClick("help-close-btn", () => helpDialog?.close());
    bindClick("theme-toggle-btn", () => applyTheme(currentTheme() === "dark" ? "light" : "dark"));
    bindClick("download-example-csv-btn", downloadExampleDataset);
    bindClick("download-template-xlsx-btn", downloadExcelTemplate);
    bindClick("save-analysis-config-btn", saveAnalysisConfig);
    bindClick("download-analysis-config-btn", downloadAnalysisConfig);
    bindClick("load-analysis-config-btn", loadAnalysisConfigFromFile);
    bindClick("standard-dataset-import-btn", importStandardDataset);
    bindClick("refresh-runs-btn", refreshRuns);
    bindClick("open-run-btn", openSelectedRun);
    bindClick("export-run-btn", exportSelectedRun);
    bindClick("report-run-btn", reportSelectedRun);
    bindClick("delete-run-btn", deleteSelectedRun);
    bindClick("compare-methods-btn", () => {
        const { categoricalTarget, numericTarget } = targetState(importedDatasetSummary);
        if (!categoricalTarget && !numericTarget) {
            showToast("Для сравнения методов нужен target.", "warning");
            return;
        }
        if (analysisGoalInput) analysisGoalInput.value = "compare";
        if (modelTypeInput) modelTypeInput.value = categoricalTarget ? "compare_classification" : "compare_regression";
        selectedAnalysisTask = "compare";
        selectedAnalysisMethod = modelTypeInput?.value || (categoricalTarget ? "compare_classification" : "compare_regression");
        updateModelAdvancedSettings(modelTypeInput?.value || "pca");
        trainModel();
    });

    document.addEventListener("click", (event) => {
        const target = event.target;
        const fullscreenId = target?.getAttribute?.("data-chart-fullscreen");
        const pngId = target?.getAttribute?.("data-chart-png");
        const recommendedModel = target?.closest?.("[data-recommend-model]")?.getAttribute?.("data-recommend-model");
        const goalCard = target?.closest?.("[data-analysis-goal-card]")?.getAttribute?.("data-analysis-goal-card");
        const bestModel = target?.getAttribute?.("data-load-best-model");
        const runExport = target?.getAttribute?.("data-run-export");
        const stepButton = target?.closest?.("[data-target-section]");
        const targetSection = stepButton?.getAttribute?.("data-target-section");
        const resultTab = target?.getAttribute?.("data-result-tab");
        const rowRunId = target?.closest?.("[data-run-id]")?.dataset?.runId;
        const rowModelKey = target?.closest?.("[data-model-key]")?.dataset?.modelKey;
        const openRun = target?.getAttribute?.("data-open-run");
        const deleteRun = target?.getAttribute?.("data-delete-run");
        const openModel = target?.getAttribute?.("data-open-model");
        const loadModel = target?.getAttribute?.("data-load-model");
        const deleteModel = target?.getAttribute?.("data-delete-model");
        const downloadModelZip = target?.getAttribute?.("data-download-model-zip");
        if (fullscreenId) openFullscreenPlot(fullscreenId);
        if (pngId) downloadPlotPng(pngId);
        if (recommendedModel) chooseRecommendedModel(recommendedModel);
        if (goalCard && analysisGoalInput) {
            const task = analysisTaskDefinitions(importedDatasetSummary).find((item) => item.goal === goalCard);
            if (!task?.enabled) {
                showToast(task?.disabledReason || "Эта задача недоступна для текущего датасета.", "warning");
                return;
            }
            selectedAnalysisTask = goalCard;
            selectedAnalysisMethod = defaultMethodForTask(goalCard, importedDatasetSummary);
            syncAnalysisSelectionToInputs();
            updateModelAdvancedSettings(selectedAnalysisMethod || "pca");
            updateTargetVisibility();
            renderAnalysisWorkflowGuide(importedDatasetSummary);
            renderAnalysisRunSummary();
        }
        if (resultTab) showResultTab(resultTab);
        if (rowRunId) selectRun(rowRunId);
        if (rowModelKey) selectSavedModel(rowModelKey);
        if (target?.hasAttribute?.("data-download-comparison-csv")) downloadCurrentComparisonCsv();
        if (target?.hasAttribute?.("data-download-predictions-csv")) downloadPredictionsCsv();
        if (target?.hasAttribute?.("data-download-result-json")) downloadCurrentResultJson();
        if (runExport) exportRunById(runExport);
        if (openRun) openSelectedRun(openRun);
        if (deleteRun) deleteSelectedRun(deleteRun);
        if (openModel) openSavedModelMetadata(parseModelKey(openModel));
        if (loadModel) loadSelectedModel(parseModelKey(loadModel));
        if (downloadModelZip) downloadSelectedModel("zip", parseModelKey(downloadModelZip));
        if (deleteModel) deleteSelectedModel(parseModelKey(deleteModel));
        if (targetSection) {
            document.querySelectorAll(".workflow-tab").forEach((tab) => tab.classList.remove("workflow-tab--active"));
            stepButton.classList.add("workflow-tab--active");
            document.getElementById(targetSection)?.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        if (bestModel) {
            const [modelType, modelId] = bestModel.split(":");
            selectedModel = { modelType, modelId };
            setLoadedModelBadge();
            showToast("Лучшая модель выбрана для применения.", "success");
        }
    });

    document.addEventListener("change", (event) => {
        const taskSelect = event.target?.closest?.("#analysis-task-select");
        if (!taskSelect) return;
        const goal = taskSelect.value;
        const task = analysisTaskDefinitions(importedDatasetSummary).find((item) => item.goal === goal);
        if (!task?.enabled) {
            showToast(task?.disabledReason || "Эта задача недоступна для текущего датасета.", "warning");
            renderAnalysisWorkflowGuide(importedDatasetSummary);
            return;
        }
        selectedAnalysisTask = goal;
        selectedAnalysisMethod = defaultMethodForTask(goal, importedDatasetSummary);
        syncAnalysisSelectionToInputs();
        updateModelAdvancedSettings(selectedAnalysisMethod || "pca");
        updateTargetVisibility();
        renderAnalysisWorkflowGuide(importedDatasetSummary);
        renderAnalysisRunSummary();
    });

    document.querySelectorAll(".layout-card[data-layout]").forEach((card) => {
        card.addEventListener("click", () => {
            datasetLayoutInput.value = card.dataset.layout;
            updateDatasetLayoutUI();
        });
    });

    if (datasetFilesInput) {
        datasetFilesInput.addEventListener("change", updateDatasetFileHint);
    }
    if (workflowModeInput) {
        workflowModeInput.addEventListener("change", applyWorkflowMode);
        applyWorkflowMode();
    }
    if (datasetLayoutInput) {
        datasetLayoutInput.addEventListener("change", updateDatasetLayoutUI);
    }
    if (datasetTargetSourceInput) {
        datasetTargetSourceInput.addEventListener("change", updateDatasetLayoutUI);
    }
    if (datasetSheetModeInput) {
        datasetSheetModeInput.addEventListener("change", () => {
            if (datasetSheetModeInput.value === "sheet_as_class") {
                datasetTargetSourceInput.value = "sheet_name";
            }
            renderImportCheckSummary();
        });
    }
    if (datasetSheetInput) {
        datasetSheetInput.addEventListener("change", () => {
            const sheet = firstExcelSheetPreview();
            const columns = sheet?.columns || [];
            optionList(datasetIdColumnInput, columns, sheet?.id_candidates?.[0] || "");
            optionList(datasetTargetColumnInput, columns, sheet?.numeric_target_candidates?.[0] || sheet?.target_candidates?.[0] || "");
            optionList(datasetAxisColumnInput, columns, sheet?.axis_candidates?.[0] || columns[0] || "");
            renderImportCheckSummary();
        });
    }
    [datasetIdColumnInput, datasetTargetColumnInput, datasetAxisColumnInput, datasetGridModeInput].forEach((input) => {
        if (input) input.addEventListener("change", renderImportCheckSummary);
    });
    if (datasetTargetRegexInput) datasetTargetRegexInput.addEventListener("input", renderImportCheckSummary);
    document.querySelectorAll(".preset-card[data-preset]").forEach((card) => {
        card.addEventListener("click", () => {
            preprocessingPresetInput.value = card.dataset.preset;
            updatePreprocessingPresetCards();
        });
    });
    if (preprocessingPresetInput) {
        preprocessingPresetInput.addEventListener("change", updatePreprocessingPresetCards);
    }
    ["pre-baseline-method", "pre-smoothing-method", "pre-normalization-method", "pre-crop-enabled", "pre-crop-min", "pre-crop-max"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("change", updatePreprocessingParamVisibility);
    });
    processingBlockIds.forEach((key) => {
        const el = document.getElementById(`block-${key}`);
        if (el) el.addEventListener("change", updateProcessingBlockVisibility);
    });
    [prePlotModeInput, prePlotScaleInput, prePlotLimitInput, prePlotStrategyInput, prePlotVisibilityInput].forEach((input) => {
        if (input) input.addEventListener("change", rerenderActivePreviewPlot);
    });
    if (datasetPreviewVersionInput) {
        datasetPreviewVersionInput.addEventListener("change", () => {
            if (importedDatasetId) renderImportedSpectraPreview(importedDatasetId).catch(() => {});
        });
    }
    if (modelConfigModeInput) modelConfigModeInput.addEventListener("change", updateModelModeUI);
    if (analysisGoalInput) analysisGoalInput.addEventListener("change", autoTuneModelParams);
    if (modelTypeInput) {
        modelTypeInput.addEventListener("change", () => {
            if (analysisGoalInput) analysisGoalInput.value = goalForModelType(modelTypeInput.value);
            updateTargetVisibility();
            updateModelAdvancedSettings(modelTypeInput.value);
            renderAnalysisWorkflowGuide(importedDatasetSummary);
            renderAnalysisRunSummary();
        });
        updateTargetVisibility();
    }
    [
        trainDatasetVersionInput,
        nComponentsInput,
        randomStateInput,
        document.getElementById("model-n-clusters"),
        document.getElementById("model-linkage"),
        document.getElementById("model-kernel"),
        document.getElementById("model-c"),
        document.getElementById("model-gamma"),
        document.getElementById("model-epsilon"),
        document.getElementById("model-n-estimators"),
        document.getElementById("model-max-depth"),
    ].forEach((input) => {
        if (input) input.addEventListener("change", renderAnalysisRunSummary);
    });
    if (inferFileInput) {
        inferFileInput.addEventListener("change", updateInferFilesHint);
        updateInferFilesHint();
    }
    if (loadedModelBadge) {
        setLoadedModelBadge();
    }
    updateDatasetLayoutUI();
    updatePreprocessingPresetCards();
    updatePreprocessingParamVisibility();
    updateModelModeUI();
    updateModelAdvancedSettings(modelTypeInput?.value || "pca");
    setActiveDataset(null);
    refreshModels().catch((error) => showToast("warning", `Не удалось загрузить список моделей: ${error.message}`));
    refreshRuns().catch((error) => showToast("warning", `Не удалось загрузить историю запусков: ${error.message}`));
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAnalysisPage);
} else {
    initAnalysisPage();
}
