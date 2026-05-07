let selectedModel = null;

const trainFileInput = document.getElementById("train-file");
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
const resultExplainer = document.getElementById("result-explainer");
const validationSummary = document.getElementById("validation-summary");
const validationPlotEl = document.getElementById("validation-plot");
const modelsList = document.getElementById("models-list");
const targetsWrap = document.getElementById("targets-wrap");
const loadedModelBadge = document.getElementById("loaded-model-badge");
const inferFilesHint = document.getElementById("infer-files-hint");

function updateTargetVisibility() {
    const mt = modelTypeInput.value;
    const supervised = mt === "pls" || mt === "plsda";
    targetsWrap.style.display = supervised ? "flex" : "none";
    const importWrap = document.getElementById("targets-import-wrap");
    if (importWrap) {
        importWrap.style.display = supervised ? "flex" : "none";
    }
}

function setLoadedModelBadge() {
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
    const files = Array.from(inferFileInput.files || []);
    if (files.length === 0) {
        inferFilesHint.textContent = "Файлы для применения модели не выбраны.";
        return;
    }
    inferFilesHint.textContent = `Выбрано файлов: ${files.length}. ${files.map((f) => f.name).join(", ")}`;
}

function renderText(obj) {
    textResult.textContent = JSON.stringify(obj, null, 2);
}

function clearStructuredBlocks() {
    resultExplainer.textContent = "";
    validationSummary.innerHTML = "";
    Plotly.purge("validation-plot");
    validationPlotEl.style.display = "none";
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
        Plotly.newPlot(
            "validation-plot",
            [
                {
                    type: "heatmap",
                    z: validation.confusion_matrix,
                    x: validation.classes,
                    y: validation.classes,
                    colorscale: "Blues",
                    showscale: true,
                },
            ],
            {
                title: "Confusion Matrix (test)",
                xaxis: { title: "Predicted class" },
                yaxis: { title: "True class" },
                template: "plotly_white",
            },
            { responsive: true }
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

function renderStructuredResult(payload, mode) {
    clearStructuredBlocks();
    if (mode === "train") {
        const saved = payload.saved_model || {};
        resultExplainer.textContent = `Модель сохранена: ${saved.model_id || "unknown"} (${saved.model_type || "n/a"}).`;
        renderValidationBlock(payload.validation);
        return;
    }

    if (mode === "infer") {
        const meta = payload.model_metadata || {};
        const filesCount = Array.isArray(payload.batch_results) ? payload.batch_results.length : 1;
        resultExplainer.textContent = `Инференс выполнен.\nМодель: ${meta.model_id || "n/a"} (${meta.model_type || "n/a"})\nОбработано файлов: ${filesCount}`;
        renderInferenceExplanation(payload);
    }
}

function renderPlot(plotPayload) {
    if (!plotPayload || !plotPayload.data) {
        Plotly.purge("plot");
        return;
    }
    Plotly.newPlot("plot", plotPayload.data, plotPayload.layout || {}, { responsive: true });
}

async function previewFile() {
    const file = trainFileInput.files[0];
    if (!file) {
        previewResult.textContent = "Выберите файл для проверки.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/analysis/upload-preview", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
        previewResult.textContent = data.detail || "Ошибка валидации файла.";
        return;
    }

    previewResult.textContent = `Файл: ${data.filename}\nСпектров: ${data.sample_count}\nПризнаков: ${data.feature_count}\nФормат: ${data.source_format}`;
}

async function trainModel() {
    const file = trainFileInput.files[0];
    if (!file) {
        alert("Сначала выберите обучающий файл.");
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

    const response = await fetch("/analysis/train", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
        alert(data.detail || "Не удалось обучить модель");
        return;
    }

    renderText(data);
    renderStructuredResult(data, "train");
    renderPlot(data.plot);
    await refreshModels();
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
    const response = await fetch("/analysis/models");
    const data = await response.json();
    modelsList.innerHTML = "";

    if (!Array.isArray(data) || data.length === 0) {
        modelsList.textContent = "Модели пока не сохранены.";
        return;
    }

    data.forEach((model) => {
        const row = document.createElement("label");
        row.className = "model-item";
        row.innerHTML = `
            <input type="radio" name="saved-model" value="${model.model_type}:${model.model_id}">
            <span><strong>${model.model_name || model.model_id}</strong> [${model.model_type}]</span>
        `;
        modelsList.appendChild(row);
    });
}

function getCheckedModel() {
    const checked = document.querySelector("input[name='saved-model']:checked");
    if (!checked) {
        return null;
    }
    const [modelType, modelId] = checked.value.split(":");
    return { modelType, modelId };
}

async function loadSelectedModel() {
    const model = getCheckedModel();
    if (!model) {
        alert("Выберите модель из списка.");
        return;
    }

    const response = await fetch(`/analysis/models/${model.modelType}/${model.modelId}/load`, { method: "POST" });
    const data = await response.json();

    if (!response.ok) {
        alert(data.detail || "Не удалось загрузить модель");
        return;
    }

    selectedModel = { modelType: model.modelType, modelId: model.modelId };
    setLoadedModelBadge();
    renderText({ message: "Модель загружена", selectedModel, metadata: data.model });
}

async function deleteSelectedModel() {
    const model = getCheckedModel();
    if (!model) {
        alert("Выберите модель для удаления.");
        return;
    }

    if (!confirm(`Удалить модель ${model.modelId} [${model.modelType}]?`)) {
        return;
    }

    const response = await fetch(`/analysis/models/${model.modelType}/${model.modelId}`, { method: "DELETE" });
    const data = await response.json();

    if (!response.ok) {
        alert(data.detail || "Не удалось удалить модель");
        return;
    }

    if (selectedModel && selectedModel.modelType === model.modelType && selectedModel.modelId === model.modelId) {
        selectedModel = null;
        setLoadedModelBadge();
    }

    renderText(data);
    await refreshModels();
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

    const response = await fetch("/analysis/infer-batch", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
        alert(data.detail || "Ошибка инференса");
        return;
    }

    renderText(data);
    renderStructuredResult(data, "infer");
    renderPlot(data.plot);
}

document.getElementById("preview-btn").addEventListener("click", previewFile);
document.getElementById("train-btn").addEventListener("click", trainModel);
document.getElementById("refresh-models-btn").addEventListener("click", refreshModels);
document.getElementById("load-model-btn").addEventListener("click", loadSelectedModel);
document.getElementById("delete-model-btn").addEventListener("click", deleteSelectedModel);
document.getElementById("upload-model-btn").addEventListener("click", uploadOwnModel);
document.getElementById("infer-btn").addEventListener("click", runInference);
document.getElementById("load-targets-btn").addEventListener("click", loadTargetsFromFile);
modelTypeInput.addEventListener("change", updateTargetVisibility);
inferFileInput.addEventListener("change", updateInferFilesHint);

updateTargetVisibility();
updateInferFilesHint();
setLoadedModelBadge();
refreshModels();
