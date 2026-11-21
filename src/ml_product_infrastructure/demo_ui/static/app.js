"use strict";

const ui = Object.fromEntries([
  "file-input", "drop-zone", "preview", "upload-placeholder", "replace-hint",
  "file-name", "image-dimensions", "prediction-panel", "request-status",
  "empty-result", "loading-result", "result", "error", "error-message",
  "retry-prediction", "winner-name", "winner-confidence", "probabilities",
  "model-name", "model-version", "examples", "examples-error", "retry-examples",
  "classes-description",
].map((id) => [id, document.getElementById(id)]));

const percent = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });
let classes = [];
let lastFile = null;
let previewUrl = null;
let activeRequest = null;
let requestId = 0;
let dragDepth = 0;

function setState(state, message = "") {
  ui["empty-result"].hidden = state !== "empty";
  ui["loading-result"].hidden = state !== "loading";
  ui.result.hidden = state !== "ready";
  ui.error.hidden = state !== "error";
  ui["prediction-panel"].setAttribute("aria-busy", String(state === "loading"));
  ui["request-status"].classList.toggle("ready", state === "ready");
  const status = {
    empty: "Ожидание изображения",
    loading: "Вычисление…",
    ready: "Предсказание готово",
    error: "Ошибка запроса",
  };
  ui["request-status"].textContent = status[state];
  if (state === "error") ui["error-message"].textContent = message;
  ui["retry-prediction"].hidden = !lastFile;
}

function clearSelection() {
  ui.examples.querySelectorAll(".selected").forEach((button) => {
    button.classList.remove("selected");
    button.setAttribute("aria-pressed", "false");
  });
}

function showPreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  ui.preview.onload = () => {
    ui["image-dimensions"].textContent = `${ui.preview.naturalWidth} × ${ui.preview.naturalHeight} px`;
  };
  ui.preview.onerror = () => { ui["image-dimensions"].textContent = ""; };
  ui.preview.src = previewUrl;
  ui.preview.hidden = false;
  ui["upload-placeholder"].hidden = true;
  ui["replace-hint"].hidden = false;
  ui["file-name"].textContent = file.name;
  ui["image-dimensions"].textContent = "";
}

async function responseJson(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("Сервис вернул непонятный ответ. Попробуйте ещё раз.");
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Не удалось обработать изображение. Попробуйте ещё раз.";
    throw new Error(detail);
  }
  return data;
}

function displayPrediction(data) {
  const labels = data.classes || classes;
  if (!Array.isArray(labels) || !Array.isArray(data.probabilities)
      || labels.length !== data.probabilities.length || labels.length < 5
      || labels.some((label) => typeof label !== "string" || !label)
      || data.probabilities.some((value) => !Number.isFinite(value) || value < 0 || value > 1)
      || typeof data.model_name !== "string" || typeof data.model_version !== "string") {
    throw new Error("В ответе модели не хватает данных для отображения. Попробуйте ещё раз.");
  }
  const ranked = data.probabilities.map((value, index) => ({ name: labels[index], value }))
    .sort((a, b) => b.value - a.value).slice(0, 5);
  ui["winner-name"].textContent = ranked[0].name;
  ui["winner-confidence"].textContent = `${percent.format(ranked[0].value * 100)} %`;
  ui.probabilities.replaceChildren(...ranked.map(({ name, value }) => {
    const item = document.createElement("li");
    item.className = "probability";
    const label = document.createElement("div");
    label.className = "probability-label";
    const className = document.createElement("span");
    className.textContent = name;
    const confidence = document.createElement("span");
    confidence.className = "probability-value";
    confidence.textContent = `${percent.format(value * 100)} %`;
    label.append(className, confidence);
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = `${value * 100}%`;
    track.append(fill);
    item.append(label, track);
    return item;
  }));
  ui["model-name"].textContent = data.model_name;
  ui["model-version"].textContent = data.model_version;
  setState("ready");
}

async function classify(file, exampleButton = null, imageUrl = null) {
  const currentId = ++requestId;
  if (activeRequest) activeRequest.abort();
  const controller = new AbortController();
  activeRequest = controller;
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 30000);
  clearSelection();
  setState("loading");
  try {
    if (imageUrl) {
      const url = new URL(imageUrl, window.location.origin);
      if (url.origin !== window.location.origin) throw new Error("Пример недоступен. Выберите своё изображение.");
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error("Не удалось загрузить пример. Попробуйте другой или выберите своё изображение.");
      const blob = await response.blob();
      file = new File([blob], "example.png", { type: blob.type || "image/png" });
    }
    if (currentId !== requestId) return;
    if (!file || !file.size) throw new Error("Файл пуст. Выберите изображение PNG или JPEG.");
    if (file.size > 5 * 1024 * 1024) throw new Error("Изображение слишком большое. Выберите файл до 5 МиБ.");
    if (file.type ? !["image/png", "image/jpeg"].includes(file.type) : !/\.(png|jpe?g)$/i.test(file.name)) {
      throw new Error("Этот формат не поддерживается. Выберите PNG или JPEG.");
    }
    lastFile = file;
    showPreview(file);
    if (exampleButton) {
      exampleButton.classList.add("selected");
      exampleButton.setAttribute("aria-pressed", "true");
    }
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/predict", { method: "POST", body: form, signal: controller.signal });
    const data = await responseJson(response);
    if (currentId === requestId) displayPrediction(data);
  } catch (error) {
    if (currentId !== requestId) return;
    let message;
    if (timedOut) {
      message = "Сервис не успел ответить. Попробуйте ещё раз через несколько секунд.";
    } else if (error instanceof TypeError) {
      message = "Не удалось связаться с сервисом. Проверьте соединение и попробуйте ещё раз.";
    } else {
      message = error.message || "Не удалось получить ответ. Попробуйте ещё раз.";
    }
    setState("error", message);
  } finally {
    window.clearTimeout(timeout);
    if (currentId === requestId) {
      activeRequest = null;
      ui["prediction-panel"].setAttribute("aria-busy", "false");
    }
  }
}

async function loadExamples() {
  ui["examples-error"].hidden = true;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10000);
  try {
    const data = await responseJson(await fetch("/api/examples", { signal: controller.signal }));
    if (!Array.isArray(data.classes) || !data.classes.length || !Array.isArray(data.examples)) throw new Error();
    classes = data.classes;
    ui["classes-description"].textContent = `Классы модели — ${classes.join(", ")}.`;
    ui["classes-description"].hidden = false;
    const buttons = data.examples.map((example) => {
      const url = new URL(example.image_url, window.location.origin);
      if (url.origin !== window.location.origin) throw new Error();
      const button = document.createElement("button");
      button.className = "example";
      button.type = "button";
      button.setAttribute("aria-label", `Распознать пример ${example.class_name}`);
      button.setAttribute("aria-pressed", "false");
      const image = document.createElement("img");
      image.src = url.href;
      image.alt = "";
      image.width = 46;
      image.height = 46;
      const label = document.createElement("span");
      label.textContent = example.class_name;
      const arrow = document.createElement("span");
      arrow.className = "example-arrow";
      arrow.textContent = "↗";
      arrow.setAttribute("aria-hidden", "true");
      button.append(image, label, arrow);
      button.addEventListener("click", () => classify(null, button, example.image_url));
      return button;
    });
    ui.examples.replaceChildren(...buttons);
  } catch {
    ui["examples-error"].hidden = false;
  } finally {
    window.clearTimeout(timeout);
  }
}

ui["drop-zone"].addEventListener("click", () => ui["file-input"].click());
ui["drop-zone"].addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    ui["file-input"].click();
  }
});
ui["file-input"].addEventListener("change", () => {
  const file = ui["file-input"].files[0];
  if (file) classify(file);
  ui["file-input"].value = "";
});
for (const name of ["dragenter", "dragover", "dragleave", "drop"]) {
  window.addEventListener(name, (event) => event.preventDefault());
}
ui["drop-zone"].addEventListener("dragenter", () => {
  dragDepth += 1;
  ui["drop-zone"].classList.add("drag-over");
});
ui["drop-zone"].addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) ui["drop-zone"].classList.remove("drag-over");
});
ui["drop-zone"].addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  ui["drop-zone"].classList.remove("drag-over");
  const file = event.dataTransfer.files[0];
  if (file) classify(file);
});
ui["retry-prediction"].addEventListener("click", () => { if (lastFile) classify(lastFile); });
ui["retry-examples"].addEventListener("click", loadExamples);
window.addEventListener("beforeunload", () => {
  if (activeRequest) activeRequest.abort();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
loadExamples();
