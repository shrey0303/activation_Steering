import { API_BASE } from './constants';

/**
 * Lightweight API client for all REST endpoints.
 */

const DEFAULT_TIMEOUT = 120_000;   // 2 minutes
const LONG_TIMEOUT = 660_000;      // 11 minutes (scan can take ~8 min on CPU; backend timeout = 600s)

async function request(path, options = {}) {
    const timeout = options._timeout || DEFAULT_TIMEOUT;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    const url = `${API_BASE}${path}`;
    try {
        const res = await fetch(url, {
            headers: { 'Content-Type': 'application/json', ...options.headers },
            ...options,
            signal: controller.signal,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || err.error || `HTTP ${res.status}`);
        }
        return res.json();
    } catch (e) {
        if (e.name === 'AbortError') {
            throw new Error(`Request to ${path} timed out after ${timeout / 1000}s`);
        }
        throw e;
    } finally {
        clearTimeout(timer);
    }
}

// ── System ───────────────────────────────────────────────────
// Health is at root /health (not under /api/v1), so use fetch directly
export const getHealth = () => fetch('/health').then(r => r.json());
export const getMetrics = () => request('/metrics');

// ── Models ───────────────────────────────────────────────────
export const getModels = () => request('/models');

export const loadModel = (modelName, { quantize = true, quantizationBits = 4, device = 'auto' } = {}) =>
    request('/models/load', {
        method: 'POST',
        body: JSON.stringify({
            model_name: modelName,
            quantize,
            quantization_bits: quantizationBits,
            device,
        }),
    });

export const getLoadStatus = () => request('/models/load-status');

export const unloadModel = () =>
    request('/models/unload', { method: 'POST' });

// ── Scan ─────────────────────────────────────────────────────
export const scanModel = (forceRescan = false) =>
    request('/scan', {
        method: 'POST',
        body: JSON.stringify({ force_rescan: forceRescan }),
        _timeout: LONG_TIMEOUT,
    });

// ── Analyze ──────────────────────────────────────────────────
export const analyzePrompt = (prompt, expectedResponse, behaviorDescription = null) =>
    request('/analyze', {
        method: 'POST',
        body: JSON.stringify({
            prompt,
            expected_response: expectedResponse || '',
            behavior_description: behaviorDescription,
            analysis_type: 'full',
        }),
        _timeout: LONG_TIMEOUT,
    });

// ── Generate ─────────────────────────────────────────────────
export const generateText = (prompt, steering = null, maxTokens = 200) =>
    request('/generate', {
        method: 'POST',
        body: JSON.stringify({
            prompt,
            max_tokens: maxTokens,
            temperature: 0.7,
            steering,
        }),
    });

// ── Activations ──────────────────────────────────────────────
export const getActivations = (prompt, aggregation = 'mean') =>
    request('/activations', {
        method: 'POST',
        body: JSON.stringify({ prompt, aggregation }),
    });

// ── Patches ──────────────────────────────────────────────────
export const exportPatch = (data) =>
    request('/patches/export', {
        method: 'POST',
        body: JSON.stringify(data),
        _timeout: LONG_TIMEOUT,  // CAA auto-compute takes ~60s per layer
    });

export const listPatches = () => request('/patches');

export const downloadPatch = (patchId) => request(`/patches/${patchId}`);

/**
 * Trigger a browser download of the patch JSON file.
 */
export const downloadPatchFile = (patchId) => {
    const url = `${API_BASE}/patches/download/${patchId}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
};

/**
 * Client-side fallback: download patch data as JSON blob.
 */
export const downloadPatchBlob = (patchData, filename = 'steerops_patch.json') => {
    const blob = new Blob([JSON.stringify(patchData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

// ── Features (PCA Feature Dictionary) ────────────────────────
export const extractFeatures = () =>
    request('/features/extract', { method: 'POST', _timeout: LONG_TIMEOUT });

export const getFeatures = () => request('/features');

export const getFeature = (featureId) => request(`/features/${featureId}`);

export const updateFeatureLabel = (featureId, label) =>
    request(`/features/${featureId}/label`, {
        method: 'PUT',
        body: JSON.stringify({ label }),
    });

// ── Remote (HuggingFace Inference API) ───────────────────────
export const remoteConnect = (modelName, hfToken = null) =>
    request('/models/remote-connect', {
        method: 'POST',
        body: JSON.stringify({ model_name: modelName, hf_token: hfToken }),
    });

export const remoteDisconnect = () =>
    request('/models/remote-disconnect', { method: 'POST' });

export const remoteScan = () =>
    request('/remote/scan', { method: 'POST', body: JSON.stringify({}) });

export const remoteGenerate = (prompt, maxTokens = 200, temperature = 0.7) =>
    request('/remote/generate', {
        method: 'POST',
        body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature }),
    });

export const remoteActivations = (prompt) =>
    request('/remote/activations', {
        method: 'POST',
        body: JSON.stringify({ prompt }),
    });

