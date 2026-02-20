import { useState, useCallback, useEffect, useRef } from 'react';

// ── Resizable panel constants ──────────────────────────────
const MIN_PANEL = 240;
const MAX_PANEL = 500;
const DEFAULT_PANEL = 320;
import StatusBar from './components/StatusBar';
import ControlPanel from './components/ControlPanel';
import ChatInterface from './components/ChatInterface';
import ActivationHeatmap from './components/ActivationHeatmap';
import FeatureBrowser from './components/FeatureBrowser';
import DiagnosticsPanel from './components/DiagnosticsPanel';
import ExportModal from './components/ExportModal';
import LoadingBanner from './components/LoadingBanner';
import ToastContainer, { useToast } from './components/Toast';
import useWebSocket from './hooks/useWebSocket';
import useAnalysis from './hooks/useAnalysis';
import {
    generateText, getActivations, getHealth, getMetrics, getModels,
    loadModel, unloadModel, getLoadStatus,
    remoteConnect, remoteDisconnect, remoteScan, remoteGenerate, remoteActivations,
} from './utils/api';

export default function App() {
    // ── State ──────────────────────────────────────────────────
    const [mode, setMode] = useState('local'); // Determines UI view

    // Core model info per mode
    const [localModelInfo, setLocalModelInfo] = useState(null);
    const [remoteModelInfo, setRemoteModelInfo] = useState(null);
    const modelInfo = mode === 'remote' ? remoteModelInfo : localModelInfo;
    const isRemote = mode === 'remote';

    // Model loading info
    const [modelLoading, setModelLoading] = useState(false);
    const [loadingModelName, setLoadingModelName] = useState('');

    // App features per mode
    const [localActivations, setLocalActivations] = useState(null);
    const [remoteActivations, setRemoteActivations] = useState(null);
    const activations = mode === 'remote' ? remoteActivations : localActivations;

    // Local-only steering state
    const [localSelectedLayers, setLocalSelectedLayers] = useState([]);
    const [localSteeringStrength, setLocalSteeringStrength] = useState(0);
    
    // Derived steering values (remote cannot be steered)
    const selectedLayers = mode === 'remote' ? [] : localSelectedLayers;
    const steeringStrength = mode === 'remote' ? 0 : localSteeringStrength;

    // Helper setters (only update local state)
    const setSelectedLayers = useCallback((updater) => {
        if (mode !== 'remote') setLocalSelectedLayers(updater);
    }, [mode]);

    const setSteeringStrength = useCallback((updater) => {
        if (mode !== 'remote') setLocalSteeringStrength(updater);
    }, [mode]);

    // Shared state
    const [localMessages, setLocalMessages] = useState([]);
    const [remoteMessages, setRemoteMessages] = useState([]);
    const messages = mode === 'remote' ? remoteMessages : localMessages;
    
    // For setMessages, provide a helper that updates the current mode's array
    const setMessages = useCallback((updater) => {
        if (mode === 'remote') {
            setRemoteMessages(updater);
        } else {
            setLocalMessages(updater);
        }
    }, [mode]);

    const [metrics, setMetrics] = useState(null);
    const [streamingText, setStreamingText] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const [showExportModal, setShowExportModal] = useState(false);
    const [selectedFeatures, setSelectedFeatures] = useState([]);
    const [lastMetrics, setLastMetrics] = useState(null);
    const streamRef = useRef('');
    const stoppedRef = useRef(false);

    // ── Resizable panels ─────────────────────────────────────
    const [leftWidth, setLeftWidth] = useState(DEFAULT_PANEL);
    const [rightWidth, setRightWidth] = useState(DEFAULT_PANEL);
    const dragging = useRef(null); // 'left' | 'right' | null

    useEffect(() => {
        const onMouseMove = (e) => {
            if (!dragging.current) return;
            if (dragging.current === 'left') {
                setLeftWidth(Math.min(MAX_PANEL, Math.max(MIN_PANEL, e.clientX)));
            } else {
                setRightWidth(Math.min(MAX_PANEL, Math.max(MIN_PANEL, window.innerWidth - e.clientX)));
            }
        };
        const onMouseUp = () => { dragging.current = null; document.body.style.cursor = ''; document.body.style.userSelect = ''; };
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
        return () => { window.removeEventListener('mousemove', onMouseMove); window.removeEventListener('mouseup', onMouseUp); };
    }, []);

    // ── Analysis Hook ──────────────────────────────────────────
    const analysis = useAnalysis();

    // Keep analysis hook mode in sync with app mode
    useEffect(() => {
        analysis.setMode(mode);
    }, [mode, analysis]);

    // ── Toast Notifications ──────────────────────────────────────
    const { toasts, addToast, removeToast } = useToast();

    // ── WebSocket ──────────────────────────────────────────────
    const onToken = useCallback((data) => {
        if (stoppedRef.current) return; // Ignore tokens after stop
        streamRef.current += data.text;
        setStreamingText(streamRef.current);
        // Forward per-token diagnostics to DiagnosticsPanel
        if (data.diagnostics) {
            setLastMetrics(prev => ({
                ...prev,
                steering_diagnostics: data.diagnostics,
                active_interventions: data.diagnostics.active_interventions || [],
                entropy: data.diagnostics.entropy,
                circuit_breaker_triggered: data.diagnostics.entropy > 6.0,
            }));
        }
    }, []);

    const onDone = useCallback((data) => {
        const wasStopped = stoppedRef.current;
        setMessages((prev) => [
            ...prev,
            {
                role: 'assistant',
                content: streamRef.current,
                steered: data.metadata?.steering_applied,
                metrics: data.metadata,
                stopped: wasStopped,
            },
        ]);
        setStreamingText('');
        streamRef.current = '';
        setIsGenerating(false);
        stoppedRef.current = false;
        if (data.metadata) setLastMetrics(data.metadata);
    }, []);

    const onError = useCallback((data) => {
        setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: `Error: ${data.error?.message}` },
        ]);
        setStreamingText('');
        streamRef.current = '';
        setIsGenerating(false);
        stoppedRef.current = false;
    }, []);

    const { isConnected, sendMessage } = useWebSocket({ onToken, onDone, onError });

    // ── Stop Generation ─────────────────────────────────────────
    const handleStop = useCallback(() => {
        stoppedRef.current = true;
        // Send stop signal to WebSocket
        sendMessage({ type: 'stop' });
        // Immediately finalize current streaming text
        if (streamRef.current) {
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: streamRef.current,
                    stopped: true,
                },
            ]);
        }
        setStreamingText('');
        streamRef.current = '';
        setIsGenerating(false);
    }, [sendMessage]);

    // ── Polling for model info ─────────────────────────────────
    const fetchModelInfo = useCallback(async () => {
        try {
            const models = await getModels();
            if (models.models?.[0]) setLocalModelInfo(models.models[0]);
            else setLocalModelInfo(null);
        } catch { }
        try {
            const m = await getMetrics();
            setMetrics(m);
        } catch { }
    }, []);

    useEffect(() => {
        fetchModelInfo();
        const interval = setInterval(fetchModelInfo, 15000);
        return () => clearInterval(interval);
    }, [fetchModelInfo]);

    // ── Model Load / Unload ────────────────────────────────────
    const handleLoadModel = useCallback(
        async (modelName, options = {}) => {
            setModelLoading(true);
            setLoadingModelName(modelName);
            addToast(`Starting download of ${modelName}…`, 'info', 4000);
            try {
                // Fire-and-forget: backend loads in background thread
                await loadModel(modelName, options);

                // Poll for completion every 3 seconds
                const poll = async () => {
                    try {
                        const status = await getLoadStatus();
                        if (status.status === 'done' && status.model) {
                            setLocalModelInfo(status.model);
                            setModelLoading(false);
                            setLoadingModelName('');
                            setSelectedLayers([]);
                            setSteeringStrength(0);
                            setLocalActivations(null);
                            setMode('local');
                            analysis.reset('local');
                            addToast(`${modelName} loaded! Auto-scanning layers…`, 'success', 8000);
                            // Auto-scan: trigger scan immediately after model load
                            setTimeout(() => analysis.scan(false), 500);
                            return;
                        } else if (status.status === 'error') {
                            addToast(`Failed to load ${modelName}: ${status.error}`, 'error', 10000);
                            setModelLoading(false);
                            setLoadingModelName('');
                            return;
                        }
                        // Still loading — poll again
                        setTimeout(poll, 3000);
                    } catch {
                        setTimeout(poll, 3000);
                    }
                };
                setTimeout(poll, 2000);
            } catch (e) {
                addToast(`Failed to load ${modelName}: ${e.message}`, 'error', 10000);
                setModelLoading(false);
                setLoadingModelName('');
            }
        },
        [analysis, addToast],
    );

    const handleUnloadModel = useCallback(async () => {
        try {
            if (isRemote) {
                // Remote mode: disconnect instead of unload
                await remoteDisconnect();
                setRemoteModelInfo(null);
                setRemoteActivations(null);
                analysis.reset('remote');
            } else {
                await unloadModel();
                setLocalModelInfo(null);
                setSelectedLayers([]);
                setSteeringStrength(0);
                setLocalActivations(null);
                analysis.reset('local');
            }
        } catch (e) {
            setMessages((prev) => [
                ...prev,
                { role: 'system', content: `Failed to ${isRemote ? 'disconnect' : 'unload'}: ${e.message}` },
            ]);
        }
    }, [analysis, isRemote]);

    // ── Remote Connect / Disconnect ─────────────────────────────
    const handleRemoteConnect = useCallback(
        async (modelName, hfToken = null) => {
            // If a local model is still downloading, don't interrupt it.
            const localDownloading = modelLoading && !isRemote;

            if (!localDownloading) {
                setModelLoading(true);
            }
            try {
                const res = await remoteConnect(modelName, hfToken);
                if (res.model) {
                    setRemoteModelInfo(res.model);
                }
                setRemoteActivations(null);
                analysis.reset('remote');
                setMode('remote');
            } catch (e) {
                let msg = e.message;
                // Friendlier error messages
                if (msg.includes('Failed to resolve') || msg.includes('getaddrinfo')) {
                    msg = `Network error: Cannot reach HuggingFace. Check your internet connection.`;
                } else if (msg.includes('401') || msg.includes('Unauthorized')) {
                    msg = `Authentication required for "${modelName}". Please provide an HF Token.`;
                } else if (msg.includes('403') || msg.includes('Access denied')) {
                    msg = `Access denied for "${modelName}". Your HF Token may lack permissions for this model.`;
                } else if (msg.includes('404')) {
                    msg = `Model "${modelName}" not found on HuggingFace Hub.`;
                }
                addToast(msg, 'error', 8000);
            } finally {
                if (!localDownloading) {
                    setModelLoading(false);
                }
            }
        },
        [analysis, modelLoading, isRemote, addToast],
    );

    // This is now redundant with handleUnloadModel handling both scenarios based on mode,
    // but preserving for direct calls
    const handleRemoteDisconnect = useCallback(async () => {
        try {
            await remoteDisconnect();
            setRemoteModelInfo(null);
            setRemoteActivations(null);
            analysis.reset('remote');
        } catch (e) {
            setMessages((prev) => [
                ...prev,
                { role: 'system', content: `Remote disconnect failed: ${e.message}` },
            ]);
        }
    }, [analysis]);

    // ── Handlers ───────────────────────────────────────────────
    const handleSend = useCallback(
        async (prompt) => {
            setMessages((prev) => [...prev, { role: 'user', content: prompt }]);
            setIsGenerating(true);
            streamRef.current = '';
            stoppedRef.current = false;
            setStreamingText('');

            const steeringEnabled = selectedLayers.length > 0 && steeringStrength !== 0;

            if (isRemote) {
                // Remote mode: use HF Inference API
                try {
                    const result = await remoteGenerate(prompt, 200, 0.7);
                    setMessages((prev) => [
                        ...prev,
                        {
                            role: 'assistant',
                            content: result.text,
                            steered: false,
                            metrics: result,
                            remote: true,
                        },
                    ]);
                } catch (e) {
                    let msg = e.message;
                    if (msg.includes('401') || msg.includes('Unauthorized')) {
                        msg = 'HF Inference API requires authentication. Add your HF Token in the Control Panel.';
                    } else if (msg.includes('Failed to resolve') || msg.includes('getaddrinfo')) {
                        msg = 'Cannot reach HuggingFace servers. Check your internet connection.';
                    }
                    setMessages((prev) => [
                        ...prev,
                        { role: 'assistant', content: `Remote error: ${msg}` },
                    ]);
                }
                setIsGenerating(false);

                // Simulated activations
                try {
                    const act = await remoteActivations(prompt);
                    setRemoteActivations(act.activations);
                } catch { }
            } else if (isConnected) {
                // Local mode: WebSocket streaming
                sendMessage({
                    type: 'generate',
                    prompt,
                    max_tokens: 200,
                    temperature: 0.7,
                    steering: steeringEnabled
                        ? { layer: selectedLayers[0], strength: steeringStrength }
                        : null,
                });

                // Capture local activations
                try {
                    const act = await getActivations(prompt);
                    setActivations(act.activations);
                } catch { }
            } else {
                // Local fallback to REST
                try {
                    const result = await generateText(
                        prompt,
                        steeringEnabled
                            ? { layer: selectedLayers[0], strength: steeringStrength }
                            : null,
                    );
                    setMessages((prev) => [
                        ...prev,
                        {
                            role: 'assistant',
                            content: result.text,
                            steered: result.steering_applied,
                            metrics: result,
                        },
                    ]);
                } catch (e) {
                    setMessages((prev) => [
                        ...prev,
                        { role: 'assistant', content: `Error: ${e.message}` },
                    ]);
                }
                setIsGenerating(false);

                try {
                    const act = await getActivations(prompt);
                    setActivations(act.activations);
                } catch { }
            }
        },
        [isConnected, isRemote, sendMessage, selectedLayers, steeringStrength],
    );

    const handleSelectLayer = useCallback((layerIdx) => {
        setSelectedLayers((prev) =>
            prev.includes(layerIdx)
                ? prev.filter((l) => l !== layerIdx)
                : [...prev, layerIdx],
        );
    }, []);

    const handleScan = useCallback(
        async (forceRescan = false) => {
            if (isRemote) {
                // Remote scan — use config-based scan
                try {
                    const result = await remoteScan();
                    // Store in analysis hook's scanResult format
                    analysis.setScanResult(result);
                } catch (e) {
                    setMessages((prev) => [
                        ...prev,
                        { role: 'system', content: `Remote scan failed: ${e.message}` },
                    ]);
                }
            } else {
                await analysis.scan(forceRescan);
            }
        },
        [analysis, isRemote],
    );

    const handleAnalyze = useCallback(
        async (prompt, expectedResponse, behaviorDescription = null) => {
            const result = await analysis.analyze(prompt, expectedResponse, behaviorDescription);
            if (result?.detected_layers?.length) {
                setSelectedLayers(result.detected_layers);
                // Auto-set recommended strength from first resolved layer
                const firstLayer = Object.values(result.detailed_analysis || {})[0];
                if (firstLayer?.recommended_intervention?.strength != null) {
                    setSteeringStrength(firstLayer.recommended_intervention.strength);
                }
            }
        },
        [analysis],
    );

    const steeringEnabled = selectedLayers.length > 0 && steeringStrength !== 0;

    // ── Render ─────────────────────────────────────────────────
    return (
        <div className="flex flex-col h-screen">
            <StatusBar
                modelInfo={modelInfo}
                isConnected={isConnected}
                metrics={metrics}
            />

            {/* Loading Banner — non-blocking, shown while model downloads */}
            {modelLoading && loadingModelName && (
                <LoadingBanner modelName={loadingModelName} />
            )}

            {/* Toast Notifications */}
            <ToastContainer toasts={toasts} onRemove={removeToast} />

            <div className="flex-1 flex overflow-hidden">
                {/* Left: Control Panel */}
                <div style={{ width: leftWidth, minWidth: MIN_PANEL, maxWidth: MAX_PANEL }} className="border-r border-border-subtle bg-bg-secondary flex flex-col flex-shrink-0">
                    <ControlPanel
                        appMode={mode}
                        onModeChange={setMode}
                        scanResult={analysis.scanResult}
                        analysisResult={analysis.analysisResult}
                        selectedLayers={selectedLayers}
                        steeringStrength={steeringStrength}
                        onScan={handleScan}
                        onAnalyze={handleAnalyze}
                        onSelectLayer={handleSelectLayer}
                        onStrengthChange={setSteeringStrength}
                        onClearLayers={() => setSelectedLayers([])}
                        onLoadModel={handleLoadModel}
                        onUnloadModel={handleUnloadModel}
                        onRemoteConnect={handleRemoteConnect}
                        onRemoteDisconnect={handleRemoteDisconnect}
                        modelInfo={modelInfo}
                        modelLoading={modelLoading}
                        scanLoading={analysis.scanLoading}
                        analysisLoading={analysis.loading}
                        isRemote={isRemote}
                    />

                    {/* Export Button */}
                    {!isRemote && steeringEnabled && (
                        <div className="p-4 border-t border-border-subtle">
                            <button
                                onClick={() => setShowExportModal(true)}
                                className="w-full py-2 rounded-lg text-sm font-medium
                  bg-accent-blue/20 border border-accent-blue/30
                  hover:border-accent-blue/60 transition"
                            >
                                Export Patch
                            </button>
                        </div>
                    )}

                    {/* Feature Browser — PCA Feature Dictionary */}
                    <div className="flex-1 overflow-hidden border-t border-border-subtle">
                        <FeatureBrowser
                            modelInfo={modelInfo}
                            selectedFeatures={selectedFeatures}
                            onSelectFeature={(fid) => {
                                setSelectedFeatures(prev =>
                                    prev.includes(fid)
                                        ? prev.filter(f => f !== fid)
                                        : [...prev, fid]
                                );
                            }}
                        />
                    </div>
                </div>

                {/* Left resize handle */}
                <div
                    className="w-1 cursor-col-resize hover:bg-accent-blue/40 active:bg-accent-blue/60 transition-colors flex-shrink-0"
                    onMouseDown={() => { dragging.current = 'left'; document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none'; }}
                />

                {/* Centre: Chat */}
                <div className="flex-1 bg-bg-primary flex flex-col">
                    <ChatInterface
                        messages={messages}
                        streamingText={streamingText}
                        isGenerating={isGenerating}
                        onSend={handleSend}
                        onStop={handleStop}
                        steeringEnabled={steeringEnabled}
                        selectedLayers={selectedLayers}
                        steeringStrength={steeringStrength}
                    />
                </div>

                {/* Right resize handle */}
                <div
                    className="w-1 cursor-col-resize hover:bg-accent-blue/40 active:bg-accent-blue/60 transition-colors flex-shrink-0"
                    onMouseDown={() => { dragging.current = 'right'; document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none'; }}
                />

                {/* Right: Heatmap */}
                <div style={{ width: rightWidth, minWidth: MIN_PANEL, maxWidth: MAX_PANEL }} className="border-l border-border-subtle bg-bg-secondary flex flex-col flex-shrink-0">
                    <div className="flex-1 overflow-hidden">
                        <ActivationHeatmap
                            activations={activations}
                            scanResult={analysis.scanResult}
                            analysisResult={analysis.analysisResult}
                            selectedLayers={selectedLayers}
                            onSelectLayer={handleSelectLayer}
                        />
                    </div>
                    <DiagnosticsPanel
                        metrics={lastMetrics}
                        isGenerating={isGenerating}
                    />
                </div>
            </div>

            {/* Export Modal */}
            {showExportModal && (
                <ExportModal
                    onClose={() => setShowExportModal(false)}
                    selectedLayers={selectedLayers}
                    steeringStrength={steeringStrength}
                    analysisResult={analysis.analysisResult}
                />
            )}
        </div>
    );
}
