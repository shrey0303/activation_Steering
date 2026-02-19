import { useState, useMemo, useRef, useEffect } from 'react';
import { CATEGORY_COLORS, CATEGORY_LABELS, SUPPORTED_MODELS } from '../utils/constants';

const CATEGORY_DESCRIPTIONS = {
    token_embedding: 'Raw vocabulary lookup',
    positional_morphological: 'Position & morphology',
    syntactic_processing: 'Grammar & dependencies',
    entity_semantic: 'Entity recognition & semantics',
    knowledge_retrieval: 'Factual recall (FF key-value)',
    reasoning_planning: 'Logic & multi-step inference',
    safety_alignment: 'Safety filters & refusal',
    information_integration: 'Cross-layer merging',
    style_personality: 'Tone & personality',
    output_distribution: 'Token probability shaping',
};

/**
 * Left panel: model selector, model scan, layer selector
 * with details, strength slider, and intuitive analysis.
 */
export default function ControlPanel({
    scanResult,
    analysisResult,
    selectedLayers,
    steeringStrength,
    onScan,
    onAnalyze,
    onSelectLayer,
    onStrengthChange,
    onClearLayers,
    onLoadModel,
    onUnloadModel,
    onRemoteConnect,
    onRemoteDisconnect,
    modelInfo,
    modelLoading,
    scanLoading,
    analysisLoading,
    isRemote,
    appMode,
    onModeChange,
}) {
    const [prompt, setPrompt] = useState('');
    const [expectedResponse, setExpectedResponse] = useState('');
    const [behaviorDescription, setBehaviorDescription] = useState('');
    const [analysisMode, setAnalysisMode] = useState('response'); // 'response' | 'behavior'
    const [modelInput, setModelInput] = useState('');
    const [quantize, setQuantize] = useState(true);
    const [showAnalysisHelp, setShowAnalysisHelp] = useState(false);
    const mode = appMode;
    const setMode = onModeChange;
    const [hfToken, setHfToken] = useState('');
    const [showToken, setShowToken] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const dropdownRef = useRef(null);
    const inputRef = useRef(null);

    const layers = scanResult?.layer_profiles || [];
    const detectedLayers = analysisResult?.detected_layers || [];

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Filter models based on search input
    const filteredModels = useMemo(() => {
        const query = modelInput.toLowerCase().trim();
        if (!query) return SUPPORTED_MODELS;
        return SUPPORTED_MODELS.filter(m =>
            m.name.toLowerCase().includes(query) ||
            m.id.toLowerCase().includes(query) ||
            m.arch.toLowerCase().includes(query) ||
            m.params.toLowerCase().includes(query)
        );
    }, [modelInput]);

    // Group filtered models by architecture
    const groupedModels = useMemo(() => {
        const groups = {};
        filteredModels.forEach(m => {
            if (!groups[m.arch]) groups[m.arch] = [];
            groups[m.arch].push(m);
        });
        return groups;
    }, [filteredModels]);

    const categoryGroups = useMemo(() => {
        const groups = {};
        layers.forEach((l) => {
            const cat = l.category || 'unknown';
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push(l);
        });
        return groups;
    }, [layers]);

    const handleAnalyze = () => {
        if (analysisMode === 'behavior') {
            if (behaviorDescription.trim()) {
                onAnalyze(prompt.trim() || '', '', behaviorDescription);
            }
        } else {
            if (prompt.trim() && expectedResponse.trim()) {
                onAnalyze(prompt, expectedResponse, null);
            }
        }
    };

    const canAnalyze = analysisMode === 'behavior'
        ? behaviorDescription.trim()
        : (prompt.trim() && expectedResponse.trim());

    const handleLoadModel = (modelId = null) => {
        const name = (modelId || modelInput).replace(/\s+/g, '').trim();
        if (!name) return;
        setShowDropdown(false);

        if (mode === 'remote') {
            // Don't cancel an in-progress local download
            if (modelLoading && !isRemote) {
                // Keep local download going — just connect remote in parallel
            }
            onRemoteConnect(name, hfToken || null);
        } else {
            onLoadModel(name, { quantize });
        }
    };

    const handleSelectModel = (model) => {
        setModelInput(model.id);
        setShowDropdown(false);
        // Auto-load after selection
        setTimeout(() => {
            const name = model.id.trim();
            if (mode === 'remote') {
                onRemoteConnect(name, hfToken || null);
            } else {
                onLoadModel(name, { quantize });
            }
        }, 100);
    };

    const handleDisconnect = () => {
        if (isRemote) {
            onRemoteDisconnect();
        } else {
            onUnloadModel();
        }
    };

    return (
        <div className="flex flex-col h-full overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 border-b border-border-subtle flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-accent-green animate-pulse-subtle" />
                <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary">
                    Control Panel
                </h2>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-5">

                {/* ── Model Selector ───────────────────────────────── */}
                <section>
                    <h3 className="text-xs font-semibold uppercase text-text-tertiary mb-2 tracking-wider">
                        Model
                    </h3>

                    {/* Mode Toggle */}
                    <div className="flex rounded-lg overflow-hidden border border-border-subtle mb-3">
                        <button
                            onClick={() => setMode('local')}
                            className={`flex-1 py-1.5 text-[11px] font-medium transition-all ${mode === 'local'
                                ? 'bg-accent-blue/20 text-accent-blue border-r border-accent-blue/30'
                                : 'bg-bg-primary text-text-tertiary border-r border-border-subtle hover:text-text-secondary'
                                }`}
                        >
                            Local
                        </button>
                        <button
                            onClick={() => setMode('remote')}
                            className={`flex-1 py-1.5 text-[11px] font-medium transition-all ${mode === 'remote'
                                ? 'bg-accent-purple/20 text-accent-purple'
                                : 'bg-bg-primary text-text-tertiary hover:text-text-secondary'
                                }`}
                        >
                            Remote (HF API)
                        </button>
                    </div>

                    {/* Active Model Badge */}
                    {modelInfo?.loaded && (
                        <div className={`mb-2 p-2.5 rounded-lg text-xs border ${isRemote
                            ? 'bg-accent-purple/10 border-accent-purple/20'
                            : 'bg-accent-green/10 border-accent-green/20'
                            }`}>
                            <div className="flex justify-between items-center">
                                <span className={`font-medium truncate mr-2 ${isRemote ? 'text-accent-purple' : 'text-accent-green'
                                    }`}>
                                    {isRemote && '⊕ '}{modelInfo.name}
                                </span>
                                <button
                                    onClick={handleDisconnect}
                                    className="text-[10px] text-accent-red/70 hover:text-accent-red border border-accent-red/30
                                     px-1.5 py-0.5 rounded transition flex-shrink-0"
                                >
                                    {isRemote ? 'Disconnect' : 'Unload'}
                                </button>
                            </div>
                            <div className="flex gap-3 mt-1 text-text-tertiary flex-wrap">
                                <span>{modelInfo.device}</span>
                                {!isRemote && <span>{modelInfo.memory_mb?.toFixed(0)}MB</span>}
                                {modelInfo.quantized && (
                                    <span className="text-accent-amber">{modelInfo.quantization_bits}-bit</span>
                                )}
                                <span>{modelInfo.num_layers} layers</span>
                                {modelInfo.hidden_dim > 0 && (
                                    <span>dim={modelInfo.hidden_dim}</span>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Searchable Model Dropdown */}
                    <div className="relative" ref={dropdownRef}>
                        <div className="flex gap-1.5">
                            <input
                                ref={inputRef}
                                value={modelInput}
                                onChange={(e) => { setModelInput(e.target.value); setShowDropdown(true); }}
                                onFocus={() => setShowDropdown(true)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter') { handleLoadModel(); }
                                    if (e.key === 'Escape') { setShowDropdown(false); }
                                }}
                                placeholder="Search models…"
                                className="flex-1 px-3 py-2 rounded-lg bg-bg-primary border border-border-subtle
                placeholder:text-text-tertiary text-xs
                focus:outline-none focus:border-accent-green/50 transition"
                            />
                            <button
                                onClick={() => handleLoadModel()}
                                disabled={modelLoading || !modelInput.trim()}
                                className={`px-3 py-2 rounded-lg text-xs font-medium transition-all border
                disabled:opacity-40 disabled:cursor-not-allowed ${mode === 'remote'
                                        ? 'bg-gradient-to-r from-accent-purple/20 to-accent-blue/20 border-accent-purple/30 hover:border-accent-purple/60'
                                        : 'bg-gradient-to-r from-accent-blue/20 to-accent-purple/20 border-accent-blue/30 hover:border-accent-blue/60'
                                    }`}
                            >
                                {modelLoading ? '…' : '→'}
                            </button>
                        </div>

                        {/* Dropdown */}
                        {showDropdown && (
                            <div className="absolute z-50 left-0 right-0 mt-1 max-h-72 overflow-y-auto
                                rounded-lg border border-border-subtle bg-bg-secondary shadow-xl
                                animate-fade-in">
                                {Object.entries(groupedModels).length > 0 ? (
                                    Object.entries(groupedModels).map(([arch, models]) => (
                                        <div key={arch}>
                                            <div className="px-3 py-1.5 text-[9px] font-semibold uppercase tracking-wider
                                                text-text-tertiary bg-bg-primary/80 sticky top-0 border-b border-border-subtle">
                                                {arch}
                                            </div>
                                            {models.map(m => (
                                                <button
                                                    key={m.id}
                                                    onClick={() => handleSelectModel(m)}
                                                    className="w-full text-left px-3 py-2 flex items-center justify-between
                                                        hover:bg-accent-blue/10 transition-colors border-b border-border-subtle/50
                                                        last:border-0 group"
                                                >
                                                    <div className="min-w-0 flex-1">
                                                        <div className="text-xs font-medium text-text-primary truncate group-hover:text-accent-blue transition">
                                                            {m.name}
                                                        </div>
                                                        <div className="text-[9px] text-text-tertiary truncate font-mono mt-0.5">
                                                            {m.id}
                                                        </div>
                                                    </div>
                                                    <div className="flex items-center gap-2 ml-2 flex-shrink-0">
                                                        <span className="text-[9px] font-mono text-accent-amber">{m.params}</span>
                                                        <span className="text-[8px] text-text-tertiary">{m.vram}</span>
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    ))
                                ) : (
                                    <div className="px-3 py-4 text-xs text-text-tertiary text-center">
                                        No matching models found.
                                        <br />
                                        <span className="text-[9px]">Press Enter to try loading "{modelInput}" directly from HuggingFace</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Mode-specific options */}
                    {mode === 'local' && (
                        <label className="flex items-center gap-2 mt-2 text-xs text-text-tertiary cursor-pointer select-none">
                            <input type="checkbox" checked={quantize} onChange={(e) => setQuantize(e.target.checked)}
                                className="rounded accent-accent-green" />
                            4-bit quantization (GPU only)
                        </label>
                    )}

                    {mode === 'remote' && (
                        <div className="mt-2 space-y-1.5">
                            <div className="flex items-center gap-1">
                                <button
                                    onClick={() => setShowToken(!showToken)}
                                    className="text-[10px] text-accent-purple hover:text-accent-purple/80 transition"
                                >
                                    {showToken ? 'Hide token' : 'HF Token (optional)'}
                                </button>
                            </div>
                            {showToken && (
                                <input
                                    type="password"
                                    value={hfToken}
                                    onChange={(e) => setHfToken(e.target.value)}
                                    placeholder="hf_xxxxxxxxxx…"
                                    className="w-full px-3 py-1.5 rounded-lg bg-bg-primary border border-border-subtle
                        placeholder:text-text-tertiary text-xs
                        focus:outline-none focus:border-accent-purple/50 transition"
                                />
                            )}
                            <p className="text-[9px] text-text-tertiary">
                                Remote mode: instant scan, generation via HF servers. No model download needed.
                            </p>
                        </div>
                    )}

                    {modelLoading && (
                        <div className={`mt-2 p-2 rounded-lg text-xs animate-pulse border ${mode === 'remote'
                            ? 'bg-accent-purple/10 border-accent-purple/20 text-accent-purple'
                            : 'bg-accent-blue/10 border-accent-blue/20 text-accent-blue'
                            }`}>
                            {mode === 'remote'
                                ? 'Connecting to HuggingFace… Fetching model config.'
                                : 'Downloading & loading model… This may take a few minutes.'}
                        </div>
                    )}
                </section>

                {/* ── Scan ──────────────────────────────────────────── */}
                <section>
                    <h3 className="text-xs font-semibold uppercase text-text-tertiary mb-2 tracking-wider">
                        Model Scan
                    </h3>
                    <button
                        onClick={() => onScan()}
                        disabled={scanLoading || (!modelInfo?.loaded && !modelLoading)}
                        className={`w-full py-2.5 rounded-lg font-medium text-sm transition-all
                  bg-gradient-to-r ${isRemote
                            ? 'from-accent-purple/20 to-accent-blue/20 border border-accent-purple/30 hover:border-accent-purple/60 hover:shadow-lg hover:shadow-accent-purple/10'
                            : 'from-accent-green/20 to-accent-blue/20 border border-accent-green/30 hover:border-accent-green/60 hover:shadow-lg hover:shadow-accent-green/10'
                        } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                        {scanLoading ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                Scanning…
                            </span>
                        ) : scanResult
                            ? 'Re-Scan Model'
                            : isRemote
                                ? 'Remote Scan (Config-Based)'
                                : 'Scan Model Layers'}
                    </button>
                    {isRemote && (
                        <p className="text-[9px] text-accent-purple/70 mt-1">
                            Remote scan uses config metadata — no weights analysed.
                        </p>
                    )}

                    {scanResult && (
                        <div className="mt-3 p-3 rounded-lg bg-bg-primary/60 border border-border-subtle text-xs space-y-1 animate-fade-in">
                            <div className="flex justify-between">
                                <span className="text-text-tertiary">Architecture</span>
                                <span className="font-mono">{scanResult.architecture}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-text-tertiary">Layers</span>
                                <span className="font-mono">{scanResult.num_layers}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-text-tertiary">Hidden Dim</span>
                                <span className="font-mono">{scanResult.hidden_dim}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-text-tertiary">Scan Time</span>
                                <span className="font-mono">
                                    {scanResult.from_cache ? 'cached' : `${scanResult.scan_time_ms?.toFixed(0)}ms`}
                                </span>
                            </div>
                            {scanResult.remote && (
                                <div className="flex justify-between">
                                    <span className="text-text-tertiary">Mode</span>
                                    <span className="font-mono text-accent-purple">config-based (remote)</span>
                                </div>
                            )}
                        </div>
                    )}
                </section>

                {/* ── Layer Map with Details ─────────────────────────── */}
                {layers.length > 0 && (
                    <section>
                        <div className="flex justify-between items-center mb-2">
                            <h3 className="text-xs font-semibold uppercase text-text-tertiary tracking-wider">
                                Layer Map
                            </h3>
                            {selectedLayers.length > 0 && (
                                <button onClick={onClearLayers}
                                    className="text-[10px] text-accent-red hover:text-red-400 transition">
                                    Clear ({selectedLayers.length})
                                </button>
                            )}
                        </div>

                        <div className="space-y-3">
                            {Object.entries(categoryGroups).map(([category, catLayers]) => (
                                <div key={category}>
                                    <div className="flex items-center gap-2 mb-1.5">
                                        <div className="w-2.5 h-2.5 rounded-full"
                                            style={{ backgroundColor: CATEGORY_COLORS[category] || '#6b7280' }} />
                                        <span className="text-xs font-medium text-text-secondary">
                                            {CATEGORY_LABELS[category] || category}
                                        </span>
                                        <span className="text-[9px] text-text-tertiary">
                                            — {CATEGORY_DESCRIPTIONS[category] || ''}
                                        </span>
                                    </div>
                                    <div className="flex flex-wrap gap-1">
                                        {catLayers.map((l) => {
                                            const isSelected = selectedLayers.includes(l.layer_index);
                                            const isAnalyzed = detectedLayers.includes(l.layer_index);
                                            return (
                                                <button
                                                    key={l.layer_index}
                                                    onClick={() => onSelectLayer(l.layer_index)}
                                                    className={`
                            px-2 py-1 rounded text-[10px] font-mono transition-all
                            border select-none relative group
                            ${isSelected
                                                            ? 'bg-accent-green/20 border-accent-green text-accent-green'
                                                            : isAnalyzed
                                                                ? 'bg-accent-amber/15 border-accent-amber/50 text-accent-amber'
                                                                : 'bg-bg-primary border-border-subtle text-text-tertiary hover:border-border-hover'
                                                        }
                          `}
                                                    title={`Layer ${l.layer_index} · ${CATEGORY_LABELS[l.category] || l.category}\nConfidence: ${(l.confidence * 100).toFixed(0)}%\n${l.description || ''}`}
                                                >
                                                    L{l.layer_index}
                                                    {isAnalyzed && <span className="ml-0.5 text-[8px] text-accent-amber">●</span>}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* ── Steering Strength ─────────────────────────────── */}
                {!isRemote && selectedLayers.length > 0 && (
                    <section className="animate-slide-in">
                        <h3 className="text-xs font-semibold uppercase text-text-tertiary mb-2 tracking-wider">
                            Steering Strength
                        </h3>
                        <div className="p-3 rounded-lg bg-bg-primary/60 border border-border-subtle space-y-2">
                            <div className="flex items-center gap-3">
                                <span className="text-[10px] text-accent-red font-mono w-5">-10</span>
                                <input type="range" min="-10" max="10" step="0.5"
                                    value={steeringStrength}
                                    onChange={(e) => onStrengthChange(parseFloat(e.target.value))}
                                    className="flex-1" />
                                <span className="text-[10px] text-accent-green font-mono w-5">+10</span>
                            </div>
                            <div className="text-center">
                                <span className={`text-lg font-mono font-bold ${steeringStrength > 0 ? 'text-accent-green'
                                    : steeringStrength < 0 ? 'text-accent-red' : 'text-text-tertiary'
                                    }`}>
                                    {steeringStrength > 0 ? '+' : ''}{steeringStrength.toFixed(1)}
                                </span>
                            </div>
                            <div className="text-[9px] text-text-tertiary text-center space-y-0.5">
                                <div>← <span className="text-accent-red">Suppress</span> · <span className="text-text-tertiary">Neutral</span> · <span className="text-accent-green">Amplify</span> →</div>
                                <div>Layers: {selectedLayers.map(l => `L${l}`).join(', ')}</div>
                            </div>
                        </div>
                    </section>
                )}

                {/* ── Behavior Analysis ─────────────────────────────── */}
                <section>
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs font-semibold uppercase text-text-tertiary tracking-wider">
                            Behavior Analysis
                        </h3>
                        {!isRemote && (
                            <button onClick={() => setShowAnalysisHelp(!showAnalysisHelp)}
                                className="text-[10px] text-accent-blue hover:text-accent-blue/80 transition">
                                {showAnalysisHelp ? 'Hide help' : '? How it works'}
                            </button>
                        )}
                    </div>

                    {isRemote ? (
                        <div className="p-3 rounded-lg bg-accent-purple/5 border border-accent-purple/20 text-xs space-y-2">
                            <p className="text-accent-purple font-medium">⊕ Remote Mode — Steering Unavailable</p>
                            <p className="text-text-tertiary text-[10px] leading-relaxed">
                                Activation steering uses <strong className="text-text-secondary">Contrastive Activation Addition (CAA)</strong> which
                                requires direct access to model internals — hidden states, gradients, and layer
                                activations. This is only possible with a locally loaded model.
                            </p>
                            <p className="text-text-tertiary text-[10px] leading-relaxed">
                                Remote mode via HuggingFace API only exposes text-in → text-out inference.
                                To steer model behavior, switch to <strong className="text-accent-blue">Local</strong> mode and load the model.
                            </p>
                            <div className="flex gap-2 mt-1">
                                <span className="text-[9px] px-2 py-0.5 rounded bg-accent-green/10 border border-accent-green/20 text-accent-green">
                                    ✓ Chat (remote)
                                </span>
                                <span className="text-[9px] px-2 py-0.5 rounded bg-accent-red/10 border border-accent-red/20 text-accent-red">
                                    ✗ Scan / Analyze / Steer
                                </span>
                            </div>
                        </div>
                    ) : (
                        <>
                            {showAnalysisHelp && (
                                <div className="mb-3 p-3 rounded-lg bg-accent-blue/5 border border-accent-blue/20 text-[10px] text-text-secondary space-y-1.5 animate-fade-in">
                                    <p className="font-medium text-accent-blue">How Behavior Analysis Works:</p>
                                    <p>1. <strong>Enter the prompt</strong> you sent to the model</p>
                                    <p>2. <strong>Enter the expected response</strong> — what the model <em>should</em> say</p>
                                    <p>3. Click <strong>Analyze</strong> — SteerOps compares the model's actual internal activations against the expected behavior</p>
                                    <p>4. Layers with <span className="text-accent-amber">anomalous behavior</span> are detected (● markers)</p>
                                    <p>5. The heatmap highlights these layers — you can then <span className="text-accent-green">amplify</span> (positive strength) or <span className="text-accent-red">suppress</span> (negative) them</p>
                                    <p className="text-text-tertiary mt-1">Example: If the model is rude, set expected to a polite version → detected layers control "politeness" → amplify them.</p>
                                </div>
                            )}

                            <div className="space-y-2">
                                <div>
                                    <label className="text-[10px] text-text-tertiary block mb-1">
                                        {analysisMode === 'behavior'
                                            ? 'Prompt context (optional — for prompt-specific behavior)'
                                            : 'Prompt used (what you sent to the model)'}
                                    </label>
                                    <textarea
                                        value={prompt}
                                        onChange={(e) => setPrompt(e.target.value)}
                                        placeholder={analysisMode === 'behavior'
                                            ? 'Optional: add a prompt to steer behavior for this specific input'
                                            : 'e.g. "Tell me about climate change"'}
                                        rows={2}
                                        className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-border-subtle
              placeholder:text-text-tertiary text-sm resize-none
              focus:outline-none focus:border-accent-green/50 transition"
                                    />
                                </div>

                                {/* Analysis Mode Toggle */}
                                <div className="flex rounded-lg overflow-hidden border border-border-subtle">
                                    <button
                                        onClick={() => setAnalysisMode('response')}
                                        className={`flex-1 py-1.5 text-[10px] font-medium transition-all ${analysisMode === 'response'
                                            ? 'bg-accent-purple/20 text-accent-purple border-r border-accent-purple/30'
                                            : 'bg-bg-primary text-text-tertiary border-r border-border-subtle hover:text-text-secondary'
                                            }`}
                                    >
                                        Expected Response
                                    </button>
                                    <button
                                        onClick={() => setAnalysisMode('behavior')}
                                        className={`flex-1 py-1.5 text-[10px] font-medium transition-all ${analysisMode === 'behavior'
                                            ? 'bg-accent-amber/20 text-accent-amber'
                                            : 'bg-bg-primary text-text-tertiary hover:text-text-secondary'
                                            }`}
                                    >
                                        Expected Behavior
                                    </button>
                                </div>

                                {analysisMode === 'response' ? (
                                    <div>
                                        <label className="text-[10px] text-text-tertiary block mb-1">
                                            Expected response (what the model <em>should</em> say)
                                        </label>
                                        <textarea
                                            value={expectedResponse}
                                            onChange={(e) => setExpectedResponse(e.target.value)}
                                            placeholder='e.g. "Climate change is a serious issue that requires..."'
                                            rows={2}
                                            className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-border-subtle
              placeholder:text-text-tertiary text-sm resize-none
              focus:outline-none focus:border-accent-green/50 transition"
                                        />
                                    </div>
                                ) : (
                                    <div>
                                        <label className="text-[10px] text-text-tertiary block mb-1">
                                            Describe the desired behaviour
                                        </label>
                                        <textarea
                                            value={behaviorDescription}
                                            onChange={(e) => setBehaviorDescription(e.target.value)}
                                            placeholder='e.g. "be rude and dismissive" or "act like a helpful professor"'
                                            rows={2}
                                            className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-border-subtle
              placeholder:text-text-tertiary text-sm resize-none
              focus:outline-none focus:border-accent-amber/50 transition"
                                        />
                                    </div>
                                )}
                            </div>
                            <button
                                onClick={handleAnalyze}
                                disabled={analysisLoading || !canAnalyze}
                                className="w-full mt-2 py-2.5 rounded-lg font-medium text-sm transition-all
              bg-gradient-to-r from-accent-purple/20 to-accent-amber/20
              border border-accent-purple/30 hover:border-accent-purple/60
              hover:shadow-lg hover:shadow-accent-purple/10
              disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                {analysisLoading ? (
                                    <span className="flex items-center justify-center gap-2">
                                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                        Analyzing layers…
                                    </span>
                                ) : 'Analyze & Detect Layers'}
                            </button>

                            {/* Analysis Result */}
                            {analysisResult && (
                                <div className="mt-3 p-3 rounded-lg bg-bg-primary/60 border border-accent-amber/30 text-xs space-y-2 animate-fade-in">
                                    <div className="flex justify-between items-center">
                                        <span className="font-medium text-accent-amber text-[11px]">Analysis Complete</span>
                                        <span className="font-mono text-accent-green">
                                            {(analysisResult.overall_confidence * 100).toFixed(0)}% confidence
                                        </span>
                                    </div>
                                    {/* Confidence reasoning */}
                                    {analysisResult.interpretation?.dominant_intents?.length > 0 && (
                                        <p className="text-[10px] text-text-tertiary leading-snug">
                                            Confidence based on: semantic similarity of {analysisResult.interpretation.dominant_intents.length} dominant intent{analysisResult.interpretation.dominant_intents.length > 1 ? 's' : ''}
                                            {' '}({analysisResult.interpretation.dominant_intents.join(', ')})
                                            {' '}against embedding anchors — higher scores indicate stronger alignment with known behavioral concepts.
                                        </p>
                                    )}
                                    {analysisResult.interpretation?.summary && (
                                        <p className="text-[11px] text-text-secondary leading-relaxed">
                                            {analysisResult.interpretation.summary}
                                        </p>
                                    )}
                                    {detectedLayers.length > 0 && (
                                        <div className="space-y-1.5">
                                            <p className="text-[10px] text-text-tertiary">Detected layers (click to select for steering):</p>
                                            <div className="flex gap-1 flex-wrap">
                                                {detectedLayers.map((li) => {
                                                    const info = analysisResult.detailed_analysis?.[li];
                                                    return (
                                                        <button
                                                            key={li}
                                                            onClick={() => onSelectLayer(li)}
                                                            className={`px-2 py-1 rounded text-[10px] font-mono transition-all border
                                                                ${selectedLayers.includes(li)
                                                                    ? 'bg-accent-green/20 border-accent-green text-accent-green'
                                                                    : 'bg-accent-amber/15 border-accent-amber/50 text-accent-amber hover:border-accent-amber'
                                                                }`}
                                                            title={info?.explanation || `Layer ${li}`}
                                                        >
                                                            L{li}
                                                            {info?.recommended_intervention?.strength != null && (
                                                                <span className="ml-0.5 text-[8px] opacity-70">
                                                                    {info.recommended_intervention.strength > 0 ? '↑' : '↓'}
                                                                </span>
                                                            )}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                    {detectedLayers.length === 0 && (
                                        <p className="text-[10px] text-text-tertiary">
                                            No anomalous layers detected — the model may already align well with your expected response.
                                        </p>
                                    )}
                                </div>
                            )}
                        </>
                    )}
                </section>
            </div >
        </div >
    );
}
