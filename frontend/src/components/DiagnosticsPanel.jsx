import { useState, useEffect, useRef } from 'react';

/**
 * Real-time Steering Diagnostics — shows per-token metrics during generation.
 * Displays: cosine similarity, effective strength, norm deviation,
 * gating status, cooldown status, and circuit breaker events.
 */
export default function DiagnosticsPanel({ metrics, isGenerating }) {
    const [history, setHistory] = useState([]);
    const [expanded, setExpanded] = useState(true);
    const scrollRef = useRef(null);

    // Accumulate per-generation metrics
    useEffect(() => {
        if (metrics?.steering_diagnostics) {
            setHistory(prev => [...prev, metrics.steering_diagnostics]);
        }
    }, [metrics]);

    // Clear on new generation
    useEffect(() => {
        if (isGenerating) setHistory([]);
    }, [isGenerating]);

    // Auto-scroll
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [history]);

    const lastDiag = metrics?.steering_diagnostics;
    const interventions = metrics?.active_interventions || [];
    const totalOverhead = interventions.reduce((sum, h) => sum + (h.overhead_ms || 0), 0);
    const entropy = metrics?.entropy;
    const circuitBroken = metrics?.circuit_breaker_triggered;

    // Sparkline data: last 50 effective strengths
    const strengthHistory = history.slice(-50).map(d => d?.effective_strength ?? 0);
    const maxStr = Math.max(...strengthHistory, 0.1);

    return (
        <div className="flex flex-col border-t border-border-subtle"
            style={{ resize: 'vertical', overflow: 'auto', minHeight: '60px', maxHeight: '600px' }}>
            {/* Header */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="px-4 py-2 flex items-center justify-between
                    hover:bg-bg-tertiary/50 transition"
            >
                <h3 className="text-[11px] font-semibold text-text-primary flex items-center gap-2">
                    <span className={`w-1.5 h-1.5 rounded-full ${isGenerating ? 'bg-green-400 animate-pulse' : 'bg-text-muted/40'}`} />
                    Steering Diagnostics
                </h3>
                <div className="flex items-center gap-2">
                    {totalOverhead > 0 && (
                        <span className="text-[9px] font-mono text-text-muted">
                            +{totalOverhead.toFixed(1)}ms
                        </span>
                    )}
                    <span className={`text-text-muted text-[10px] transition-transform ${expanded ? 'rotate-180' : ''}`}>▾</span>
                </div>
            </button>

            {expanded && (
                <div className="px-4 pb-3" style={{ minHeight: '180px', maxHeight: '400px', overflowY: 'auto' }}>
                    {/* Active Hooks Summary */}
                    {interventions.length > 0 ? (
                        <div className="space-y-2">
                            {/* Hook cards */}
                            {interventions.map((hook, i) => (
                                <div key={i} className="rounded-md bg-bg-tertiary/50 border border-border-subtle p-2">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-[10px] font-mono text-accent-blue">
                                            L{hook.layer}
                                        </span>
                                        <div className="flex items-center gap-2">
                                            {hook.fired && (
                                                <span className="text-[8px] px-1 py-0.5 rounded bg-green-500/15 text-green-400 font-medium">
                                                    FIRED
                                                </span>
                                            )}
                                            {hook.cooldown_remaining > 0 && (
                                                <span className="text-[8px] px-1 py-0.5 rounded bg-amber-500/15 text-amber-400 font-medium">
                                                    COOLDOWN {hook.cooldown_remaining}
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    {/* Mini stats */}
                                    <div className="grid grid-cols-3 gap-1 text-[9px]">
                                        <div>
                                            <span className="text-text-muted">Strength</span>
                                            <div className="text-text-primary font-mono">{hook.strength?.toFixed(2) ?? '—'}</div>
                                        </div>
                                        <div>
                                            <span className="text-text-muted">Tokens</span>
                                            <div className="text-text-primary font-mono">{hook.token_count ?? 0}</div>
                                        </div>
                                        <div>
                                            <span className="text-text-muted">Overhead</span>
                                            <div className="text-text-primary font-mono">{hook.overhead_ms?.toFixed(1) ?? '0'}ms</div>
                                        </div>
                                    </div>

                                    {/* Gate threshold bar */}
                                    <div className="mt-1.5">
                                        <div className="flex justify-between text-[8px] text-text-muted mb-0.5">
                                            <span>Gate: {hook.gate_threshold?.toFixed(3) ?? '—'}</span>
                                        </div>
                                        <div className="h-1 rounded-full bg-bg-primary overflow-hidden">
                                            <div
                                                className="h-full rounded-full bg-accent-purple/60 transition-all duration-300"
                                                style={{ width: `${Math.min((hook.gate_threshold || 0) * 100, 100)}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            ))}

                            {/* Entropy & Circuit Breaker */}
                            {entropy !== undefined && (
                                <div className={`flex items-center justify-between px-2 py-1.5 rounded-md text-[10px]
                                    ${circuitBroken
                                        ? 'bg-red-500/10 border border-red-500/30'
                                        : 'bg-bg-tertiary/30 border border-border-subtle'
                                    }`}
                                >
                                    <span className="text-text-muted">Entropy</span>
                                    <span className={`font-mono ${entropy > 6 ? 'text-red-400' : entropy > 4 ? 'text-amber-400' : 'text-green-400'}`}>
                                        {entropy?.toFixed(2)} nats
                                    </span>
                                    {circuitBroken && (
                                        <span className="text-[8px] px-1 py-0.5 rounded bg-red-500/20 text-red-400 font-medium animate-pulse">
                                            CIRCUIT BREAKER
                                        </span>
                                    )}
                                </div>
                            )}

                            {/* Strength sparkline */}
                            {strengthHistory.length > 1 && (
                                <div className="mt-1">
                                    <span className="text-[9px] text-text-muted mb-1 block">Effective Strength Timeline</span>
                                    <div className="flex items-end gap-px h-6">
                                        {strengthHistory.map((s, i) => (
                                            <div
                                                key={i}
                                                className="flex-1 bg-accent-purple/40 rounded-t-sm transition-all duration-75"
                                                style={{ height: `${Math.max(1, (s / maxStr) * 100)}%` }}
                                                title={`Token ${i}: ${s.toFixed(3)}`}
                                            />
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-[10px] text-text-muted/60 text-center py-4 space-y-2">
                            {entropy !== undefined ? (
                                <div className="space-y-2">
                                    <div className="flex items-center justify-center gap-2">
                                        <span className="text-text-muted">Output Entropy:</span>
                                        <span className={`font-mono font-medium ${entropy > 6 ? 'text-red-400' : entropy > 4 ? 'text-amber-400' : 'text-green-400'}`}>
                                            {entropy?.toFixed(2)} nats
                                        </span>
                                    </div>
                                    <span className="text-text-muted/40">No steering hooks active — showing baseline entropy</span>
                                </div>
                            ) : (
                                <div>
                                    <p className="font-medium text-text-muted/80">No active steering hooks</p>
                                    <p className="mt-1">1. Go to Behavior Analysis → type a behavior</p>
                                    <p>2. Click "Analyze & Detect Layers"</p>
                                    <p>3. Send a prompt in chat to see live diagnostics</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
