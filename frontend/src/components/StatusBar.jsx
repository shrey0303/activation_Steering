/**
 * Top header bar with logo, model status, and connection indicator.
 */
export default function StatusBar({ modelInfo, isConnected, metrics }) {
    return (
        <header className="h-12 px-4 flex items-center justify-between border-b border-border-subtle bg-bg-secondary/50 backdrop-blur-sm">
            {/* Left: Logo */}
            <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-green to-accent-blue flex items-center justify-center text-sm font-bold">
                        L
                    </div>
                    <span className="font-semibold text-sm tracking-tight">
                        Steer<span className="text-accent-green">Ops</span>
                    </span>
                </div>
                <div className="h-4 w-px bg-border-subtle" />
                <span className="text-[10px] text-text-tertiary font-mono">
                    v1.0.0 — Activation Debugger
                </span>
            </div>

            {/* Center: Model Info */}
            <div className="flex items-center gap-4">
                {modelInfo?.loaded ? (
                    <>
                        <div className="flex items-center gap-1.5">
                            <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse-subtle" />
                            <span className="text-xs text-text-secondary font-mono">
                                {modelInfo.name?.split('/').pop() || 'Unknown'}
                            </span>
                        </div>
                        <span className="text-[10px] text-text-tertiary">
                            {modelInfo.device?.toUpperCase()} · {modelInfo.quantized ? `${modelInfo.quantization_bits}bit` : 'FP'}
                        </span>
                    </>
                ) : (
                    <span className="text-xs text-text-tertiary">No model loaded</span>
                )}
            </div>

            {/* Right: Status */}
            <div className="flex items-center gap-3">
                {metrics && (
                    <span className="text-[10px] font-mono text-text-tertiary">
                        RAM {metrics.ram_percent?.toFixed(0)}%
                        {metrics.gpu_used_mb ? ` · GPU ${metrics.gpu_used_mb.toFixed(0)}MB` : ''}
                    </span>
                )}
                <div className="flex items-center gap-1.5">
                    <div
                        className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-accent-green' : 'bg-accent-red animate-pulse'
                            }`}
                    />
                    <span className="text-[10px] text-text-tertiary">
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                </div>
            </div>
        </header>
    );
}
