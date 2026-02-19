import { useState, useRef, useEffect } from 'react';

/**
 * Centre panel: chat-style input and streaming response output
 * with stop-generation button.
 */
export default function ChatInterface({
    messages,
    streamingText,
    isGenerating,
    onSend,
    onStop,
    steeringEnabled,
    selectedLayers,
    steeringStrength,
}) {
    const [input, setInput] = useState('');
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, streamingText]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() && !isGenerating) {
            onSend(input.trim());
            setInput('');
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="px-4 py-3 border-b border-border-subtle flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary">
                        Chat
                    </h2>
                    {steeringEnabled && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-mono
              bg-accent-green/15 text-accent-green border border-accent-green/30">
                            Steering Active · L{selectedLayers.join(',L')} · {steeringStrength > 0 ? '+' : ''}{steeringStrength.toFixed(1)}
                        </span>
                    )}
                </div>
                {isGenerating && (
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1.5 text-xs text-text-tertiary">
                            <div className="flex gap-0.5">
                                <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                            Generating…
                        </div>
                        {/* Stop Button */}
                        <button
                            onClick={onStop}
                            className="px-2.5 py-1 rounded-lg text-xs font-medium transition-all
                bg-accent-red/15 border border-accent-red/30
                text-accent-red hover:bg-accent-red/25 hover:border-accent-red/50
                hover:shadow-lg hover:shadow-accent-red/10
                animate-fade-in"
                            title="Stop generation"
                        >
                            ⏹ Stop
                        </button>
                    </div>
                )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && !streamingText && (
                    <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-40">
                        <div className="text-4xl opacity-30">S</div>
                        <div>
                            <p className="text-sm font-medium text-text-secondary">SteerOps Debugger</p>
                            <p className="text-xs text-text-tertiary mt-1">
                                Load a model → Scan layers → Send prompts to see activations<br />
                                Use analysis to find which layers to steer for specific behaviors
                            </p>
                        </div>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`animate-slide-in ${msg.role === 'user' ? 'flex justify-end'
                            : msg.role === 'system' ? 'flex justify-center' : ''
                            }`}
                    >
                        <div
                            className={`max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed ${msg.role === 'user'
                                ? 'bg-accent-green/10 border border-accent-green/20 text-text-primary'
                                : msg.role === 'system'
                                    ? 'bg-accent-amber/10 border border-accent-amber/20 text-accent-amber text-xs max-w-[70%]'
                                    : 'bg-bg-tertiary border border-border-subtle text-text-primary'
                                }`}
                        >
                            {/* System messages */}
                            {msg.role === 'system' && (
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                            )}

                            {/* Steered badge */}
                            {msg.role === 'assistant' && msg.steered && (
                                <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-border-subtle">
                                    <span className="text-[10px] font-mono text-accent-green">
                                        Steered
                                    </span>
                                    {msg.metrics && (
                                        <span className="text-[10px] text-text-tertiary font-mono ml-auto">
                                            {msg.metrics.tokens_generated} tok · {msg.metrics.latency_ms}ms ·
                                            {msg.metrics.tokens_per_sec} tok/s
                                        </span>
                                    )}
                                </div>
                            )}
                            {msg.role === 'assistant' && !msg.steered && msg.metrics && (
                                <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-border-subtle">
                                    <span className="text-[10px] text-text-tertiary font-mono">
                                        {msg.metrics.tokens_generated} tok · {msg.metrics.latency_ms}ms ·
                                        {msg.metrics.tokens_per_sec} tok/s
                                    </span>
                                </div>
                            )}
                            {/* Stopped badge */}
                            {msg.role === 'assistant' && msg.stopped && (
                                <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-border-subtle">
                                    <span className="text-[10px] font-mono text-accent-amber">
                                        ⏹ Generation stopped
                                    </span>
                                </div>
                            )}
                            {msg.role !== 'system' && (
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                            )}
                        </div>
                    </div>
                ))}

                {/* Streaming text */}
                {streamingText && (
                    <div className="animate-fade-in">
                        <div className="max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed bg-bg-tertiary border border-border-subtle">
                            <p className="whitespace-pre-wrap">
                                {streamingText}<span className="typing-cursor" />
                            </p>
                        </div>
                    </div>
                )}

                <div ref={bottomRef} />
            </div>

            {/* Input */}
            <form
                onSubmit={handleSubmit}
                className="p-4 border-t border-border-subtle"
            >
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={isGenerating ? 'Generating… (click Stop to cancel)' : 'Type a prompt…'}
                        disabled={isGenerating}
                        className="flex-1 px-4 py-2.5 rounded-lg bg-bg-primary border border-border-subtle
              placeholder:text-text-tertiary text-sm
              focus:outline-none focus:border-accent-green/50 transition
              disabled:opacity-50"
                    />
                    {isGenerating ? (
                        <button
                            type="button"
                            onClick={onStop}
                            className="px-5 py-2.5 rounded-lg font-medium text-sm
                bg-accent-red/20 text-accent-red border border-accent-red/30
                hover:bg-accent-red/30 hover:border-accent-red/50
                transition-all hover:shadow-lg hover:shadow-accent-red/20"
                        >
                            ⏹ Stop
                        </button>
                    ) : (
                        <button
                            type="submit"
                            disabled={!input.trim()}
                            className="px-5 py-2.5 rounded-lg font-medium text-sm
                bg-accent-green text-white hover:bg-accent-green-hover
                transition-all disabled:opacity-40 disabled:cursor-not-allowed
                hover:shadow-lg hover:shadow-accent-green/20"
                        >
                            Send
                        </button>
                    )}
                </div>
            </form>
        </div>
    );
}
