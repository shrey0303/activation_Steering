import { useState, useEffect } from 'react';

/**
 * Persistent loading banner shown at top of the app while model is downloading.
 * Non-blocking — user can interact with everything else.
 */
export default function LoadingBanner({ modelName, onCancel }) {
    const [elapsed, setElapsed] = useState(0);
    const [dots, setDots] = useState('');

    useEffect(() => {
        const timer = setInterval(() => {
            setElapsed((prev) => prev + 1);
            setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
        }, 1000);
        return () => clearInterval(timer);
    }, []);

    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    const timeStr = minutes > 0
        ? `${minutes}m ${seconds}s`
        : `${seconds}s`;

    return (
        <div className="bg-gradient-to-r from-accent-blue/10 via-accent-purple/10 to-accent-blue/10
            border-b border-accent-blue/20 px-4 py-2 flex items-center gap-3 animate-fade-in">
            {/* Spinning loader */}
            <div className="relative w-5 h-5 flex-shrink-0">
                <div className="absolute inset-0 rounded-full border-2 border-accent-blue/20" />
                <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-accent-blue
                    animate-spin" />
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-text-primary truncate">
                    Downloading <span className="text-accent-blue">{modelName}</span>{dots}
                </p>
                <p className="text-[10px] text-text-tertiary">
                    {elapsed < 10
                        ? 'Starting download…'
                        : `Loading for ${timeStr} — you can use Remote mode while this downloads`
                    }
                </p>
            </div>

            {/* Progress bar (indeterminate) */}
            <div className="w-24 h-1 rounded-full bg-bg-primary overflow-hidden flex-shrink-0">
                <div className="h-full bg-gradient-to-r from-accent-blue to-accent-purple rounded-full
                    animate-progress" style={{
                        animation: 'progress 2s ease-in-out infinite',
                    }} />
            </div>

            {/* Timer */}
            <span className="text-[10px] font-mono text-text-tertiary flex-shrink-0">
                {timeStr}
            </span>
        </div>
    );
}
