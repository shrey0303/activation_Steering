import { useState, useEffect, useCallback } from 'react';

/**
 * Toast notification system for non-blocking alerts.
 * Shows temporary messages that auto-dismiss.
 */

let _toastId = 0;

export function useToast() {
    const [toasts, setToasts] = useState([]);

    const addToast = useCallback((message, type = 'info', duration = 5000) => {
        const id = ++_toastId;
        setToasts((prev) => [...prev, { id, message, type, duration }]);
        if (duration > 0) {
            setTimeout(() => {
                setToasts((prev) => prev.filter((t) => t.id !== id));
            }, duration);
        }
        return id;
    }, []);

    const removeToast = useCallback((id) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    }, []);

    return { toasts, addToast, removeToast };
}

const typeStyles = {
    info: 'bg-accent-blue/15 border-accent-blue/40 text-accent-blue',
    success: 'bg-accent-green/15 border-accent-green/40 text-accent-green',
    error: 'bg-accent-red/15 border-accent-red/40 text-accent-red',
    warning: 'bg-accent-amber/15 border-accent-amber/40 text-accent-amber',
};

const typeIcons = {
    info: 'i',
    success: '✓',
    error: '✕',
    warning: '!',
};

export default function ToastContainer({ toasts, onRemove }) {
    if (!toasts.length) return null;

    return (
        <div className="fixed top-14 right-4 z-50 space-y-2 max-w-sm">
            {toasts.map((toast) => (
                <div
                    key={toast.id}
                    className={`flex items-start gap-2 px-4 py-3 rounded-lg border backdrop-blur-sm
                        shadow-lg animate-slide-in text-sm ${typeStyles[toast.type] || typeStyles.info}`}
                >
                    <span className="text-base flex-shrink-0 font-bold">{typeIcons[toast.type] || '·'}</span>
                    <p className="flex-1 text-xs leading-relaxed">{toast.message}</p>
                    <button
                        onClick={() => onRemove(toast.id)}
                        className="text-xs opacity-60 hover:opacity-100 transition flex-shrink-0 ml-1"
                    >
                        ×
                    </button>
                </div>
            ))}
        </div>
    );
}
