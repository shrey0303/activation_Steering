import { useCallback, useEffect, useRef, useState } from 'react';
import { WS_URL } from '../utils/constants';

/**
 * WebSocket hook with infinite auto-reconnect + keep-alive pings.
 */
export default function useWebSocket({ onToken, onDone, onError } = {}) {
    const [isConnected, setIsConnected] = useState(false);
    const wsRef = useRef(null);
    const retriesRef = useRef(0);
    const reconnectTimerRef = useRef(null);
    const pingTimerRef = useRef(null);
    const mountedRef = useRef(true);

    const MAX_RETRIES = 50; // Stop retrying after 50 attempts

    const connect = useCallback(() => {
        if (!mountedRef.current) return;
        if (wsRef.current?.readyState === WebSocket.OPEN) return;
        if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

        // Stop after MAX_RETRIES to prevent infinite reconnect loop
        if (retriesRef.current >= MAX_RETRIES) {
            console.warn(`WebSocket: gave up after ${MAX_RETRIES} retries`);
            onError?.({ type: 'error', message: 'Connection lost. Please refresh the page.' });
            return;
        }

        // Clear any pending reconnect
        if (reconnectTimerRef.current) {
            clearTimeout(reconnectTimerRef.current);
            reconnectTimerRef.current = null;
        }

        try {
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                setIsConnected(true);
                retriesRef.current = 0;

                // Start keep-alive pings every 25s
                if (pingTimerRef.current) clearInterval(pingTimerRef.current);
                pingTimerRef.current = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ping' }));
                    }
                }, 25000);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    switch (data.type) {
                        case 'token':
                            onToken?.(data);
                            break;
                        case 'done':
                            onDone?.(data);
                            break;
                        case 'error':
                            onError?.(data);
                            break;
                        case 'pong':
                        case 'stopped':
                            break;
                        default:
                            break;
                    }
                } catch (e) {
                    console.error('WS parse error:', e);
                }
            };

            ws.onclose = () => {
                setIsConnected(false);
                if (pingTimerRef.current) {
                    clearInterval(pingTimerRef.current);
                    pingTimerRef.current = null;
                }

                if (!mountedRef.current) return;

                // Exponential backoff: 3s → 6s → 12s → 30s max
                const delay = Math.min(3000 * Math.pow(2, retriesRef.current), 30000);
                retriesRef.current += 1;
                reconnectTimerRef.current = setTimeout(connect, delay);
            };

            ws.onerror = () => {
                // onclose will fire after this, handling reconnect
            };

            wsRef.current = ws;
        } catch {
            // Schedule retry if WebSocket constructor fails
            const delay = Math.min(3000 * Math.pow(2, retriesRef.current), 30000);
            retriesRef.current += 1;
            reconnectTimerRef.current = setTimeout(connect, delay);
        }
    }, [onToken, onDone, onError]);

    // Reconnect when tab becomes visible again
    useEffect(() => {
        const handleVisibility = () => {
            if (document.visibilityState === 'visible' && !isConnected) {
                retriesRef.current = 0;
                connect();
            }
        };
        document.addEventListener('visibilitychange', handleVisibility);
        return () => document.removeEventListener('visibilitychange', handleVisibility);
    }, [connect, isConnected]);

    useEffect(() => {
        mountedRef.current = true;
        connect();
        return () => {
            mountedRef.current = false;
            if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
            if (pingTimerRef.current) clearInterval(pingTimerRef.current);
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [connect]);

    const sendMessage = useCallback((msg) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(msg));
        }
    }, []);

    return { isConnected, sendMessage, connect };
}

