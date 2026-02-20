import { useState, useRef, useMemo } from 'react';
import { analyzePrompt, scanModel } from '../utils/api';

/**
 * Hook for managing analysis state with dual-mode support.
 * Maintains separate scan/analysis results for local and remote modes
 * so toggling between modes preserves each mode's data.
 */
export default function useAnalysis() {
    // ── Per-mode state storage ──────────────────────────────
    const [localScanResult, setLocalScanResult] = useState(null);
    const [localAnalysisResult, setLocalAnalysisResult] = useState(null);
    const [remoteScanResult, setRemoteScanResult] = useState(null);
    const [remoteAnalysisResult, setRemoteAnalysisResult] = useState(null);

    const [loading, setLoading] = useState(false);
    const [scanLoading, setScanLoading] = useState(false);
    const [error, setError] = useState(null);

    // Track which mode is active — set by App.jsx via setMode()
    const [mode, setMode] = useState('local');

    // ── Derived values: return current mode's data ──────────
    const scanResult = useMemo(
        () => (mode === 'remote' ? remoteScanResult : localScanResult),
        [mode, remoteScanResult, localScanResult]
    );

    const analysisResult = useMemo(
        () => (mode === 'remote' ? remoteAnalysisResult : localAnalysisResult),
        [mode, remoteAnalysisResult, localAnalysisResult]
    );

    // ── Scan (local model only — remote uses remoteScan API) ─
    const scan = async (forceRescan = false) => {
        setScanLoading(true);
        setError(null);
        try {
            const result = await scanModel(forceRescan);
            setLocalScanResult(result);
            return result;
        } catch (e) {
            setError(e.message);
            return null;
        } finally {
            setScanLoading(false);
        }
    };

    // ── Unified setter (writes to current mode) ─────────────
    const setScanResult = (result) => {
        if (mode === 'remote') {
            setRemoteScanResult(result);
        } else {
            setLocalScanResult(result);
        }
    };

    // ── Analyze ─────────────────────────────────────────────
    const analyze = async (prompt, expectedResponse, behaviorDescription = null) => {
        setLoading(true);
        setError(null);
        try {
            const result = await analyzePrompt(prompt, expectedResponse, behaviorDescription);
            if (mode === 'remote') {
                setRemoteAnalysisResult(result);
            } else {
                setLocalAnalysisResult(result);
            }
            return result;
        } catch (e) {
            setError(e.message);
            return null;
        } finally {
            setLoading(false);
        }
    };

    // ── Reset: clears data for the specified or current mode ─
    const reset = (targetMode = null) => {
        const m = targetMode || mode;
        if (m === 'remote') {
            setRemoteScanResult(null);
            setRemoteAnalysisResult(null);
        } else {
            setLocalScanResult(null);
            setLocalAnalysisResult(null);
        }
        setError(null);
    };

    // ── Reset all modes ─────────────────────────────────────
    const resetAll = () => {
        setLocalScanResult(null);
        setLocalAnalysisResult(null);
        setRemoteScanResult(null);
        setRemoteAnalysisResult(null);
        setError(null);
    };

    return {
        // Mode control
        setMode,
        // Current mode's data (reactive — changes on mode switch or data update)
        scanResult,
        analysisResult,
        // Setters
        setScanResult,
        // Actions
        scan,
        scanLoading,
        analyze,
        loading,
        error,
        reset,
        resetAll,
    };
}
