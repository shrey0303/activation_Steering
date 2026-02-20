import { useState, useEffect, useMemo } from 'react';
import { getFeatures, updateFeatureLabel, extractFeatures } from '../utils/api';

/**
 * Feature Discovery Browser — browse, search, select, and relabel
 * PCA features from the offline feature dictionary.
 */
export default function FeatureBrowser({
    onSelectFeature,
    selectedFeatures = [],
    modelInfo,
}) {
    const [features, setFeatures] = useState([]);
    const [search, setSearch] = useState('');
    const [groupBy, setGroupBy] = useState('layer'); // 'layer' | 'label'
    const [editingId, setEditingId] = useState(null);
    const [editLabel, setEditLabel] = useState('');
    const [loading, setLoading] = useState(false);
    const [extracting, setExtracting] = useState(false);
    const [error, setError] = useState(null);
    const [expandedLayers, setExpandedLayers] = useState(new Set());

    // Fetch features on mount or model change
    useEffect(() => {
        if (!modelInfo?.model_name) return;
        fetchFeatures();
    }, [modelInfo?.model_name]);

    const fetchFeatures = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getFeatures();
            setFeatures(data.features || []);
        } catch (e) {
            // Features not extracted yet — not an error
            setFeatures([]);
        } finally {
            setLoading(false);
        }
    };

    const handleExtract = async () => {
        setExtracting(true);
        setError(null);
        try {
            await extractFeatures();
            await fetchFeatures();
        } catch (e) {
            setError(e.message);
        } finally {
            setExtracting(false);
        }
    };

    const handleRelabel = async (featureId) => {
        try {
            await updateFeatureLabel(featureId, editLabel);
            setFeatures(prev =>
                prev.map(f =>
                    f.feature_id === featureId ? { ...f, label: editLabel } : f
                )
            );
            setEditingId(null);
            setEditLabel('');
        } catch (e) {
            setError(e.message);
        }
    };

    // Filter and group
    const filtered = useMemo(() => {
        if (!search) return features;
        const q = search.toLowerCase();
        return features.filter(f =>
            f.label?.toLowerCase().includes(q) ||
            f.feature_id?.toLowerCase().includes(q)
        );
    }, [features, search]);

    const grouped = useMemo(() => {
        const map = {};
        filtered.forEach(f => {
            const key = groupBy === 'layer' ? `Layer ${f.layer_idx}` : (f.label || 'Unlabeled');
            if (!map[key]) map[key] = [];
            map[key].push(f);
        });
        return Object.entries(map).sort((a, b) => {
            if (groupBy === 'layer') {
                return parseInt(a[0].replace('Layer ', '')) - parseInt(b[0].replace('Layer ', ''));
            }
            return a[0].localeCompare(b[0]);
        });
    }, [filtered, groupBy]);

    const isSelected = (fid) => selectedFeatures.includes(fid);
    const labeledCount = features.filter(f => f.label && !f.label.startsWith('L')).length;

    const toggleLayer = (key) => {
        setExpandedLayers(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    // ── Render ─────────────────────────────────────────────────
    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="px-4 py-3 border-b border-border-subtle">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                        <span className="text-accent-purple">◈</span>
                        Feature Dictionary
                    </h3>
                    <span className="text-[10px] text-text-muted px-1.5 py-0.5 rounded bg-bg-tertiary">
                        {features.length} features • {labeledCount} labeled
                    </span>
                </div>

                {/* Search */}
                <div className="relative">
                    <input
                        type="text"
                        placeholder="Search features..."
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        className="w-full pl-7 pr-3 py-1.5 text-xs rounded-md
                            bg-bg-tertiary border border-border-subtle
                            text-text-primary placeholder:text-text-muted
                            focus:border-accent-purple/50 focus:outline-none focus:ring-1 focus:ring-accent-purple/20
                            transition"
                    />
                    <svg className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                </div>

                {/* Controls */}
                <div className="flex items-center gap-2 mt-2">
                    <div className="flex rounded-md overflow-hidden border border-border-subtle text-[10px]">
                        <button
                            onClick={() => setGroupBy('layer')}
                            className={`px-2 py-0.5 transition ${groupBy === 'layer' ? 'bg-accent-purple/20 text-accent-purple' : 'text-text-muted hover:text-text-secondary'}`}
                        >
                            By Layer
                        </button>
                        <button
                            onClick={() => setGroupBy('label')}
                            className={`px-2 py-0.5 border-l border-border-subtle transition ${groupBy === 'label' ? 'bg-accent-purple/20 text-accent-purple' : 'text-text-muted hover:text-text-secondary'}`}
                        >
                            By Label
                        </button>
                    </div>

                    {features.length === 0 && modelInfo && (
                        <button
                            onClick={handleExtract}
                            disabled={extracting}
                            className="ml-auto px-2.5 py-0.5 text-[10px] rounded-md
                                bg-accent-purple/20 text-accent-purple border border-accent-purple/30
                                hover:border-accent-purple/60 disabled:opacity-50
                                transition"
                        >
                            {extracting ? 'Extracting...' : 'Extract Features'}
                        </button>
                    )}
                </div>
            </div>

            {/* Error */}
            {error && (
                <div className="mx-4 mt-2 px-2 py-1 text-[10px] text-red-400 bg-red-500/10 rounded border border-red-500/20">
                    {error}
                </div>
            )}

            {/* Feature List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
                {loading ? (
                    <div className="flex items-center justify-center h-32 text-text-muted text-xs">
                        <div className="animate-spin w-4 h-4 border-2 border-accent-purple/30 border-t-accent-purple rounded-full mr-2" />
                        Loading features...
                    </div>
                ) : features.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-32 text-text-muted text-xs px-4 text-center">
                        <p className="mb-2">No features extracted yet</p>
                        <p className="text-[10px] text-text-muted/60">
                            Load a model and click "Extract Features" to run
                            offline PCA analysis
                        </p>
                    </div>
                ) : (
                    <div className="py-1">
                        {grouped.map(([group, items]) => (
                            <div key={group} className="mb-0.5">
                                {/* Group Header */}
                                <button
                                    onClick={() => toggleLayer(group)}
                                    className="w-full px-4 py-1.5 flex items-center justify-between
                                        text-[11px] font-medium text-text-secondary
                                        hover:bg-bg-tertiary/50 transition"
                                >
                                    <span className="flex items-center gap-1.5">
                                        <span className={`transition-transform ${expandedLayers.has(group) ? 'rotate-90' : ''}`}>▸</span>
                                        {group}
                                    </span>
                                    <span className="text-[9px] text-text-muted font-normal">
                                        {items.length}
                                    </span>
                                </button>

                                {/* Feature Items */}
                                {expandedLayers.has(group) && items.map(f => (
                                    <div
                                        key={f.feature_id}
                                        className={`mx-2 mb-0.5 px-2.5 py-1.5 rounded-md text-[11px] cursor-pointer
                                            transition-all duration-150
                                            ${isSelected(f.feature_id)
                                                ? 'bg-accent-purple/15 border border-accent-purple/40 shadow-sm shadow-accent-purple/10'
                                                : 'bg-bg-tertiary/30 border border-transparent hover:bg-bg-tertiary hover:border-border-subtle'
                                            }`}
                                        onClick={() => onSelectFeature?.(f.feature_id, f)}
                                    >
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2 min-w-0">
                                                <span className="text-[9px] font-mono text-text-muted shrink-0">
                                                    {f.feature_id}
                                                </span>
                                                {editingId === f.feature_id ? (
                                                    <input
                                                        type="text"
                                                        value={editLabel}
                                                        onChange={e => setEditLabel(e.target.value)}
                                                        onKeyDown={e => {
                                                            if (e.key === 'Enter') handleRelabel(f.feature_id);
                                                            if (e.key === 'Escape') setEditingId(null);
                                                        }}
                                                        onBlur={() => handleRelabel(f.feature_id)}
                                                        onClick={e => e.stopPropagation()}
                                                        autoFocus
                                                        className="flex-1 px-1 py-0.5 text-[10px] rounded
                                                            bg-bg-primary border border-accent-purple/40
                                                            text-text-primary focus:outline-none"
                                                    />
                                                ) : (
                                                    <span
                                                        className={`truncate ${f.label && !f.label.startsWith('L') ? 'text-text-primary' : 'text-text-muted italic'}`}
                                                        onDoubleClick={(e) => {
                                                            e.stopPropagation();
                                                            setEditingId(f.feature_id);
                                                            setEditLabel(f.label || '');
                                                        }}
                                                    >
                                                        {f.label && !f.label.startsWith('L') ? f.label : 'unlabeled'}
                                                    </span>
                                                )}
                                            </div>

                                            {/* Variance badge */}
                                            <span className="text-[8px] font-mono text-text-muted shrink-0 ml-1">
                                                {(f.variance_explained * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Selection summary */}
            {selectedFeatures.length > 0 && (
                <div className="px-4 py-2 border-t border-border-subtle bg-accent-purple/5">
                    <div className="flex items-center justify-between text-[10px]">
                        <span className="text-accent-purple font-medium">
                            {selectedFeatures.length} feature{selectedFeatures.length > 1 ? 's' : ''} selected
                        </span>
                        <button
                            onClick={() => selectedFeatures.forEach(fid => onSelectFeature?.(fid))}
                            className="text-text-muted hover:text-text-secondary transition"
                        >
                            Clear
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
