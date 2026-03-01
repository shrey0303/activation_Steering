// Fix: prevent heatmap render with stale scan data
// Fix: prevent heatmap render with stale scan data
// Fix: prevent heatmap render with stale scan data
// Fix: prevent heatmap render with stale scan data
import { useEffect, useRef, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { CATEGORY_COLORS, CATEGORY_LABELS } from '../utils/constants';

/**
 * Right panel: Interactive D3.js heatmap showing per-layer activation
 * magnitudes with detailed layer info panel and analysis overlay.
 */

const CATEGORY_DESCRIPTIONS = {
    embedding: 'Converts tokens to vectors. Captures raw input features.',
    syntactic: 'Handles grammar, sentence structure, and word relationships.',
    semantic: 'Processes meaning, context, and factual knowledge.',
    reasoning: 'Performs logic, inference, and complex thought chains.',
    integration: 'Combines and routes information between layer groups.',
    output: 'Shapes final predictions and token probabilities.',
    unknown: 'Role not yet determined â€” run analysis for classification.',
};

export default function ActivationHeatmap({
    activations,
    scanResult,
    selectedLayers,
    analysisResult,
    onSelectLayer,
}) {
    const svgRef = useRef(null);
    const containerRef = useRef(null);
    const [hoveredLayer, setHoveredLayer] = useState(null);
    const [detailLayer, setDetailLayer] = useState(null);

    const layers = scanResult?.layer_profiles || [];
    const numLayers = scanResult?.num_layers || 0;

    const detectedLayers = analysisResult?.detected_layers || [];
    const detailedAnalysis = analysisResult?.detailed_analysis || {};

    // Merge activation values with scan data + analysis data
    // NOTE: selectedLayers excluded from data memo to avoid full D3 re-render on click
    const data = useMemo(() => {
        return layers.map((l) => {
            const analysisInfo = detailedAnalysis[l.layer_index] || null;
            return {
                ...l,
                activation: activations?.[l.layer_index] ?? 0,
                isDetected: detectedLayers.includes(l.layer_index),
                analysisInfo,
            };
        });
    }, [layers, activations, detectedLayers, detailedAnalysis]);

    // D3 rendering
    useEffect(() => {
        if (!svgRef.current || data.length === 0) return;

        const container = containerRef.current;
        const width = container.clientWidth - 8;
        const barHeight = 20;
        const height = Math.max(data.length * (barHeight + 4) + 60, 200);

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();
        svg.attr('width', width).attr('height', height);

        const margin = { top: 8, right: 80, bottom: 8, left: 42 };
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        // Scales
        const yScale = d3
            .scaleBand()
            .domain(data.map((d) => d.layer_index))
            .range([0, innerH])
            .padding(0.12);

        // Normalize activations for color/width
        const maxAct = Math.max(...data.map((d) => d.activation), 0.01);
        const normalise = (v) => v / maxAct;

        const colorScale = d3
            .scaleSequential(d3.interpolateYlOrRd)
            .domain([0, 1]);

        const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);

        // Background track bars
        g.selectAll('.track')
            .data(data)
            .join('rect')
            .attr('x', 0)
            .attr('y', (d) => yScale(d.layer_index))
            .attr('width', innerW)
            .attr('height', yScale.bandwidth())
            .attr('rx', 3)
            .attr('fill', '#1a1a2e')
            .attr('opacity', 0.4);

        // Activation bars
        const bars = g.selectAll('.heatmap-bar')
            .data(data)
            .join('rect')
            .attr('class', 'heatmap-cell')
            .attr('x', 0)
            .attr('y', (d) => yScale(d.layer_index))
            .attr('width', (d) => xScale(normalise(d.activation)))
            .attr('height', yScale.bandwidth())
            .attr('rx', 3)
            .attr('fill', (d) => {
                if (d.selected) return '#10b981';
                if (d.isDetected) return '#f59e0b';
                return colorScale(normalise(d.activation));
            })
            .attr('stroke', (d) => {
                if (d.selected) return '#10b981';
                if (d.isDetected) return '#f59e0b';
                return 'transparent';
            })
            .attr('stroke-width', (d) => (d.selected || d.isDetected ? 1.5 : 0))
            .attr('opacity', (d) => (d.selected ? 1 : d.isDetected ? 0.9 : 0.8))
            .style('cursor', 'pointer')
            .on('click', (event, d) => onSelectLayer(d.layer_index))
            .on('mouseenter', (event, d) => setHoveredLayer(d))
            .on('mouseleave', () => setHoveredLayer(null))
            .on('dblclick', (event, d) => {
                event.stopPropagation();
                setDetailLayer(d);
            });

        // Steering direction indicators for selected layers
        g.selectAll('.steer-arrow')
            .data(data.filter(d => d.selected))
            .join('text')
            .attr('x', (d) => xScale(normalise(d.activation)) + 4)
            .attr('y', (d) => yScale(d.layer_index) + yScale.bandwidth() / 2)
            .attr('dominant-baseline', 'middle')
            .attr('font-size', '11px')
            .text('â—†')
            .attr('fill', '#10b981');

        // Y-axis labels (layer index)
        g.selectAll('.layer-label')
            .data(data)
            .join('text')
            .attr('x', -6)
            .attr('y', (d) => yScale(d.layer_index) + yScale.bandwidth() / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('fill', (d) => {
                if (d.selected) return '#10b981';
                if (d.isDetected) return '#f59e0b';
                return '#6b7280';
            })
            .attr('font-size', '9px')
            .attr('font-family', 'JetBrains Mono, monospace')
            .attr('font-weight', (d) => d.selected || d.isDetected ? '700' : '400')
            .text((d) => `L${d.layer_index}`)
            .style('cursor', 'pointer')
            .on('click', (event, d) => onSelectLayer(d.layer_index));

        // Category label on right
        g.selectAll('.cat-label')
            .data(data)
            .join('text')
            .attr('x', innerW + 6)
            .attr('y', (d) => yScale(d.layer_index) + yScale.bandwidth() / 2)
            .attr('dominant-baseline', 'middle')
            .attr('fill', (d) => CATEGORY_COLORS[d.category] || '#6b7280')
            .attr('font-size', '8px')
            .attr('font-weight', '600')
            .text((d) => (d.category || 'UNK').slice(0, 4).toUpperCase());

        // Activation percentage on bar (if wide enough)
        g.selectAll('.act-value')
            .data(data)
            .join('text')
            .attr('x', (d) => Math.max(xScale(normalise(d.activation)) - 4, 18))
            .attr('y', (d) => yScale(d.layer_index) + yScale.bandwidth() / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('fill', '#ffffff')
            .attr('font-size', '8px')
            .attr('font-family', 'JetBrains Mono, monospace')
            .attr('opacity', 0.9)
            .text((d) => {
                const norm = normalise(d.activation);
                return norm > 0.08 ? `${(norm * 100).toFixed(0)}%` : '';
            });

    }, [data, onSelectLayer]);

    // Separate useEffect for selection styling â€” avoids full D3 re-render on click
    useEffect(() => {
        if (!svgRef.current || data.length === 0) return;
        const svg = d3.select(svgRef.current);
        const maxAct = Math.max(...data.map((d) => d.activation), 0.01);
        const normalise = (v) => v / maxAct;
        const colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);

        svg.selectAll('.heatmap-cell')
            .attr('fill', (d) => {
                if (selectedLayers.includes(d.layer_index)) return '#10b981';
                if (d.isDetected) return '#f59e0b';
                return colorScale(normalise(d.activation));
            })
            .attr('stroke', (d) => {
                if (selectedLayers.includes(d.layer_index)) return '#10b981';
                if (d.isDetected) return '#f59e0b';
                return 'transparent';
            })
            .attr('stroke-width', (d) => (selectedLayers.includes(d.layer_index) || d.isDetected ? 1.5 : 0))
            .attr('opacity', (d) => (selectedLayers.includes(d.layer_index) ? 1 : d.isDetected ? 0.9 : 0.8));

        svg.selectAll('.layer-label')
            .attr('fill', (d) => {
                if (selectedLayers.includes(d.layer_index)) return '#10b981';
                if (d.isDetected) return '#f59e0b';
                return '#6b7280';
            })
            .attr('font-weight', (d) => selectedLayers.includes(d.layer_index) || d.isDetected ? '700' : '400');
    }, [selectedLayers, data]);

    // â”€â”€ Empty state â”€â”€
    if (numLayers === 0) {
        return (
            <div className="flex flex-col h-full">
                <div className="px-4 py-3 border-b border-border-subtle">
                    <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary">
                        Layer Activations
                    </h2>
                </div>
                <div className="flex-1 flex items-center justify-center text-center opacity-40 p-4">
                    <div>
                        <div className="text-3xl mb-3 opacity-30 font-bold">||</div>
                        <p className="text-xs text-text-tertiary">
                            Scan the model to see layers.<br />
                            Send a prompt to visualize activations.
                        </p>
                        <div className="mt-4 p-3 rounded-lg bg-bg-tertiary/50 text-[10px] text-text-tertiary text-left space-y-1">
                            <div><strong className="text-text-secondary">1.</strong> Load a model from the left panel</div>
                            <div><strong className="text-text-secondary">2.</strong> Click "Scan Model" to detect layer roles</div>
                            <div><strong className="text-text-secondary">3.</strong> Send a prompt in chat â€” heatmap updates live</div>
                            <div><strong className="text-text-secondary">4.</strong> Double-click any bar for layer details</div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="px-4 py-3 border-b border-border-subtle">
                <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary">
                        Layer Activations
                    </h2>
                    <span className="text-[10px] font-mono text-text-tertiary">
                        {numLayers} layers
                    </span>
                </div>
                {/* Legend */}
                <div className="flex gap-3 mt-2 flex-wrap">
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-sm bg-[#10b981]" />
                        <span className="text-[9px] text-text-tertiary">Selected</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-sm bg-[#f59e0b]" />
                        <span className="text-[9px] text-text-tertiary">Detected</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-sm bg-gradient-to-r from-[#fef08a] to-[#dc2626]" />
                        <span className="text-[9px] text-text-tertiary">Activation</span>
                    </div>
                </div>
            </div>

            {/* Hover tooltip */}
            {hoveredLayer && (
                <div className="mx-2 mt-2 p-2.5 rounded-lg bg-bg-tertiary border border-border-subtle
                    text-xs space-y-1 animate-fade-in z-10">
                    <div className="flex justify-between items-center">
                        <span className="font-mono font-bold text-text-primary">
                            Layer {hoveredLayer.layer_index}
                        </span>
                        <span
                            className="px-1.5 py-0.5 rounded text-[9px] font-medium"
                            style={{
                                color: CATEGORY_COLORS[hoveredLayer.category] || '#6b7280',
                                backgroundColor: `${CATEGORY_COLORS[hoveredLayer.category] || '#6b7280'}20`,
                            }}
                        >
                            {CATEGORY_LABELS[hoveredLayer.category] || hoveredLayer.category}
                        </span>
                    </div>
                    <p className="text-[10px] text-text-tertiary leading-relaxed">
                        {CATEGORY_DESCRIPTIONS[hoveredLayer.category] || ''}
                    </p>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 mt-1 text-[10px]">
                        <span className="text-text-tertiary">Activation</span>
                        <span className="font-mono text-right">{(hoveredLayer.activation * 100).toFixed(1)}%</span>
                        <span className="text-text-tertiary">Confidence</span>
                        <span className="font-mono text-right">{(hoveredLayer.confidence * 100).toFixed(0)}%</span>
                        {hoveredLayer.behavioral_role && (
                            <>
                                <span className="text-text-tertiary">Role</span>
                                <span className="font-mono text-right truncate">{hoveredLayer.behavioral_role}</span>
                            </>
                        )}
                    </div>
                    {hoveredLayer.isDetected && hoveredLayer.analysisInfo && (
                        <div className="mt-1 pt-1 border-t border-border-subtle">
                            <div className="flex items-center gap-1 mb-0.5">
                                <span className="text-accent-amber text-[9px]">â— Analysis detected</span>
                            </div>
                            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
                                <span className="text-text-tertiary">Anomaly</span>
                                <span className="font-mono text-right text-accent-amber">
                                    {(hoveredLayer.analysisInfo.anomaly_score * 100).toFixed(0)}%
                                </span>
                                {hoveredLayer.analysisInfo.recommended_intervention?.strength != null && (
                                    <>
                                        <span className="text-text-tertiary">Suggested</span>
                                        <span className="font-mono text-right text-accent-green">
                                            {hoveredLayer.analysisInfo.recommended_intervention.strength > 0 ? '+' : ''}
                                            {hoveredLayer.analysisInfo.recommended_intervention.strength.toFixed(1)}
                                        </span>
                                    </>
                                )}
                            </div>
                        </div>
                    )}
                    <p className="text-[9px] text-text-tertiary/50 mt-1">Click to select Â· Double-click for details</p>
                </div>
            )}

            {/* Heatmap SVG + Detail Overlay */}
            <div className="flex-1 overflow-y-auto relative">
                <div ref={containerRef} className="p-1">
                    <svg ref={svgRef} />
                </div>

                {/* Detail panel (on double-click) â€” overlays heatmap */}
                {detailLayer && (
                    <>
                        {/* Backdrop to dismiss */}
                        <div
                            className="absolute inset-0 bg-black/20 z-20"
                            onClick={() => setDetailLayer(null)}
                        />
                        <div className="absolute top-4 left-2 right-2 z-30 p-3 rounded-lg bg-bg-tertiary border border-accent-blue/30
                            text-xs space-y-2 animate-slide-in shadow-xl">
                            <div className="flex justify-between items-center">
                                <span className="font-mono font-bold text-accent-blue">
                                    Layer {detailLayer.layer_index} â€” {CATEGORY_LABELS[detailLayer.category] || detailLayer.category}
                                </span>
                                <button
                                    onClick={() => setDetailLayer(null)}
                                    className="text-text-tertiary hover:text-text-primary text-sm"
                                >Ã—</button>
                            </div>
                            <p className="text-[11px] text-text-secondary leading-relaxed">
                                {CATEGORY_DESCRIPTIONS[detailLayer.category] || ''}
                            </p>
                            <p className="text-[11px] text-text-secondary leading-relaxed">
                                {detailLayer.description || 'No detailed description available.'}
                            </p>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
                                <span className="text-text-tertiary">Activation</span>
                                <span className="font-mono text-right">{(detailLayer.activation * 100).toFixed(1)}%</span>
                                <span className="text-text-tertiary">Confidence</span>
                                <span className="font-mono text-right">{(detailLayer.confidence * 100).toFixed(0)}%</span>
                                <span className="text-text-tertiary">Role</span>
                                <span className="font-mono text-right">{detailLayer.behavioral_role || 'â€”'}</span>
                                <span className="text-text-tertiary">Category</span>
                                <span className="font-mono text-right" style={{ color: CATEGORY_COLORS[detailLayer.category] }}>
                                    {detailLayer.category}
                                </span>
                            </div>
                            {/* Weight stats if available */}
                            {detailLayer.weight_stats && Object.keys(detailLayer.weight_stats).length > 0 && (
                                <div className="mt-1 pt-1 border-t border-border-subtle">
                                    <span className="text-[9px] text-text-tertiary font-semibold uppercase">Weight Stats</span>
                                    <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 mt-1 text-[10px]">
                                        {Object.entries(detailLayer.weight_stats).slice(0, 6).map(([key, val]) => (
                                            <div key={key} className="contents">
                                                <span className="text-text-tertiary truncate">{key.replace(/_/g, ' ')}</span>
                                                <span className="font-mono text-right">{typeof val === 'number' ? val.toFixed(3) : val}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {/* Analysis info if this layer was detected */}
                            {detailLayer.analysisInfo && (
                                <div className="mt-1 pt-1 border-t border-accent-amber/20">
                                    <span className="text-[9px] text-accent-amber font-semibold uppercase">Analysis Result</span>
                                    <p className="text-[10px] text-text-secondary mt-0.5">
                                        {detailLayer.analysisInfo.explanation || 'Anomalous behavior detected at this layer.'}
                                    </p>
                                    {detailLayer.analysisInfo.recommended_intervention && (
                                        <div className="mt-1 p-1.5 rounded bg-accent-green/10 border border-accent-green/20">
                                            <span className="text-[9px] text-accent-green">
                                                Recommended: strength{' '}
                                                <span className="font-mono font-bold">
                                                    {detailLayer.analysisInfo.recommended_intervention.strength > 0 ? '+' : ''}
                                                    {detailLayer.analysisInfo.recommended_intervention.strength.toFixed(1)}
                                                </span>
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}
                            <div className="flex gap-2 mt-1">
                                <button
                                    onClick={() => {
                                        onSelectLayer(detailLayer.layer_index);
                                        setDetailLayer(null);
                                    }}
                                    className="flex-1 py-1.5 rounded text-[10px] font-medium
                                        bg-accent-green/15 border border-accent-green/30 text-accent-green
                                        hover:border-accent-green/60 transition"
                                >
                                    {selectedLayers.includes(detailLayer.layer_index) ? 'Deselect' : 'Select for Steering'}
                                </button>
                            </div>
                        </div>
                    </>
                )}
            </div>

            {/* Active activations indicator */}
            {activations && Object.keys(activations).length > 0 && (
                <div className="px-3 py-2 border-t border-border-subtle text-[10px] text-text-tertiary flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse-subtle" />
                    Live activations from last prompt
                </div>
            )}
            {(!activations || Object.keys(activations).length === 0) && numLayers > 0 && (
                <div className="px-3 py-2 border-t border-border-subtle text-[10px] text-text-tertiary">
                    Send a prompt in chat to see live activations
                </div>
            )}
        </div>
    );
}



