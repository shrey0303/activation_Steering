// ── API Constants ────────────────────────────────────────────
export const API_BASE = '/api/v1';
export const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/v1/ws/generate`;

// ── Heatmap Colors ──────────────────────────────────────────
export const HEATMAP_COLORS = {
    low: '#1e40af',  // Blue
    medium: '#059669',  // Green
    high: '#d97706',  // Amber
    veryHigh: '#dc2626',  // Red
};

// ── Layer Categories ────────────────────────────────────────
export const CATEGORY_COLORS = {
    token_embedding: '#6366f1',       // Indigo
    positional_morphological: '#818cf8', // Light indigo
    syntactic_processing: '#3b82f6',  // Blue
    entity_semantic: '#06b6d4',       // Cyan
    knowledge_retrieval: '#10b981',   // Emerald
    reasoning_planning: '#f59e0b',    // Amber
    safety_alignment: '#ef4444',      // Red
    information_integration: '#8b5cf6', // Purple
    style_personality: '#ec4899',     // Pink
    output_distribution: '#f97316',   // Orange
    unknown: '#6b7280',               // Gray
};

export const CATEGORY_LABELS = {
    token_embedding: 'Token Embedding',
    positional_morphological: 'Position & Morphology',
    syntactic_processing: 'Syntactic Processing',
    entity_semantic: 'Entity & Semantic',
    knowledge_retrieval: 'Knowledge Retrieval',
    reasoning_planning: 'Reasoning & Planning',
    safety_alignment: 'Safety & Alignment',
    information_integration: 'Information Integration',
    style_personality: 'Style & Personality',
    output_distribution: 'Output Distribution',
};

// ── Defaults ────────────────────────────────────────────────
export const DEFAULT_MAX_TOKENS = 200;
export const DEFAULT_TEMPERATURE = 0.7;

// ── Supported Models ────────────────────────────────────────
// Curated list of HuggingFace models verified with SteerOps architecture.
// Architectures supported: model.layers (LLaMA/Mistral/Qwen/Phi),
// transformer.h (GPT-2/GPT-J), gpt_neox.layers (GPT-NeoX/Pythia),
// model.decoder.layers (OPT)
export const SUPPORTED_MODELS = [
    // ── LLaMA Family ─────────────────────────
    { id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', name: 'TinyLlama 1.1B Chat', params: '1.1B', arch: 'LLaMA', vram: '~2GB (4-bit)' },
    { id: 'meta-llama/Llama-2-7b-chat-hf', name: 'LLaMA 2 7B Chat', params: '7B', arch: 'LLaMA', vram: '~5GB (4-bit)' },
    { id: 'meta-llama/Llama-2-13b-chat-hf', name: 'LLaMA 2 13B Chat', params: '13B', arch: 'LLaMA', vram: '~9GB (4-bit)' },
    { id: 'meta-llama/Meta-Llama-3-8B-Instruct', name: 'LLaMA 3 8B Instruct', params: '8B', arch: 'LLaMA', vram: '~6GB (4-bit)' },
    { id: 'meta-llama/Meta-Llama-3.1-8B-Instruct', name: 'LLaMA 3.1 8B Instruct', params: '8B', arch: 'LLaMA', vram: '~6GB (4-bit)' },
    { id: 'meta-llama/Llama-3.2-1B-Instruct', name: 'LLaMA 3.2 1B Instruct', params: '1B', arch: 'LLaMA', vram: '~1.5GB (4-bit)' },
    { id: 'meta-llama/Llama-3.2-3B-Instruct', name: 'LLaMA 3.2 3B Instruct', params: '3B', arch: 'LLaMA', vram: '~3GB (4-bit)' },

    // ── Mistral Family ───────────────────────
    { id: 'mistralai/Mistral-7B-Instruct-v0.2', name: 'Mistral 7B Instruct v0.2', params: '7B', arch: 'Mistral', vram: '~5GB (4-bit)' },
    { id: 'mistralai/Mistral-7B-Instruct-v0.3', name: 'Mistral 7B Instruct v0.3', params: '7B', arch: 'Mistral', vram: '~5GB (4-bit)' },
    { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', name: 'Mixtral 8x7B MoE', params: '46.7B', arch: 'Mistral', vram: '~26GB (4-bit)' },
    { id: 'mistralai/Mistral-Nemo-Instruct-2407', name: 'Mistral Nemo 12B', params: '12B', arch: 'Mistral', vram: '~8GB (4-bit)' },

    // ── Qwen Family ──────────────────────────
    { id: 'Qwen/Qwen2-0.5B-Instruct', name: 'Qwen2 0.5B Instruct', params: '0.5B', arch: 'Qwen', vram: '~1GB (4-bit)' },
    { id: 'Qwen/Qwen2-1.5B-Instruct', name: 'Qwen2 1.5B Instruct', params: '1.5B', arch: 'Qwen', vram: '~2GB (4-bit)' },
    { id: 'Qwen/Qwen2-7B-Instruct', name: 'Qwen2 7B Instruct', params: '7B', arch: 'Qwen', vram: '~5GB (4-bit)' },
    { id: 'Qwen/Qwen2.5-0.5B-Instruct', name: 'Qwen2.5 0.5B Instruct', params: '0.5B', arch: 'Qwen', vram: '~1GB (4-bit)' },
    { id: 'Qwen/Qwen2.5-1.5B-Instruct', name: 'Qwen2.5 1.5B Instruct', params: '1.5B', arch: 'Qwen', vram: '~2GB (4-bit)' },
    { id: 'Qwen/Qwen2.5-3B-Instruct', name: 'Qwen2.5 3B Instruct', params: '3B', arch: 'Qwen', vram: '~3GB (4-bit)' },
    { id: 'Qwen/Qwen2.5-7B-Instruct', name: 'Qwen2.5 7B Instruct', params: '7B', arch: 'Qwen', vram: '~5GB (4-bit)' },

    // ── Phi Family (Microsoft) ───────────────
    { id: 'microsoft/phi-2', name: 'Phi-2 2.7B', params: '2.7B', arch: 'Phi', vram: '~3GB (4-bit)' },
    { id: 'microsoft/Phi-3-mini-4k-instruct', name: 'Phi-3 Mini 3.8B', params: '3.8B', arch: 'Phi', vram: '~3GB (4-bit)' },
    { id: 'microsoft/Phi-3.5-mini-instruct', name: 'Phi-3.5 Mini 3.8B', params: '3.8B', arch: 'Phi', vram: '~3GB (4-bit)' },

    // ── Gemma Family (Google) ────────────────
    { id: 'google/gemma-2b-it', name: 'Gemma 2B Instruct', params: '2B', arch: 'Gemma', vram: '~2GB (4-bit)' },
    { id: 'google/gemma-7b-it', name: 'Gemma 7B Instruct', params: '7B', arch: 'Gemma', vram: '~5GB (4-bit)' },
    { id: 'google/gemma-2-2b-it', name: 'Gemma 2 2B Instruct', params: '2B', arch: 'Gemma', vram: '~2GB (4-bit)' },
    { id: 'google/gemma-2-9b-it', name: 'Gemma 2 9B Instruct', params: '9B', arch: 'Gemma', vram: '~6GB (4-bit)' },

    // ── GPT-2 Family ─────────────────────────
    { id: 'openai-community/gpt2', name: 'GPT-2 124M', params: '124M', arch: 'GPT-2', vram: '~0.5GB' },
    { id: 'openai-community/gpt2-medium', name: 'GPT-2 Medium 355M', params: '355M', arch: 'GPT-2', vram: '~1GB' },
    { id: 'openai-community/gpt2-large', name: 'GPT-2 Large 774M', params: '774M', arch: 'GPT-2', vram: '~1.5GB' },
    { id: 'openai-community/gpt2-xl', name: 'GPT-2 XL 1.5B', params: '1.5B', arch: 'GPT-2', vram: '~2GB' },

    // ── GPT-NeoX / Pythia Family ─────────────
    { id: 'EleutherAI/pythia-70m', name: 'Pythia 70M', params: '70M', arch: 'GPT-NeoX', vram: '~0.3GB' },
    { id: 'EleutherAI/pythia-160m', name: 'Pythia 160M', params: '160M', arch: 'GPT-NeoX', vram: '~0.5GB' },
    { id: 'EleutherAI/pythia-410m', name: 'Pythia 410M', params: '410M', arch: 'GPT-NeoX', vram: '~1GB' },
    { id: 'EleutherAI/pythia-1b', name: 'Pythia 1B', params: '1B', arch: 'GPT-NeoX', vram: '~1.5GB' },
    { id: 'EleutherAI/pythia-1.4b', name: 'Pythia 1.4B', params: '1.4B', arch: 'GPT-NeoX', vram: '~2GB' },
    { id: 'EleutherAI/pythia-2.8b', name: 'Pythia 2.8B', params: '2.8B', arch: 'GPT-NeoX', vram: '~3GB (4-bit)' },
    { id: 'EleutherAI/gpt-neox-20b', name: 'GPT-NeoX 20B', params: '20B', arch: 'GPT-NeoX', vram: '~12GB (4-bit)' },

    // ── OPT Family (Meta) ────────────────────
    { id: 'facebook/opt-125m', name: 'OPT 125M', params: '125M', arch: 'OPT', vram: '~0.5GB' },
    { id: 'facebook/opt-350m', name: 'OPT 350M', params: '350M', arch: 'OPT', vram: '~1GB' },
    { id: 'facebook/opt-1.3b', name: 'OPT 1.3B', params: '1.3B', arch: 'OPT', vram: '~2GB' },
    { id: 'facebook/opt-2.7b', name: 'OPT 2.7B', params: '2.7B', arch: 'OPT', vram: '~3GB (4-bit)' },
    { id: 'facebook/opt-6.7b', name: 'OPT 6.7B', params: '6.7B', arch: 'OPT', vram: '~5GB (4-bit)' },

    // ── Falcon Family ────────────────────────
    { id: 'tiiuae/falcon-rw-1b', name: 'Falcon RW 1B', params: '1B', arch: 'Falcon', vram: '~1.5GB' },
    { id: 'tiiuae/falcon-7b-instruct', name: 'Falcon 7B Instruct', params: '7B', arch: 'Falcon', vram: '~5GB (4-bit)' },

    // ── StableLM Family ──────────────────────
    { id: 'stabilityai/stablelm-2-zephyr-1_6b', name: 'StableLM 2 Zephyr 1.6B', params: '1.6B', arch: 'StableLM', vram: '~2GB' },
    { id: 'stabilityai/stablelm-zephyr-3b', name: 'StableLM Zephyr 3B', params: '3B', arch: 'StableLM', vram: '~3GB (4-bit)' },

    // ── Sarvam Family ───────────────────────
    { id: 'sarvamai/sarvam-105b', name: 'Sarvam 105B', params: '105B', arch: 'Sarvam', vram: '~60GB (4-bit)' },
    { id: 'sarvamai/sarvam-30b', name: 'Sarvam 30B', params: '30B', arch: 'Sarvam', vram: '~18GB (4-bit)' }

];