/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                bg: {
                    primary: '#0a0a0a',
                    secondary: '#1a1a1a',
                    tertiary: '#141414',
                },
                border: {
                    subtle: '#2a2a2a',
                    hover: '#3a3a3a',
                },
                accent: {
                    green: '#10b981',
                    'green-hover': '#059669',
                    amber: '#f59e0b',
                    red: '#ef4444',
                    blue: '#3b82f6',
                    purple: '#8b5cf6',
                },
                text: {
                    primary: '#ffffff',
                    secondary: '#9ca3af',
                    tertiary: '#6b7280',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
            animation: {
                'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
                'slide-in': 'slide-in 0.3s ease-out',
                'fade-in': 'fade-in 0.2s ease-out',
            },
            keyframes: {
                'pulse-subtle': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                },
                'slide-in': {
                    '0%': { transform: 'translateY(10px)', opacity: 0 },
                    '100%': { transform: 'translateY(0)', opacity: 1 },
                },
                'fade-in': {
                    '0%': { opacity: 0 },
                    '100%': { opacity: 1 },
                },
            },
        },
    },
    plugins: [],
}
