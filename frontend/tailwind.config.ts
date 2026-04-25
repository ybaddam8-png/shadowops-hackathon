import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        navy: {
          50: '#f0f4ff',
          100: '#e6ebff',
          200: '#c7d5ff',
          300: '#a8bfff',
          400: '#7a9aff',
          500: '#4c75ff',
          600: '#3554e6',
          700: '#1e2b99',
          800: '#141d4d',
          900: '#0a0f26',
        },
      },
      backgroundColor: {
        'dark-bg': '#0f172a',
        'card-bg': '#1e293b',
        'input-bg': '#334155',
      },
      borderColor: {
        'border': '#334155',
      },
    },
  },
  plugins: [],
} satisfies Config
