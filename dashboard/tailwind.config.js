/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        'dark': '#0f172a',
        'dark-secondary': '#1e293b',
        'dark-tertiary': '#334155',
        'neon-blue': '#0ea5e9',
        'neon-purple': '#a855f7',
        'neon-cyan': '#06b6d4'
      },
      backdropBlur: {
        'sm': '4px',
        'md': '12px',
        'lg': '16px',
        'xl': '24px'
      },
      boxShadow: {
        'glow-blue': '0 0 20px rgba(14, 165, 233, 0.3)',
        'glow-purple': '0 0 20px rgba(168, 85, 247, 0.3)',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)'
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 3s ease-in-out infinite'
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 20px rgba(14, 165, 233, 0.3)' },
          '50%': { boxShadow: '0 0 30px rgba(14, 165, 233, 0.6)' }
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' }
        }
      }
    }
  },
  plugins: []
}
