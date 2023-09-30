/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['inter'],
      },
      colors: {
        'main': '#121318',
        'background': '#080808',
        'primary': '#8753FF',
        'primarydark': '#6638B3',
        'success': '#05A56E',
        'warning': '#DC2625',
        'neutral': '#9395A5',
        'neutraldark': '#6C7080',
        'border': '#333438',
        'symbol': '#8C00FF',
        'textdisabled': '#282D41',
        'signinmobile': '#E5E9F2'
      }
    },
  },
  plugins: [],
}

