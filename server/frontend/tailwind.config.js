/** @type {import('tailwindcss').Config} */
module.exports = {
  theme: {
    extend: {
      colors: {
        red: {
          500: '#ef4444',  
        },
        yellow: {
          500: '#eab308',  
        },
        green: {
          500: '#22c55e',  
        },
      },
    },
  },
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}