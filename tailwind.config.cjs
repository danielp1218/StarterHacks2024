const colors = require("tailwindcss/colors");
const svelte_ux = require("svelte-ux/plugins/tailwind.cjs");

/** @type {import('tailwindcss').Config}*/
const config = {
	content: [
		"./src/**/*.{html,svelte}",
		"./node_modules/svelte-ux/**/*.{svelte,js}",
		"./node_modules/layerchart/**/*.{svelte,js}"
	],
	theme: {
		extend: {}
	},
	variants: {
		extend: {}
	},
	plugins: [svelte_ux],
	ux: {
		themes: {
			light: {
				"color-scheme": "light",
				primary: "hsl(9.8969 85.0877% 55.2941%)",
				secondary: "hsl(293.202 84.2324% 47.2549%)",
				accent: "hsl(212.514 89.9497% 60.9804%)",
				neutral: "hsl(214.2857 19.6262% 20.9804%)",
				"surface-100": "hsl(180 100% 100%)",
				"surface-200": "hsl(0 0% 94.902%)",
				"surface-300": "hsl(180 1.9608% 90%)"
			},
			dark: {
				"color-scheme": "dark",
				primary: "hsl(0 49.8039% 50%)",
				secondary: "hsl(0 49.8039% 50%)",
				accent: "hsl(215.3846 93.9759% 67.451%)",
				neutral: "hsl(213.3333 17.6471% 20%)",
				"surface-100": "hsl(200 3.8961% 84.902%)",
				"surface-200": "hsl(212.7273 18.0328% 11.9608%)",
				"surface-300": "hsl(213.3333 17.6471% 10%)"
			}
		}
	}
};

module.exports = config;
