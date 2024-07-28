<script lang="ts">
	import { getSettings, settings } from "svelte-ux";
	import "../app.postcss";

	const { currentTheme, themes } = getSettings();

	$currentTheme.theme;
	currentTheme.setTheme("light");

	settings({});
</script>

<slot />
