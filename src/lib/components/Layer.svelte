<script lang="ts">
	import { Field, NumberStepper } from "svelte-ux";

	export let layerType = "linear";
	export let inputSize = 1;
	export let outputSize = 1;
	export let kernelSize = 1;
	export let isFirstLayer: boolean;
</script>

<h1 class="text-2xl font-semibold pb-4">Layer Type</h1>
<Field label="Type" let:id>
	<select
		bind:value={layerType}
		{id}
		class="text-sm w-full outline-none appearance-none cursor-pointer bg-surface-100"
	>
		<option value={"linear"}>Linear</option>
		<option value={"conv2d"}>2D Convolution</option>
		<option value={"relu"}>ReLU</option>
		<option value={"softmax"}>Softmax</option>
		<option value={"flatten"}>Flatten</option>
		<option value={"maxpool2d"}>2D Max Pooling</option>
	</select>
	<span slot="append"></span>
</Field>
<div class="py-2" />
{#if !["relu", "softmax", "flatten"].includes(layerType)}
	{#if isFirstLayer}
		<h1 class="text-2xl font-semibold py-4">Input Size</h1>
		<NumberStepper
			on:change={(e) => (inputSize = e.detail.value)}
			min={1}
			value={inputSize}
			bind:inputSize
			class="w-full"
		/>
		<div class="py-4" />
	{/if}
	<h1 class="text-2xl font-semibold pb-4">Output Size</h1>
	<NumberStepper
		on:change={(e) => (outputSize = e.detail.value)}
		min={1}
		value={outputSize}
		bind:outputSize
		class="w-full"
	/>
{/if}
<div class="py-2" />
{#if ["conv2d", "maxpool2d"].includes(layerType)}
	<h1 class="text-2xl font-semibold py-4">Kernel Size</h1>
	<NumberStepper
		on:change={(e) => (kernelSize = e.detail.value)}
		min={1}
		value={kernelSize}
		bind:kernelSize
		class="w-full"
	/>
{/if}
