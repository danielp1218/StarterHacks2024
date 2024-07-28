<script lang="ts">
	import { Field, NumberStepper, Tooltip } from "svelte-ux";

	export let layerType = "linear";
	export let inputSize = 1;
	export let outputSize = 1;
	export let kernelSize = 1;
	export let isFirstLayer: boolean;
</script>

<Field label="Layer Type" let:id>
	<select
		bind:value={layerType}
		{id}
		class="text-xl w-full outline-none appearance-none cursor-pointer bg-surface-100"
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
<div class="flex">
	{#if !["relu", "softmax", "flatten"].includes(layerType)}
		{#if isFirstLayer}
			<div class="pe-4">
				<h1 class="font-semibold">Input Size</h1>
				<Tooltip title="Number of units from the previous layer.">
					<NumberStepper
						on:change={(e) => (inputSize = e.detail.value)}
						min={1}
						value={inputSize}
						bind:inputSize
						class="min-w-32"
					/>
				</Tooltip>
			</div>

		{/if}
		<div class="px-4">
			<Tooltip title="Number of units in the current layer.">
				<h1 class="font-semibold">Output Size</h1>
				<NumberStepper
					on:change={(e) => (outputSize = e.detail.value)}
					min={1}
					value={outputSize}
					bind:outputSize
					class="min-w-32"
				/>
			</Tooltip>
		</div>
	{/if}
	{#if ["conv2d", "maxpool2d"].includes(layerType)}
		<div class="px-4">
			<h1 class="font-semibold">Kernel Size</h1>
			<Tooltip title="Kernel size is the dimensions of the filter used in a convolutional neural network.">
				<NumberStepper
					on:change={(e) => (kernelSize = e.detail.value)}
					min={1}
					value={kernelSize}
					bind:kernelSize
					class="w-min-32"
				/>
			</Tooltip>
		</div>
	{/if}
</div>