<script lang="ts">
	import { Button, Card, Field, NumberStepper, Paginate, Progress, Step, Steps } from 'svelte-ux';
	import Layer from '$lib/components/Layer.svelte';
	import { mdiTrashCan } from '@mdi/js';
	import { slide } from 'svelte/transition';
	import { io } from 'socket.io-client';

	interface Layer {
		'type': string;
		'inputSize': number | undefined;
		'outputSize': number | undefined;
		'kernelSize': number | undefined;
	}

	let layers: Layer[] = [{
		'type': 'linear',
		'inputSize': 1,
		'outputSize': 1,
		'kernelSize': undefined
	}];

	const addLayer = () => {
		layers = [...layers, {
			'type': 'linear',
			'inputSize': 1,
			'outputSize': 1,
			'kernelSize': undefined
		}];
	};

	let optimizerType: string = 'sgd';
	let learningRate = 0.1;
	let schedulerType: string = 'plateau';
	let patience = 10;
	let epochs = 2;
	let datasetFile: FileList;

	// How websocket thing will work:
	// When train is clicked, loads while the files are being sent
	// load animation stops when server recieves files and acknowledges by sending a message back
	// then the training starts
	// server sends updates every epoch
	// use to update progress bar
	// when training is done, server sends a message back
	// svelte ux notification is displayed
	// maybe model is saved and can be downloaded
	// maybe also code
	// maybe also be able to test it

	const train = async () => {
		if (!datasetFile) {
			alert('Please upload a dataset');
			return;
		}

		const jsonToSend = {
			'layers': layers,
			'optimizer': {
				'type': optimizerType,
				'learningRate': learningRate
			},
			'scheduler': {
				'type': schedulerType,
				'patience': patience
			},
			'epochs': epochs
		};

		const socket = io('https://server-domain.com');

		socket.on('connect', () => {
			socket.emit('train', jsonToSend);
			socket.emit('dataset', datasetFile[0]);
		});

		waitingForServer = true;

		socket.on('started_training', (message) => {
			waitingForServer = false;
			training = true;
		});

		socket.on('epoch', (epoch) => {
			trainingEpoch = epoch;
		});

		socket.on('finished_training', (message) => {
			training = false;
			finishedTraining = true;
		});

		socket.on('send_weights', (weights) => {
			const blob = new Blob([weights], { type: 'application/octet-stream' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'weights.ckpt';
		});
	};

	let waitingForServer = false;
	let training = false;
	let trainingEpoch = 0;
	let finishedTraining = false;

	const round = (num: number, places: number) => {
		const multiplier = Math.pow(10, places);
		return Math.round(num * multiplier) / multiplier;
	};
</script>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link
	href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&family=Rosario:ital,wght@0,300..700;1,300..700&family=Work+Sans:ital,wght@0,100..900;1,100..900&display=swap"
	rel="stylesheet">

<main class="">
	<img class="bg" src="/background.png" alt="background" draggable="false">
	<div class="py-36 px-[36rem]">
		<Card class="rounded-2xl">
			<div class="p-8">
				<h1 class="gradient-text font-bold pb-12">Insert Name Here</h1>
				<Paginate data={[1, 2, 3, 4]} perPage={1} let:pagination let:current>
					<div class="flex justify-center">
						<Steps>
							<Step completed={current.page >= 1}>Layers</Step>
							<Step completed={current.page >= 2}>Optimizer</Step>
							<Step completed={current.page >= 3}>Learning Rate</Step>
							<Step completed={current.page >= 4}>Train</Step>
						</Steps>
					</div>
					<div class="py-24 px-8">
						<div>
							{#if current.page === 1}
								<h1 class="text-4xl font-semibold pb-4">Layers</h1>
								{#each layers as layer, i}
									<div transition:slide>
										<Card title="Layer {i + 1}" class="rounded-xl">
											<div class="p-4">
												{#if i !== 0}
													<Layer
														isFirstLayer={i === 0}
														inputSize={layers[i - 1].outputSize}
														bind:layerType={layers[i].type}
														bind:outputSize={layers[i].outputSize}
														bind:kernelSize={layers[i].kernelSize}
													/>
												{:else}
													<Layer
														isFirstLayer={true}
														bind:layerType={layers[i].type}
														bind:inputSize={layers[i].inputSize}
														bind:outputSize={layers[i].outputSize}
														bind:kernelSize={layers[i].kernelSize}
													/>
												{/if}
												{#if i !== 0}
													<Button
														icon={{ data: mdiTrashCan, size: "2rem", style: "color: crimson" }}
														color="danger"
														on:click={() => layers = layers.filter((_, index) => index !== i)}
													>
														Delete
													</Button>
												{/if}
											</div>
										</Card>
									</div>
									<div class="p-4" />
								{/each}
								<Button variant="fill" color="accent" size="lg" on:click={addLayer}>Add Layer</Button>
							{:else if current.page === 2}
								<Card title="Optimizer" class="rounded-xl">
									<div class="p-4">
										<h1 class="text-2xl font-semibold pb-4">Optimizer Type</h1>
										<Field label="Type" let:id>
											<select
												bind:value={optimizerType}
												{id}
												class="text-sm w-full outline-none appearance-none cursor-pointer bg-surface-100"
											>
												<option value={"sgd"}>Stochastic Gradient Descent</option>
												<option value={"adam"}>Adam</option>
												<option value={"adagrad"}>Adagrad</option>
											</select>
											<span slot="append"></span>
										</Field>
										<div class="py-4" />
										<h1 class="text-2xl font-semibold pb-4">Learning Rate</h1>
										<NumberStepper class="w-full" on:change={(e) => learningRate = e.detail.value}
																	 min={0.00000000000001} max={1} step={0.00001} value={learningRate}
																	 bind:learningRate />
									</div>
								</Card>
							{:else if current.page === 3}
								<Card title="Learning Rate Scheduler" class="rounded-xl">
									<div class="p-4">
										<h1 class="text-2xl font-semibold pb-4">Scheduler Type</h1>
										<Field label="Type" let:id>
											<select
												bind:value={schedulerType}
												{id}
												class="text-sm w-full outline-none appearance-none cursor-pointer bg-surface-100"
											>
												<option value={"plateau"}>ReduceLROnPlateau</option>
												<option value={"none"}>None</option>
											</select>
											<span slot="append"></span>
										</Field>
										<div class="py-4" />
										{#if schedulerType === 'plateau'}
											<h1 class="text-2xl font-semibold pb-4">Patience</h1>
											<NumberStepper class="w-full" on:change={(e) => patience = e.detail.value} min={2}
																		 value={patience} bind:patience />
										{/if}
									</div>
								</Card>

							{:else if current.page === 4}
								<Card title="Training" class="rounded-xl">
									<div class="p-4">
										<h1 class="text-2xl font-semibold pb-4">Epochs</h1>
										<NumberStepper class="w-full" on:change={(e) => epochs = e.detail.value}
																	 min={1} value={epochs}
																	 bind:epochs />
										<div class="py-4" />
										<div class="flex items-center gap-4">
											<label class="cursor-pointer text-md bg-primary-500 p-4 rounded">
												Upload Dataset
												<input
													accept=".zip" bind:files={datasetFile} id="avatar" name="avatar" type="file" required
												/>
											</label>
											<h1>{datasetFile?.[0]?.name ?? "Upload a file..."}</h1>
										</div>
										<div class="py-4" />
										{#if training}
											<Progress value={round(trainingEpoch / epochs, 3)} />
										{/if}
										<div class="py-4" />
										<Button size="lg" color="secondary" variant="fill" class="w-full" loading={waitingForServer}
														on:click={train}>Train
										</Button>
									</div>
								</Card>

							{/if}
						</div>
					</div>
					<div class="pt-12">
						<Button on:click={pagination.prevPage} disabled={current.isFirst}
						>Previous
						</Button
						>
						<Button
							on:click={pagination.nextPage}
							color="primary"
							variant="fill"
							disabled={current.isLast}>Next
						</Button
						>
					</div>
				</Paginate>
			</div>
		</Card>
	</div>
</main>

<style>
    * {
        font-family: Work Sans, sans-serif;
    }

    .bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: blur(24px);
    }

    .gradient-text {
        font-size: 72px;
        background: -webkit-linear-gradient(180deg, rgba(45, 25, 183, 1) 0%, rgba(183, 25, 154, 1) 52%, rgba(255, 96, 0, 1) 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    input[type="file"] {
        display: none;
    }
</style>