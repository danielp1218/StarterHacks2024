<script lang="ts">
	import { Button, Card, Field, NumberStepper, Paginate, Progress, Step, Steps, Tooltip } from "svelte-ux";
	import Layer from "$lib/components/Layer.svelte";
	import { mdiTrashCan } from "@mdi/js";
	import { slide } from "svelte/transition";
	import { io } from "socket.io-client";
	import { quintIn } from 'svelte/easing';


	interface Layer {
		type: string;
		inputSize: number | undefined;
		outputSize: number | undefined;
		kernelSize: number | undefined;
	}

	let layers: Layer[] = [
		{
			type: "linear",
			inputSize: 1,
			outputSize: 1,
			kernelSize: undefined
		}
	];

	const addLayer = () => {
		layers = [
			...layers,
			{
				type: "linear",
				inputSize: 1,
				outputSize: 1,
				kernelSize: undefined
			}
		];

		for (let i = 0; i < layers.length; i++) {
			if (!["relu", "softmax", "flatten"].includes(layers[i].type)) {
				firstLayer = i;
				return;
			}
		}
	};

	const socket = io('http://127.0.0.1:5000', {withCredentials: true});
	let optimizerType: string = "sgd";
	let learningRate = 0.1;
	let schedulerType: string = "plateau";
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
			alert("Please upload a dataset");
			return;
		}

		let jsonToSend = {
			layers: layers,
			optimizer: {
				type: optimizerType,
				learningRate: learningRate
			},
			scheduler: {
				type: schedulerType,
				patience: patience
			},
			epochs: epochs
		};

		const testJson = {
			'layers': [
				{'type': 'flatten'},
				{'type': 'linear', 'in_channels': 28*28, 'out_channels': 256},
				{'type': 'relu'},
				{'type': 'linear', 'in_channels': 256, 'out_channels': 256},
				{'type': 'relu'},
				{'type': 'linear', 'in_channels': 256, 'out_channels': 10}
			],
			'optimizer': {
				'type': 'sgd',
				'lr': 0.001,
			},
			'loss': {
				'type': 'CrossEntropyLoss'
			},
			'reduceLrOnPlateau': {
				'type': 'ReduceLROnPlateau',
			}
		}
		// @ts-ignore
		jsonToSend = testJson;

		let done = false;

		// socket.on("connect", () => {
		waitingForServer = true;
		console.log(123);
		const file = datasetFile[0];
		const formData = new FormData();
		formData.append('file', file);

		
		socket.on("clientError", (message) => {
			console.error(message);
		})

		socket.on("started_training", (message) => {
			waitingForServer = false;
			training = true;
		});

		socket.on("epoch", (epoch) => {
			trainingEpoch = epoch;
		});

		socket.on("finished_training", (message) => {
			training = false;
			finishedTraining = true;
		});

		socket.on("send_weights", (weights) => {
			const blob = new Blob([weights], { type: "application/octet-stream" });
			const url = URL.createObjectURL(blob);
			const a = document.createElement("a");
			a.href = url;
			a.download = "weights.ckpt";
		});

		// const resp = await fetch('http://localhost:5000/dataset', {
		// 	method: 'POST',
		// 	body: formData
		// })
		// console.log(await resp.text())
		// if (!done) await socket.emitWithAck("dataset", datasetFile[0]);
		console.log(456);
		if (!done) await socket.emitWithAck("train", jsonToSend);
		console.log(789);
		done = true;
		// });
	};

	let waitingForServer = false;
	let training = false;
	let trainingEpoch = 0;
	let finishedTraining = false;

	let firstLayer = 0;

	

	const round = (num: number, places: number) => {
		const multiplier = Math.pow(10, places);
		return Math.round(num * multiplier) / multiplier;
	};
</script>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous">
<link
	href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&family=Rosario:ital,wght@0,300..700;1,300..700&family=Work+Sans:ital,wght@0,100..900;1,100..900&display=swap"
	rel="stylesheet"
/>

<main class="">
	<img class="bg" src="/background.png" alt="background" draggable="false">
	<div class="py-32 flex justify-center">
		<Card class="rounded-2xl">
			<div class="p-8">
				<h1 class="gradient-text font-bold pb-12">Insert Name Here</h1>
				<Paginate data={[1, 2, 3, 4]} perPage={1} let:pagination let:current>
					<div class="flex justify-center">
						<Steps>
							<Tooltip>
								<div slot="title" class="bg-opacity-30 bg-orange-300 p-2 rounded-xl" in:slide={{duration: 500, easing:quintIn, axis:'y'}}>
									<p class="w-72 text-xs text-center">In AI, layers are fundamental units of neural networks. They consist of nodes that process data and pass results to the next layer. There are three types: input, hidden, and output layers.</p>
								</div>
								<Step completed={current.page >= 1}>Layers</Step>
							</Tooltip>
							<Tooltip title="Hello">
								<div slot="title" class="bg-opacity-30 bg-pink-400 p-2 rounded-xl" in:slide={{duration: 500, easing:quintIn, axis:'y'}}>
									<p class="w-72 text-xs text-center">test</p>
								</div>
								<Step completed={current.page >= 2}>Optimizer</Step>
							</Tooltip>
							<Tooltip title="Hello">
								<div slot="title" class="bg-opacity-30 bg-purple-600 p-2 rounded-xl" in:slide={{duration: 500, easing:quintIn, axis:'y'}}>
									<p class="w-72 text-xs text-center">test</p>
								</div>
								<Step completed={current.page >= 3}>Learning Rate</Step>
							</Tooltip>
							<Tooltip title="Hello">
								<div slot="title" class="bg-opacity-30 bg-blue-600 p-2 rounded-xl" in:slide={{duration: 500, easing:quintIn, axis:'y'}}>
									<p class="w-72 text-xs text-center">test</p>
								</div>
								<Step completed={current.page >= 4}>Train</Step>
							</Tooltip>
						</Steps>
					</div>
					<div class="py-24 px-8">
						<div>
							{#if current.page === 1}
								<h1 class="text-4xl font-semibold pb-4">Layers</h1>
								{#each layers as layer, i}
									<div transition:slide>
										<!-- title="Layer {i + 1}" -->
										<Card class="rounded-xl">
											<div class="p-4">
												{#if i !== 0}
													<Layer
														isFirstLayer={i == firstLayer}
														inputSize={layers[i - 1].outputSize}
														bind:layerType={layers[i].type}
														bind:outputSize={layers[i].outputSize}
														bind:kernelSize={layers[i].kernelSize}
													/>
												{:else}
													<Layer
														isFirstLayer={i == firstLayer}
														bind:layerType={layers[i].type}
														bind:inputSize={layers[i].inputSize}
														bind:outputSize={layers[i].outputSize}
														bind:kernelSize={layers[i].kernelSize}
													/>
												{/if}
												{#if i !== 0}
													<Button
														icon={{
															data: mdiTrashCan,
															size: "2rem",
															style: "color: crimson"
														}}
														color="danger"
														on:click={() =>
															(layers = layers.filter(
																(_, index) => index !== i
															))}
													>
														Delete
													</Button>
												{/if}
											</div>
										</Card>
									</div>
									<div class="p-4" />
								{/each}
								<Button variant="fill" color="accent" size="lg" on:click={addLayer}
									>Add Layer</Button
								>
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
												<option value={"sgd"}
													>Stochastic Gradient Descent</option
												>
												<option value={"adam"}>Adam</option>
												<option value={"adagrad"}>Adagrad</option>
											</select>
											<span slot="append"></span>
										</Field>
										<div class="py-4" />
										<h1 class="text-2xl font-semibold pb-4">Learning Rate</h1>
										<NumberStepper
											class="w-full"
											on:change={(e) => (learningRate = e.detail.value)}
											min={0.00000000000001}
											max={1}
											step={0.00001}
											value={learningRate}
											bind:learningRate
										/>
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
										{#if schedulerType === "plateau"}
											<h1 class="text-2xl font-semibold pb-4">Patience</h1>
											<NumberStepper
												class="w-full"
												on:change={(e) => (patience = e.detail.value)}
												min={2}
												value={patience}
												bind:patience
											/>
										{/if}
									</div>
								</Card>
							{:else if current.page === 4}
								<Card title="Training" class="rounded-xl">
									<div class="p-4">
										<h1 class="text-2xl font-semibold pb-4">Epochs</h1>
										<NumberStepper
											class="w-full"
											on:change={(e) => (epochs = e.detail.value)}
											min={1}
											value={epochs}
											bind:epochs
										/>
										<div class="py-4" />
										<div class="flex items-center gap-4">
											<label
												class="cursor-pointer text-md bg-primary-500 p-4 rounded"
											>
												Upload Dataset
												<input
													accept=".zip"
													bind:files={datasetFile}
													id="avatar"
													name="avatar"
													type="file"
													required
												/>
											</label>
											<h1>{datasetFile?.[0]?.name ?? "Upload a file..."}</h1>
										</div>
										<div class="py-4" />
										{#if training}
											<Progress value={round(trainingEpoch / epochs, 3)} />
										{/if}
										<div class="py-4" />
										<Button
											size="lg"
											color="secondary"
											variant="fill"
											class="w-full"
											loading={waitingForServer}
											on:click={train}
											>Train
										</Button>
									</div>
								</Card>
							{/if}
						</div>
					</div>
					<div class="pt-12">
						<Button on:click={pagination.prevPage} disabled={current.isFirst}
							>Previous
						</Button>
						<Button
							on:click={pagination.nextPage}
							color="primary"
							variant="fill"
							disabled={current.isLast}
							>Next
						</Button>
					</div>
				</Paginate>
			</div>
		</Card>
	</div>
</main>

<style>
	* {
		font-family:
			Work Sans,
			sans-serif;
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
		background: -webkit-linear-gradient(
			180deg,
			rgba(45, 25, 183, 1) 0%,
			rgba(183, 25, 154, 1) 52%,
			rgba(255, 96, 0, 1) 100%
		);
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
	}

	input[type="file"] {
		display: none;
	}
</style>
