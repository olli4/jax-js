import { AluExp, AluGroup, AluOp, DEBUG, DType, Executable, FpHash, Kernel, Routines, SlotError, UnsupportedOpError, UnsupportedRoutineError, byteWidth, findPow2, isFloatDtype, mapSetUnion, prod, range, strip1, tuneNullopt, tuneWebgpu } from "./backend-BLx6HGHC.js";
import { createAllIterationsOffsetsBuffer, wrapRoutineForScan } from "./scan-wrapper-Dml-rzI-.js";

//#region src/backend/webgpu/builtins.ts
const threefrySrc = `
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return vec2<u32>(x0, x1);
}`;
const erfSrc = `
const _erf_p: f32 = 0.3275911;
const _erf_a1: f32 = 0.254829592;
const _erf_a2: f32 = -0.284496736;
const _erf_a3: f32 = 1.421413741;
const _erf_a4: f32 = -1.453152027;
const _erf_a5: f32 = 1.061405429;
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
fn erfc(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  let E = P_t * exp(-x * x);
  return select(2.0 - E, E, x >= 0.0);
}`;

//#endregion
//#region src/backend/webgpu/codegen.ts
const headerWgsl = String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();
function dtypeToWgsl(dtype, storage = false) {
	switch (dtype) {
		case DType.Bool: return storage ? "i32" : "bool";
		case DType.Int32: return "i32";
		case DType.Uint32: return "u32";
		case DType.Float32: return "f32";
		case DType.Float16: return "f16";
		default: throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
	}
}
function maxValueWgsl(dtype) {
	switch (dtype) {
		case DType.Bool: return "1";
		case DType.Int32: return "2147483647";
		case DType.Uint32: return "4294967295u";
		case DType.Float32: return "inf()";
		case DType.Float16: return "f16(inf())";
		default: throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
	}
}
function constToWgsl(dtype, value) {
	if (dtype === DType.Bool) return value ? "true" : "false";
	if (dtype === DType.Int32) return value.toString();
	if (dtype === DType.Uint32) return value.toString() + "u";
	if (dtype === DType.Float32) {
		if (Number.isNaN(value)) return "nan()";
		if (!Number.isFinite(value)) return value > 0 ? "inf()" : "-inf()";
		return "f32(" + value.toString() + ")";
	}
	if (dtype === DType.Float16) {
		if (Number.isNaN(value)) return "f16(nan())";
		if (!Number.isFinite(value)) return value > 0 ? "f16(inf())" : "f16(-inf())";
		return "f16(" + value.toString() + ")";
	}
	throw new Error(`Unsupported const dtype: ${dtype}`);
}
const gridOffsetY = 16384;
function calculateGrid(gridSize) {
	let gridX = gridSize;
	let gridY = 1;
	if (gridSize > 65535) {
		gridX = gridOffsetY;
		gridY = Math.ceil(gridSize / gridOffsetY);
	}
	return [gridX, gridY];
}

//#endregion
//#region src/backend/webgpu/reader.ts
/**
* Graphics state used to synchronously read data from WebGPU buffers.
*
* This trick is borrowed from TensorFlow.js. Basically, the idea is to create
* an offscreen canvas with one pixel for every 4 bytes ("device storage"), then
* configure it with a WebGPU context. Copy the buffer to a texture, then draw
* the canvas onto another offscreen canvas with '2d' context ("host storage").
*
* Once it's on host storage, we can use `getImageData()` to read the pixels
* from the image directly.
*
* We use 256x256 canvases here (256 KiB). The performance of this is bad
* because it involves multiple data copies, but it still works. We also
* actually need to copy the image twice: once in "opaque" mode for the RGB
* values, and once in "premultiplied" mode for the alpha channel.
*
* https://github.com/tensorflow/tfjs/blob/tfjs-v4.22.0/tfjs-backend-webgpu/src/backend_webgpu.ts#L379
*/
var SyncReader = class SyncReader {
	static alphaModes = ["opaque", "premultiplied"];
	static width = 256;
	static height = 256;
	initialized = false;
	deviceStorage;
	deviceContexts;
	hostStorage;
	hostContext;
	constructor(device) {
		this.device = device;
	}
	#init() {
		const makeCanvas = () => new OffscreenCanvas(SyncReader.width, SyncReader.height);
		this.deviceStorage = SyncReader.alphaModes.map(makeCanvas);
		this.deviceContexts = this.deviceStorage.map((canvas, i) => {
			const context = canvas.getContext("webgpu");
			context.configure({
				device: this.device,
				format: "bgra8unorm",
				usage: GPUTextureUsage.COPY_DST,
				alphaMode: SyncReader.alphaModes[i]
			});
			return context;
		});
		this.hostStorage = makeCanvas();
		this.hostContext = this.hostStorage.getContext("2d", { willReadFrequently: true });
		this.initialized = true;
	}
	read(buffer, start, count) {
		if (!this.initialized) this.#init();
		const deviceStorage = this.deviceStorage;
		const deviceContexts = this.deviceContexts;
		const hostContext = this.hostContext;
		const pixelsSize = Math.ceil(count / 4);
		const bytesPerRow = SyncReader.width * 4;
		const valsGPU = /* @__PURE__ */ new ArrayBuffer(pixelsSize * 4);
		for (let i = 0; i < deviceContexts.length; i++) {
			const texture = deviceContexts[i].getCurrentTexture();
			const readData = (width, height, offset$1) => {
				const encoder = this.device.createCommandEncoder();
				encoder.copyBufferToTexture({
					buffer,
					bytesPerRow,
					offset: offset$1 + start
				}, { texture }, {
					width,
					height,
					depthOrArrayLayers: 1
				});
				const commandBuffer = encoder.finish();
				this.device.queue.submit([commandBuffer]);
				hostContext.clearRect(0, 0, width, height);
				hostContext.drawImage(deviceStorage[i], 0, 0);
				const values = hostContext.getImageData(0, 0, width, height).data;
				const span = new Uint8ClampedArray(valsGPU, offset$1, 4 * width * height);
				const alphaMode = SyncReader.alphaModes[i];
				for (let k = 0; k < span.length; k += 4) if (alphaMode === "premultiplied") span[k + 3] = values[k + 3];
				else {
					span[k] = values[k + 2];
					span[k + 1] = values[k + 1];
					span[k + 2] = values[k];
				}
			};
			const pixelsPerCanvas = SyncReader.width * SyncReader.height;
			const wholeChunks = Math.floor(pixelsSize / pixelsPerCanvas);
			let remainder = pixelsSize % pixelsPerCanvas;
			const remainderRows = Math.floor(remainder / SyncReader.width);
			remainder = remainder % SyncReader.width;
			let offset = 0;
			for (let j = 0; j < wholeChunks; j++) {
				readData(SyncReader.width, SyncReader.height, offset);
				offset += pixelsPerCanvas * 4;
			}
			if (remainderRows > 0) {
				readData(SyncReader.width, remainderRows, offset);
				offset += remainderRows * SyncReader.width * 4;
			}
			if (remainder > 0) readData(remainder, 1, offset);
		}
		return new Uint8Array(valsGPU, 0, count);
	}
};

//#endregion
//#region src/backend/webgpu/routines.ts
function bitonicSortUniform(pass) {
	const ar = new Uint32Array(3);
	ar[0] = pass.kind === "sort" ? 0 : 1;
	ar[1] = pass.mergeStep ?? 0;
	ar[2] = pass.mergeStage ?? 0;
	return new Uint8Array(ar.buffer);
}
/**
* Generate a bitonic sort shader.
*
* We implement a variant of bitonic sort that [only has forward comparators](
* <https://sortingalgos.miraheze.org/wiki/Bitonic_Sort#Bitonic_Sort_using_Forward_Comparators>),
* so we don't need to allocate memory for power-of-two padding.
*
* This uses workgroup shared memory up to `2*workgroupSize` elements, for each
* array in `batches`. For larger arrays, multiple passes are done:
*
* - Initial "sort" pass: each workgroup sorts its `2*workgroupSize` elements.
* - Subsequent "merge" passes: each pass merges sorted sequences of size
*   `2^(step+1)` with multiple workgroups. This doesn't use shared memory.
*
* The total number of passes is roughly `log2(n / workgroupSize)^2 / 2`.
*/
function bitonicSortShader(device, dtype, n, batches, outputIndices) {
	const ty = dtypeToWgsl(dtype, true);
	const paddedN = 1 << Math.ceil(Math.log2(n || 1));
	const numThreads = Math.ceil(paddedN / 2);
	const workgroupSize = findPow2(numThreads, device.limits.maxComputeWorkgroupSizeX);
	const workgroupsPerBatch = numThreads / workgroupSize;
	const numStages = Math.log2(paddedN);
	const numLocalStages = Math.min(numStages, Math.log2(workgroupSize * 2));
	const needsF16 = dtype === DType.Float16;
	const padValue = isFloatDtype(dtype) ? `${ty}(nan())` : maxValueWgsl(dtype);
	const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

struct Uniforms {
  kind: u32, // 0 = sort, 1 = merge
  merge_step: u32, // half_block = 2^step
  merge_stage: u32, // only used for merge
}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;
${outputIndices ? `@group(0) @binding(2) var<storage, read_write> output_idx: array<i32>;` : ""}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

var<workgroup> shared_vals: array<${ty}, ${workgroupSize * 2}>;
${outputIndices ? `var<workgroup> shared_idx: array<i32, ${workgroupSize * 2}>;` : ""}

fn compare(a: ${ty}, b: ${ty}) -> bool {
${isFloatDtype(dtype) ? `
  let min_value = min(a, b);
  return a == min_value && b != min_value;` : "  return a < b;"}
}

fn compare_and_swap(i: u32, j: u32) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
${outputIndices ? `
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;` : ""}
  }
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let blockid = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let batch = blockid / ${workgroupsPerBatch}u;
  let wg_in_batch = blockid % ${workgroupsPerBatch}u;

  let tid = local_id.x;
  let base = batch * ${n}u;

  if (uniforms.kind == 0u || (uniforms.kind == 1u && uniforms.merge_step == ${numLocalStages - 1}u)) {
    let wg_base = wg_in_batch * ${workgroupSize * 2}u;

    // Load data into shared memory (2 elements per thread)
    let idx0 = tid * 2u;
    let idx1 = tid * 2u + 1u;
    // Load from input for initial 'sort' pass, then from output (read-write) for 'merge' passes.
    if (uniforms.kind == 0u) {
      shared_vals[idx0] = select(${padValue}, input[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, input[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${outputIndices ? `
      shared_idx[idx0] = i32(wg_base + idx0);
      shared_idx[idx1] = i32(wg_base + idx1);` : ""}
    } else {
      shared_vals[idx0] = select(${padValue}, output[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, output[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${outputIndices ? `
      shared_idx[idx0] = select(${n}, output_idx[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_idx[idx1] = select(${n}, output_idx[base + wg_base + idx1], wg_base + idx1 < ${n}u);` : ""}
    }
    workgroupBarrier();

    let initial_stage = select(0u, ${numLocalStages - 1}u, uniforms.kind != 0u);
    for (var stage = initial_stage; stage < ${numLocalStages}u; stage++) {
      for (var step1 = stage + 1u; step1 > 0u; step1--) {
        let step = step1 - 1u;
        let half_block = 1u << step;
        let is_first_step = uniforms.kind == 0u && step == stage;

        let block_offset = (tid / half_block) * half_block;
        let local_offset = tid % half_block;
        let i = block_offset * 2u + local_offset;
        let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);
        compare_and_swap(i, j);

        workgroupBarrier();
      }
    }

    if (wg_base + idx0 < ${n}u) {
      output[base + wg_base + idx0] = shared_vals[idx0];
      ${outputIndices ? `output_idx[base + wg_base + idx0] = shared_idx[idx0];` : ""}
    }
    if (wg_base + idx1 < ${n}u) {
      output[base + wg_base + idx1] = shared_vals[idx1];
      ${outputIndices ? `output_idx[base + wg_base + idx1] = shared_idx[idx1];` : ""}
    }
  } else {
    // Execute single merge pass for a step >= numLocalStages.
    let half_block = 1u << uniforms.merge_step;  // half_block >= workgroupSize * 2
    let thread_in_batch = wg_in_batch * ${workgroupSize} + tid;
    let is_first_step = uniforms.merge_step == uniforms.merge_stage;

    let block_offset = (thread_in_batch / half_block) * half_block;
    let local_offset = thread_in_batch % half_block;
    let i = block_offset * 2u + local_offset;
    let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);

    // Global version of compare_and_swap()
    if (j < ${n}u) {
      let val_i = output[base + i];
      let val_j = output[base + j];
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
${outputIndices ? `
        let tmp_idx = output_idx[base + i];
        output_idx[base + i] = output_idx[base + j];
        output_idx[base + j] = tmp_idx;` : ""}
      }
    }
  }
}
`.trim();
	const grid = calculateGrid(batches * workgroupsPerBatch);
	const passes = [{ kind: "sort" }];
	for (let mergeStage = numLocalStages; mergeStage < numStages; mergeStage++) for (let mergeStep = mergeStage; mergeStep >= numLocalStages - 1; mergeStep--) passes.push({
		kind: "merge",
		mergeStep,
		mergeStage
	});
	return [{
		code,
		numInputs: 1,
		numOutputs: outputIndices ? 2 : 1,
		hasUniform: true,
		passes: passes.map((pass) => ({
			grid,
			uniform: bitonicSortUniform(pass)
		}))
	}];
}
function createSort(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	const batches = prod(shape.slice(0, -1));
	return bitonicSortShader(device, dtype, n, batches, false);
}
function createArgsort(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	const batches = prod(shape.slice(0, -1));
	return bitonicSortShader(device, dtype, n, batches, true);
}
/**
* Generate a triangular solve shader.
*
* Solves A @ X.T = B.T for X, where A is upper-triangular.
* Uses a parallelized back-substitution:
*   1. Copy b to x
*   2. For j = n-1 down to 0:
*      - Divide x[j] by a[j,j] (single thread)
*      - All threads subtract x[j] * a[i,j] from x[i] for i < j in parallel
*/
function createTriangularSolve(device, type, params) {
	const dtype = type.inputDtypes[0];
	const aShape = type.inputShapes[0];
	const bShape = type.inputShapes[1];
	const n = aShape[aShape.length - 1];
	const numRhs = bShape[bShape.length - 2];
	const numMatrices = prod(aShape.slice(0, -2));
	const needsF16 = dtype === DType.Float16;
	const ty = dtypeToWgsl(dtype, true);
	const workgroupSize = findPow2(n, device.limits.maxComputeWorkgroupSizeX);
	const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> a: array<${ty}>;
@group(0) @binding(1) var<storage, read> b: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> x: array<${ty}>;

// Shared memory for the current pivot value x[j]
var<workgroup> x_j: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_idx = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let mat_idx = wg_idx / ${numRhs}u;
  let rhs_idx = wg_idx % ${numRhs}u;

  if (mat_idx >= ${numMatrices}u) {
    return;
  }

  let a_base = mat_idx * ${n * n}u;
  let bx_base = (mat_idx * ${numRhs}u + rhs_idx) * ${n}u;
  let tid = local_id.x;

  // Step 1: Copy b to x (threads collaborate)
  for (var idx = tid; idx < ${n}u; idx += ${workgroupSize}u) {
    x[bx_base + idx] = b[bx_base + idx];
  }
  storageBarrier();

  // Step 2: Back-substitution from j = n-1 down to 0
  for (var jj = 0u; jj < ${n}u; jj++) {
    let j = ${n - 1}u - jj;

    // Thread 0 computes x[j] = x[j] / a[j,j]
    if (tid == 0u) {
      ${params.unitDiagonal ? `x_j = x[bx_base + j];` : `x_j = x[bx_base + j] / a[a_base + j * ${n}u + j];`}
      x[bx_base + j] = x_j;
    }
    workgroupBarrier();  // Sync shared memory x_j

    // All threads subtract x[j] * a[i,j] from x[i] for i < j
    for (var i = tid; i < j; i += ${workgroupSize}u) {
      x[bx_base + i] -= x_j * a[a_base + i * ${n}u + j];
    }
    workgroupBarrier();
    storageBarrier();
  }
}
`.trim();
	const totalWorkgroups = numMatrices * numRhs;
	const grid = calculateGrid(totalWorkgroups);
	return [{
		code,
		numInputs: 2,
		numOutputs: 1,
		hasUniform: false,
		passes: [{ grid }]
	}];
}
/**
* Generate a Cholesky decomposition shader.
*
* Computes the lower triangular matrix L such that A = L * L^T for each
* positive semi-definite matrix in the batch. Uses the Cholesky-Crout
* algorithm which processes column-by-column.
*
* For each column j:
*   1. All threads compute their row's sum in parallel and store to output
*   2. Thread 0 computes L[j][j] = sqrt(output[j][j]) and stores to shared memory
*   3. All threads divide their output[i][j] by L[j][j] in parallel
*/
function createCholesky(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	const batches = prod(shape.slice(0, -2));
	const needsF16 = dtype === DType.Float16;
	const ty = dtypeToWgsl(dtype, true);
	const workgroupSize = findPow2(n, device.limits.maxComputeWorkgroupSizeX);
	const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let base = batch * ${n * n}u;
  let tid = local_id.x;

  // Zero out output and copy lower triangle from input (threads collaborate)
  for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
    let row = idx / ${n}u;
    let col = idx % ${n}u;
    output[base + idx] = select(0, input[base + idx], col <= row);
  }
  storageBarrier();

  // Cholesky-Crout algorithm: process column by column
  for (var j = 0u; j < ${n}u; j++) {
    // Step 1: All threads compute sum for their rows i >= j in parallel
    // sum = A[i][j] - sum(L[i][k] * L[j][k] for k < j)
    for (var i = j + tid; i < ${n}u; i += ${workgroupSize}u) {
      var sum = output[base + i * ${n}u + j];
      for (var k = 0u; k < j; k++) {
        sum -= output[base + i * ${n}u + k] * output[base + j * ${n}u + k];
      }
      output[base + i * ${n}u + j] = sum;
    }
    storageBarrier();

    // Step 2: Thread 0 computes L[j][j] = sqrt(output[j][j])
    if (tid == 0u) {
      L_jj = sqrt(output[base + j * ${n}u + j]);
      output[base + j * ${n}u + j] = L_jj;
    }
    workgroupBarrier();

    // Step 3: All threads divide output[i][j] by L[j][j] for i > j
    for (var i = j + 1u + tid; i < ${n}u; i += ${workgroupSize}u) {
      output[base + i * ${n}u + j] /= L_jj;
    }
    storageBarrier();
  }
}
`.trim();
	const grid = calculateGrid(batches);
	return [{
		code,
		numInputs: 1,
		numOutputs: 1,
		hasUniform: false,
		passes: [{ grid }]
	}];
}
/**
* Generate an LU decomposition shader with partial pivoting.
*
* Computes PA = LU where P is a permutation matrix, L is lower triangular
* with unit diagonal, and U is upper triangular.
*
* For each column j:
*   1. Find pivot row (max absolute value in column j, rows >= j)
*   2. Swap rows j and pivot row
*   3. Compute L[i][j] = A[i][j] / A[j][j] for i > j
*   4. Update submatrix: A[i][k] -= L[i][j] * A[j][k] for i > j, k > j
*/
function createLU(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const m = shape[shape.length - 2];
	const n = shape[shape.length - 1];
	const r = Math.min(m, n);
	const batches = prod(shape.slice(0, -2));
	const needsF16 = dtype === DType.Float16;
	const ty = dtypeToWgsl(dtype, true);
	const workgroupSize = findPow2(Math.max(m, n), device.limits.maxComputeWorkgroupSizeX);
	const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> lu: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> pivots: array<i32>;
@group(0) @binding(3) var<storage, read_write> perm: array<i32>;

var<workgroup> pivot_row: u32;
var<workgroup> pivot_val: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let lu_base = batch * ${m * n}u;
  let piv_base = batch * ${r}u;
  let perm_base = batch * ${m}u;
  let tid = local_id.x;

  // Copy input to lu
  for (var idx = tid; idx < ${m * n}u; idx += ${workgroupSize}u) {
    lu[lu_base + idx] = input[lu_base + idx];
  }
  // Initialize permutation
  for (var idx = tid; idx < ${m}u; idx += ${workgroupSize}u) {
    perm[perm_base + idx] = i32(idx);
  }
  storageBarrier();

  // LU decomposition with partial pivoting
  for (var j = 0u; j < ${r}u; j++) {
    // Step 1: Thread 0 finds pivot (max abs value in column j, rows >= j)
    if (tid == 0u) {
      var max_val = abs(lu[lu_base + j * ${n}u + j]);
      var max_row = j;
      for (var i = j + 1u; i < ${m}u; i++) {
        let val = abs(lu[lu_base + i * ${n}u + j]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }
      pivot_row = max_row;
      pivot_val = lu[lu_base + max_row * ${n}u + j];
      pivots[piv_base + j] = i32(max_row);
    }
    workgroupBarrier();

    // Step 2: Swap rows j and pivot_row (threads collaborate)
    let pr = pivot_row;
    if (pr != j) {
      for (var col = tid; col < ${n}u; col += ${workgroupSize}u) {
        let tmp = lu[lu_base + j * ${n}u + col];
        lu[lu_base + j * ${n}u + col] = lu[lu_base + pr * ${n}u + col];
        lu[lu_base + pr * ${n}u + col] = tmp;
      }
      if (tid == 0u) {
        let tmp_p = perm[perm_base + j];
        perm[perm_base + j] = perm[perm_base + pr];
        perm[perm_base + pr] = tmp_p;
      }
    }
    storageBarrier();

    // Step 3: Compute L[i][j] and update submatrix
    // Each thread handles one row i > j
    for (var i = j + 1u + tid; i < ${m}u; i += ${workgroupSize}u) {
      let factor = lu[lu_base + i * ${n}u + j] / pivot_val;
      lu[lu_base + i * ${n}u + j] = factor; // L[i][j]
      for (var k = j + 1u; k < ${n}u; k++) {
        lu[lu_base + i * ${n}u + k] -= factor * lu[lu_base + j * ${n}u + k];
      }
    }
    storageBarrier();
  }
}
`.trim();
	const grid = calculateGrid(batches);
	return [{
		code,
		numInputs: 1,
		numOutputs: 3,
		hasUniform: false,
		passes: [{ grid }]
	}];
}
function createRoutineShader(device, routine) {
	switch (routine.name) {
		case Routines.Sort: return createSort(device, routine.type);
		case Routines.Argsort: return createArgsort(device, routine.type);
		case Routines.TriangularSolve: return createTriangularSolve(device, routine.type, routine.params);
		case Routines.Cholesky: return createCholesky(device, routine.type);
		case Routines.LU: return createLU(device, routine.type);
		default: throw new UnsupportedRoutineError(routine.name, "webgpu");
	}
}

//#endregion
//#region src/backend/webgpu.ts
const COPY_WORKGROUP_SIZE = 64;
const COPY_SHADER_CODE = String.raw`
${headerWgsl}

struct CopyParams {
  srcOffset: u32,
  dstOffset: u32,
  size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(1) @binding(0) var<uniform> params: CopyParams;

fn byte_mask(n: u32) -> u32 {
  if (n >= 4u) { return 0xffffffffu; }
  return (1u << (n * 8u)) - 1u;
}

fn load_unaligned(offset: u32) -> u32 {
  let word = offset >> 2u;
  let shift = (offset & 3u) * 8u;
  if (shift == 0u) {
    return src[word];
  }
  let low = src[word];
  let high = src[word + 1u];
  return (low >> shift) | (high << (32u - shift));
}

@compute @workgroup_size(${COPY_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let gid = id.x + id.y * ${gridOffsetY}u;

  // Each thread handles one *destination word* exclusively, preventing
  // read-modify-write races when dstOffset is not 4-byte aligned.
  let firstDstWord = params.dstOffset >> 2u;
  let wordIdx = firstDstWord + gid;
  let lastDstWord = (params.dstOffset + params.size + 3u) >> 2u;
  if (wordIdx >= lastDstWord) { return; }

  // Byte range of this destination word
  let wordByteStart = wordIdx * 4u;

  // Intersect [dstOffset, dstOffset+size) with [wordByteStart, wordByteStart+4)
  let copyStart = max(params.dstOffset, wordByteStart);
  let copyEnd = min(params.dstOffset + params.size, wordByteStart + 4u);
  let nbytes = copyEnd - copyStart;

  // Read corresponding source bytes (unaligned read is safe)
  let srcByteOff = params.srcOffset + (copyStart - params.dstOffset);
  let value = load_unaligned(srcByteOff);

  if (nbytes == 4u) {
    // Full word write — entire word is within copy range
    dst[wordIdx] = value;
  } else {
    // Partial word — preserve bytes outside the copy range
    let shift = (copyStart & 3u) * 8u;
    let mask = byte_mask(nbytes) << shift;
    let cur = dst[wordIdx];
    dst[wordIdx] = (cur & ~mask) | ((value << shift) & mask);
  }
}
`.trim();
/** Implementation of `Backend` that uses WebGPU in browsers. */
var WebGPUBackend = class {
	type = "webgpu";
	maxArgs;
	pipelines;
	syncReader;
	buffers;
	nextSlot;
	#cachedShaderMap = /* @__PURE__ */ new Map();
	#reusableZsb;
	#copyPipeline = null;
	constructor(device) {
		this.device = device;
		if (DEBUG >= 3 && device.adapterInfo) console.info("webgpu adapter:", device.adapterInfo.vendor, device.adapterInfo.architecture);
		this.maxArgs = this.device.limits.maxStorageBuffersPerShaderStage - 1;
		this.pipelines = new ShaderPipelineCache(device);
		this.syncReader = new SyncReader(device);
		this.buffers = /* @__PURE__ */ new Map();
		this.nextSlot = 1;
		this.#reusableZsb = this.#createBuffer(4);
		device.addEventListener("uncapturederror", (event) => {
			console.error("Uncaptured error in WebGPU backend:", event.error.message);
		});
	}
	malloc(size, initialData) {
		let buffer;
		const paddedSize = Math.ceil(size / 4) * 4;
		if (size === 0) buffer = this.#reusableZsb;
		else if (initialData) {
			if (initialData.byteLength !== size) throw new Error("initialData size does not match buffer size");
			if (initialData.byteLength < 4096) {
				buffer = this.#createBuffer(paddedSize, { mapped: true });
				new Uint8Array(buffer.getMappedRange(), 0, size).set(initialData);
				buffer.unmap();
			} else {
				buffer = this.#createBuffer(paddedSize);
				if (initialData.byteLength % 4 === 0) this.device.queue.writeBuffer(buffer, 0, initialData);
				else {
					const aligned = initialData.byteLength - initialData.byteLength % 4;
					this.device.queue.writeBuffer(buffer, 0, initialData, 0, aligned);
					const remainder = new Uint8Array(4);
					remainder.set(initialData.subarray(aligned));
					this.device.queue.writeBuffer(buffer, aligned, remainder);
				}
			}
		} else buffer = this.#createBuffer(paddedSize);
		const slot = this.nextSlot++;
		this.buffers.set(slot, {
			buffer,
			size,
			ref: 1
		});
		return slot;
	}
	incRef(slot) {
		const buffer = this.buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref++;
	}
	decRef(slot) {
		const buffer = this.buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref--;
		if (buffer.ref === 0) {
			this.buffers.delete(slot);
			if (buffer.buffer !== this.#reusableZsb) buffer.buffer.destroy();
		}
	}
	slotCount() {
		return this.buffers.size;
	}
	async read(slot, start, count) {
		const { buffer, size } = this.#getBuffer(slot);
		if (buffer === this.#reusableZsb) return new Uint8Array();
		if (start === void 0) start = 0;
		if (count === void 0) count = size - start;
		const paddedSize = Math.ceil(count / 4) * 4;
		const staging = this.#createBuffer(paddedSize, { read: true });
		try {
			const commandEncoder = this.device.createCommandEncoder();
			commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, paddedSize);
			this.device.queue.submit([commandEncoder.finish()]);
			await staging.mapAsync(GPUMapMode.READ);
			const arrayBuffer = staging.getMappedRange();
			return new Uint8Array(arrayBuffer.slice(), 0, count);
		} finally {
			staging.destroy();
		}
	}
	readSync(slot, start, count) {
		const { buffer, size } = this.#getBuffer(slot);
		if (buffer === this.#reusableZsb) return new Uint8Array();
		if (start === void 0) start = 0;
		if (count === void 0) count = size - start;
		return this.syncReader.read(buffer, start, count);
	}
	copyBufferToBuffer(srcSlot, srcOffset, dstSlot, dstOffset, size) {
		const { buffer: srcBuf } = this.#getBuffer(srcSlot);
		const { buffer: dstBuf } = this.#getBuffer(dstSlot);
		const commandEncoder = this.device.createCommandEncoder();
		const paddedSize = Math.ceil(size / 4) * 4;
		commandEncoder.copyBufferToBuffer(srcBuf, srcOffset, dstBuf, dstOffset, paddedSize);
		this.device.queue.submit([commandEncoder.finish()]);
	}
	copyBufferWithShader(srcSlot, srcOffset, dstSlot, dstOffset, size) {
		if (size <= 0) return;
		const { buffer: srcBuf } = this.#getBuffer(srcSlot);
		const { buffer: dstBuf } = this.#getBuffer(dstSlot);
		const commandEncoder = this.device.createCommandEncoder();
		const uniformBuffer = this.#encodeCopyWithShader(commandEncoder, srcBuf, srcOffset, dstBuf, dstOffset, size);
		this.device.queue.submit([commandEncoder.finish()]);
		if (uniformBuffer) uniformBuffer.destroy();
	}
	#getCopyPipeline() {
		if (this.#copyPipeline) return this.#copyPipeline;
		const shader = {
			code: COPY_SHADER_CODE,
			numInputs: 1,
			numOutputs: 1,
			hasUniform: true,
			passes: [{
				grid: [1, 1],
				uniform: new Uint8Array(16)
			}]
		};
		this.#copyPipeline = this.pipelines.prepareSync(shader);
		return this.#copyPipeline;
	}
	#encodeCopyWithShader(commandEncoder, srcBuf, srcOffset, dstBuf, dstOffset, size) {
		if (size <= 0) return null;
		const firstDstWord = dstOffset >>> 2;
		const lastDstWord = dstOffset + size + 3 >>> 2;
		const words = lastDstWord - firstDstWord;
		const workgroups = Math.ceil(words / COPY_WORKGROUP_SIZE);
		if (workgroups === 0) return null;
		const [gridX, gridY] = calculateGrid(workgroups);
		const pipeline = this.#getCopyPipeline();
		const storageBindGroup = this.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
				binding: 0,
				resource: { buffer: srcBuf }
			}, {
				binding: 1,
				resource: { buffer: dstBuf }
			}]
		});
		const uniformBuffer = this.device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM,
			mappedAtCreation: true
		});
		new Uint32Array(uniformBuffer.getMappedRange()).set([
			srcOffset,
			dstOffset,
			size,
			0
		]);
		uniformBuffer.unmap();
		const uniformBindGroup = this.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
			entries: [{
				binding: 0,
				resource: { buffer: uniformBuffer }
			}]
		});
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, storageBindGroup);
		passEncoder.setBindGroup(1, uniformBindGroup);
		passEncoder.dispatchWorkgroups(gridX, gridY);
		passEncoder.end();
		return uniformBuffer;
	}
	#cachedShader(kernel) {
		const cacheKey = FpHash.hash(kernel);
		let result = this.#cachedShaderMap.get(cacheKey);
		if (!result) {
			result = pipelineSource(this.device, kernel);
			this.#cachedShaderMap.set(cacheKey, result);
		}
		return result;
	}
	async prepareKernel(kernel) {
		if (kernel.isMultiOutput) {
			const dispatches = [];
			for (const output of kernel.outputs) {
				const singleKernel = Kernel.single(kernel.nargs, output.size, output.exp, output.reduction);
				const shader$1 = this.#cachedShader(singleKernel);
				const pipeline$1 = await this.pipelines.prepare(shader$1);
				dispatches.push({
					...shader$1,
					pipeline: pipeline$1
				});
			}
			return new Executable(kernel, dispatches);
		}
		const shader = this.#cachedShader(kernel);
		const pipeline = await this.pipelines.prepare(shader);
		return new Executable(kernel, [{
			...shader,
			pipeline
		}]);
	}
	prepareKernelSync(kernel) {
		if (kernel.isMultiOutput) {
			const dispatches = [];
			for (const output of kernel.outputs) {
				const singleKernel = Kernel.single(kernel.nargs, output.size, output.exp, output.reduction);
				const shader$1 = this.#cachedShader(singleKernel);
				const pipeline$1 = this.pipelines.prepareSync(shader$1);
				dispatches.push({
					...shader$1,
					pipeline: pipeline$1
				});
			}
			return new Executable(kernel, dispatches);
		}
		const shader = this.#cachedShader(kernel);
		const pipeline = this.pipelines.prepareSync(shader);
		return new Executable(kernel, [{
			...shader,
			pipeline
		}]);
	}
	async prepareRoutine(routine) {
		const shaders = createRoutineShader(this.device, routine);
		const dispatches = await Promise.all(shaders.map(async (shader) => {
			const pipeline = await this.pipelines.prepare(shader);
			return {
				...shader,
				pipeline
			};
		}));
		return new Executable(routine, dispatches);
	}
	prepareRoutineSync(routine) {
		const shaders = createRoutineShader(this.device, routine);
		const dispatches = shaders.map((shader) => {
			const pipeline = this.pipelines.prepareSync(shader);
			return {
				...shader,
				pipeline
			};
		});
		return new Executable(routine, dispatches);
	}
	dispatch(exe, inputs, outputs) {
		const inputBuffers = inputs.map((slot) => this.#getBuffer(slot).buffer);
		const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
		const kernel = exe.source;
		if (kernel.isMultiOutput) {
			if (exe.data.length !== kernel.outputs.length) throw new Error(`webgpu: multi-output kernel dispatch count mismatch: ${exe.data.length} vs ${kernel.outputs.length}`);
			for (let i = 0; i < exe.data.length; i++) pipelineSubmit(this.device, [exe.data[i]], inputBuffers, [outputBuffers[i]]);
		} else pipelineSubmit(this.device, exe.data, inputBuffers, outputBuffers);
	}
	/**
	* Prepare a native scan operation for efficient execution.
	* Returns null if the scan cannot be natively executed.
	*/
	prepareNativeScan(params) {
		const { bodyKernel } = params;
		if (!bodyKernel) return null;
		try {
			const shader = nativeScanShaderSource(this.device, params);
			const pipeline = this.pipelines.prepareSync(shader);
			const syntheticKernel = Kernel.single(bodyKernel.nargs, bodyKernel.size, bodyKernel.exp, bodyKernel.reduction);
			return new Executable(syntheticKernel, [{
				...shader,
				pipeline
			}]);
		} catch (e) {
			if (DEBUG >= 2) console.warn("WebGPU native scan codegen failed:", e);
			return null;
		}
	}
	/**
	* Dispatch a native scan operation.
	* @param exe - The prepared native scan executable
	* @param consts - Constant buffer slots
	* @param initCarry - Initial carry buffer slots
	* @param xs - Input xs buffer slots
	* @param carryOut - Output carry buffer slots
	* @param ysStacked - Output stacked ys buffer slots
	*/
	dispatchNativeScan(exe, consts, initCarry, xs, carryOut, ysStacked) {
		const constsBuffers = consts.map((slot) => this.#getBuffer(slot).buffer);
		const initCarryBuffers = initCarry.map((slot) => this.#getBuffer(slot).buffer);
		const xsBuffers = xs.map((slot) => this.#getBuffer(slot).buffer);
		const carryOutBuffers = carryOut.map((slot) => this.#getBuffer(slot).buffer);
		const ysStackedBuffers = ysStacked.map((slot) => this.#getBuffer(slot).buffer);
		const commandEncoder = this.device.createCommandEncoder();
		for (const { pipeline,...shader } of exe.data) {
			const allBuffers = [
				...constsBuffers,
				...initCarryBuffers,
				...xsBuffers,
				...carryOutBuffers,
				...ysStackedBuffers
			];
			const bindGroup = this.device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: allBuffers.map((buffer, i) => ({
					binding: i,
					resource: { buffer }
				}))
			});
			for (const { grid } of shader.passes) {
				if (prod(grid) === 0) continue;
				const passEncoder = commandEncoder.beginComputePass();
				passEncoder.setPipeline(pipeline);
				passEncoder.setBindGroup(0, bindGroup);
				passEncoder.dispatchWorkgroups(grid[0], grid[1]);
				passEncoder.end();
			}
		}
		this.device.queue.submit([commandEncoder.finish()]);
	}
	/**
	* Prepare a multi-kernel native scan operation for efficient execution.
	* Handles scan bodies with multiple independent kernels (e.g., 2 matmuls).
	* Returns null if the scan cannot be natively executed.
	*/
	prepareNativeScanMulti(params) {
		const { steps } = params;
		if (!steps || steps.length === 0) return null;
		try {
			const shader = nativeScanMultiShaderSource(this.device, params);
			const pipeline = this.pipelines.prepareSync(shader);
			const firstKernel = steps[0].kernel;
			return new Executable(firstKernel, [{
				...shader,
				pipeline
			}]);
		} catch (e) {
			if (DEBUG >= 2) console.warn("WebGPU native scan multi codegen failed:", e);
			return null;
		}
	}
	/**
	* Returns the minimum uniform buffer offset alignment for preencoded scan.
	*/
	getPreencodedScanAlignment() {
		return this.device.limits.minUniformBufferOffsetAlignment ?? 256;
	}
	/**
	* Prepare a preencoded scan operation for routine bodies (matmul, conv, etc.).
	* Returns the prepared executable if successful, null otherwise.
	*
	* Preencoded scan encodes all iteration dispatches in a single command buffer,
	* eliminating JS roundtrip overhead per iteration. Uses ping-pong buffers
	* for carry state and uniform-based offset bindings for xs/ys slicing.
	*
	* This approach avoids minStorageBufferOffsetAlignment issues by:
	* 1. Binding full buffers (no offset in GPUBufferBinding)
	* 2. Adding uniform offset variables to the shader
	* 3. Using dynamic uniform buffer offsets for per-iteration offsets
	*/
	preparePreencodedScan(params) {
		const { xsElemStrides, ysElemStrides, bodyRoutine, numConsts, numCarry, numX, numY, length, reverse, carrySizes, routineInputJitIds, routineOutputJitIds } = params;
		if (!bodyRoutine || bodyRoutine.data.length === 0) {
			if (DEBUG >= 2) console.log("Preencoded scan: invalid routine");
			return null;
		}
		if (numX === 0 && numY === 0) {
			if (DEBUG >= 2) console.log("Preencoded scan: no xs/ys, using direct dispatch");
			return null;
		}
		const scanInfo = {
			numConsts,
			numCarry,
			routineInputJitIds,
			routineOutputJitIds
		};
		const wrappedShaders = [];
		for (const shader of bodyRoutine.data) {
			if (shader.hasUniform) {
				if (DEBUG >= 2) console.log("Preencoded scan: shader already has uniform, skipping");
				return null;
			}
			const wrapped = wrapRoutineForScan(shader, scanInfo);
			if (DEBUG >= 2) {
				console.log("Wrapped shader code:", wrapped.code.substring(0, 500));
				console.log("Wrapped hasUniform:", wrapped.hasUniform);
			}
			if (!wrapped.hasUniform) {
				if (DEBUG >= 2) console.log("Preencoded scan: shader doesn't need offsets");
				return null;
			}
			const module = this.device.createShaderModule({ code: wrapped.code });
			const pipeline = this.device.createComputePipeline({
				layout: "auto",
				compute: {
					module,
					entryPoint: "main"
				}
			});
			wrappedShaders.push({
				...shader,
				code: wrapped.code,
				hasUniform: true,
				pipeline
			});
		}
		const alignment = this.getPreencodedScanAlignment();
		const { buffer: offsetData, alignment: offsetAlignment } = createAllIterationsOffsetsBuffer(numX, numY, length, xsElemStrides, ysElemStrides, alignment, reverse);
		const offsetBuffer = this.device.createBuffer({
			size: offsetData.length,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true
		});
		new Uint8Array(offsetBuffer.getMappedRange()).set(offsetData);
		offsetBuffer.unmap();
		if (DEBUG >= 1) console.log(`Preencoded scan: prepared for ${length} iterations with uniform offsets`);
		const copyUsesShader = carrySizes.map((size) => size % 4 !== 0);
		return {
			params,
			wrappedShaders,
			offsetBuffer,
			offsetAlignment,
			copyUsesShader
		};
	}
	/**
	* Dispatch a preencoded scan operation with routine body.
	*
	* Uses ping-pong buffers for carry and uniform-based offsets for xs/ys.
	* All iteration dispatches are encoded in a single command buffer.
	* Dynamic uniform buffer offsets are used for per-iteration offset values.
	*/
	dispatchPreencodedScan(prepared, constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots) {
		const { params, wrappedShaders, offsetBuffer, offsetAlignment, copyUsesShader } = prepared;
		const { length, carrySizes, numCarry, numConsts, routineInputJitIds, routineOutputJitIds } = params;
		const constBuffers = constSlots.map((slot) => this.#getBuffer(slot).buffer);
		const initCarryBuffers = initCarrySlots.map((slot) => this.#getBuffer(slot).buffer);
		const xsBuffers = xsSlots.map((slot) => this.#getBuffer(slot).buffer);
		const carryOutBuffers = carryOutSlots.map((slot) => this.#getBuffer(slot).buffer);
		const ysStackedBuffers = ysStackedSlots.map((slot) => this.#getBuffer(slot).buffer);
		const carryPing = carrySizes.map((size) => this.#createBuffer(size));
		const carryPong = carrySizes.map((size) => this.#createBuffer(size));
		const commandEncoder = this.device.createCommandEncoder();
		const copyUniformBuffers = [];
		for (let i = 0; i < numCarry; i++) commandEncoder.copyBufferToBuffer(initCarryBuffers[i], 0, carryPing[i], 0, carrySizes[i]);
		const xsStart = numConsts + numCarry;
		for (const shader of wrappedShaders) {
			const { pipeline, passes } = shader;
			const createStorageBindGroup = (readCarry, writeCarry) => {
				const entries = [];
				let binding = 0;
				if (DEBUG >= 2) console.log("createStorageBindGroup:", {
					numConsts,
					numCarry,
					xsStart,
					routineInputJitIds,
					routineOutputJitIds,
					constBuffers: constBuffers.length,
					xsBuffers: xsBuffers.length,
					readCarry: readCarry.length,
					writeCarry: writeCarry.length,
					ysStackedBuffers: ysStackedBuffers.length
				});
				for (let i = 0; i < routineInputJitIds.length; i++) {
					const jitId = routineInputJitIds[i];
					let buffer;
					if (jitId < numConsts) buffer = constBuffers[jitId];
					else if (jitId < xsStart) {
						const carryIdx = jitId - numConsts;
						buffer = readCarry[carryIdx];
					} else {
						const xIdx = jitId - xsStart;
						buffer = xsBuffers[xIdx];
					}
					entries.push({
						binding: binding++,
						resource: { buffer }
					});
				}
				for (let i = 0; i < routineOutputJitIds.length; i++) {
					const buffer = writeCarry[i];
					if (!buffer) throw new Error(`Preencoded scan: routine output ${i} has no corresponding carry buffer (writeCarry.length=${writeCarry.length})`);
					entries.push({
						binding: binding++,
						resource: { buffer }
					});
				}
				return this.device.createBindGroup({
					layout: pipeline.getBindGroupLayout(0),
					entries
				});
			};
			const pingBindGroup = createStorageBindGroup(carryPing, carryPong);
			const pongBindGroup = createStorageBindGroup(carryPong, carryPing);
			const uniformBindGroups = [];
			for (let iter = 0; iter < length; iter++) {
				const iterOffset = iter * offsetAlignment;
				uniformBindGroups.push(this.device.createBindGroup({
					layout: pipeline.getBindGroupLayout(1),
					entries: [{
						binding: 0,
						resource: {
							buffer: offsetBuffer,
							offset: iterOffset,
							size: offsetAlignment
						}
					}]
				}));
			}
			const filteredPasses = passes.filter(({ grid }) => prod(grid) > 0);
			for (let iter = 0; iter < length; iter++) {
				const storageBindGroup = iter % 2 === 0 ? pingBindGroup : pongBindGroup;
				for (const { grid } of filteredPasses) {
					const passEncoder = commandEncoder.beginComputePass();
					passEncoder.setPipeline(pipeline);
					passEncoder.setBindGroup(0, storageBindGroup);
					passEncoder.setBindGroup(1, uniformBindGroups[iter]);
					passEncoder.dispatchWorkgroups(grid[0], grid[1]);
					passEncoder.end();
				}
				const currentCarryBuffers = iter % 2 === 0 ? carryPong : carryPing;
				for (let c = 0; c < numCarry; c++) {
					const copySize = carrySizes[c];
					if (copySize <= 0) continue;
					const yOffset = iter * copySize;
					if (!copyUsesShader[c]) commandEncoder.copyBufferToBuffer(currentCarryBuffers[c], 0, ysStackedBuffers[c], yOffset, copySize);
					else {
						const uniformBuffer = this.#encodeCopyWithShader(commandEncoder, currentCarryBuffers[c], 0, ysStackedBuffers[c], yOffset, copySize);
						if (uniformBuffer) copyUniformBuffers.push(uniformBuffer);
					}
				}
			}
		}
		const finalCarry = length % 2 === 0 ? carryPing : carryPong;
		for (let i = 0; i < numCarry; i++) {
			const copySize = carrySizes[i];
			if (copySize <= 0) continue;
			if (!copyUsesShader[i]) commandEncoder.copyBufferToBuffer(finalCarry[i], 0, carryOutBuffers[i], 0, copySize);
			else {
				const uniformBuffer = this.#encodeCopyWithShader(commandEncoder, finalCarry[i], 0, carryOutBuffers[i], 0, copySize);
				if (uniformBuffer) copyUniformBuffers.push(uniformBuffer);
			}
		}
		this.device.queue.submit([commandEncoder.finish()]);
		for (const buf of copyUniformBuffers) buf.destroy();
		for (const buf of [...carryPing, ...carryPong]) buf.destroy();
	}
	#getBuffer(slot) {
		const buffer = this.buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		return {
			buffer: buffer.buffer,
			size: buffer.size
		};
	}
	/**
	* Create a GPU buffer.
	*
	* By default, this creates a general-purpose buffer with the given size.
	*
	* - If `mapped` is true, initialize the buffer in mapped mode so that it can
	*   be populated with data from the CPU. (Call `.unmap()` later.)
	* - If `read` is true, create a staging buffer for returning data to CPU.
	*   (Call `.mapAsync()` later.)
	*/
	#createBuffer(size, { mapped = false, read = false } = {}) {
		if (read && mapped) throw new Error("mapped and read cannot both be true");
		const buffer = this.device.createBuffer({
			size,
			usage: read ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			mappedAtCreation: mapped
		});
		return buffer;
	}
};
/** Unique symbols for indent control in shader emitter. */
const PUSH_INDENT = Symbol("pushIndent");
const POP_INDENT = Symbol("popIndent");
/**
* Create a shader code emitter with indentation support.
* Returns emit function, indent control symbols, and a function to get final code.
*/
function createShaderEmitter() {
	const shader = [];
	let indent = "";
	const emit = (...lines) => {
		for (const line of lines) if (line === PUSH_INDENT) indent += "  ";
		else if (line === POP_INDENT) indent = indent.slice(0, -2);
		else shader.push(line ? indent + line : line);
	};
	return {
		emit,
		pushIndent: PUSH_INDENT,
		popIndent: POP_INDENT,
		getCode: () => shader.join("\n")
	};
}
/**
* Translate a binary/unary AluOp to WGSL code.
*
* This is a shared helper used by both regular kernel codegen and scan codegen.
* Returns the WGSL expression string, or undefined if the op is not handled.
*
* @param op The AluOp to translate
* @param srcs Source expression strings (already generated)
* @param dtype The result dtype
* @param _srcDtype The dtype of the first source operand (for comparisons)
*/
function translateAluOpToWgsl(op, srcs, dtype, _srcDtype) {
	const [a, b, c] = srcs;
	if (op === AluOp.Add) {
		if (dtype === DType.Bool) return `(${a} || ${b})`;
		return `(${a} + ${b})`;
	}
	if (op === AluOp.Sub) return `(${a} - ${b})`;
	if (op === AluOp.Mul) {
		if (dtype === DType.Bool) return `(${a} && ${b})`;
		return `(${a} * ${b})`;
	}
	if (op === AluOp.Idiv) return isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
	if (op === AluOp.Mod) return `(${a} % ${b})`;
	if (op === AluOp.Min) {
		if (dtype === DType.Bool) return `(${a} && ${b})`;
		return `min(${strip1(a)}, ${strip1(b)})`;
	}
	if (op === AluOp.Max) {
		if (dtype === DType.Bool) return `(${a} || ${b})`;
		return `max(${strip1(a)}, ${strip1(b)})`;
	}
	if (op === AluOp.Cmplt) return `(${a} < ${b})`;
	if (op === AluOp.Sin) return `sin(${strip1(a)})`;
	if (op === AluOp.Cos) return `cos(${strip1(a)})`;
	if (op === AluOp.Asin) return `asin(${strip1(a)})`;
	if (op === AluOp.Atan) return `atan(${strip1(a)})`;
	if (op === AluOp.Exp) return `exp(${strip1(a)})`;
	if (op === AluOp.Log) return `log(${strip1(a)})`;
	if (op === AluOp.Sqrt) return `sqrt(${strip1(a)})`;
	if (op === AluOp.Reciprocal) return `(1.0 / ${a})`;
	if (op === AluOp.Floor) return `floor(${strip1(a)})`;
	if (op === AluOp.Ceil) return `ceil(${strip1(a)})`;
	if (op === AluOp.Cast) return `${dtypeToWgsl(dtype)}(${strip1(a)})`;
	if (op === AluOp.Bitcast) return `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
	if (op === AluOp.Where) return `select(${strip1(c)}, ${strip1(b)}, ${strip1(a)})`;
	return void 0;
}
/**
* Translate Erf/Erfc with f32 precision wrapper.
*/
function translateErfToWgsl(op, a, dtype) {
	const funcName = op === AluOp.Erf ? "erf" : "erfc";
	if (dtype !== DType.Float32) return `${dtypeToWgsl(dtype)}(${funcName}(f32(${strip1(a)})))`;
	return `${funcName}(${strip1(a)})`;
}
/**
* Compiles an expression into WebGPU shader source code.
*
* Returns the shader source and the number of workgroups to dispatch along x
* and y axes, to run the kernel.
*/
function pipelineSource(device, kernel) {
	const tune = tuneWebgpu(kernel);
	if (DEBUG >= 3) console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
	const { nargs, reduction: re } = kernel;
	const args = Array.from({ length: nargs }, (_, i) => `in${i}`);
	const { emit, pushIndent, popIndent, getCode } = createShaderEmitter();
	if (tune.exp.some((exp) => exp.dtype === DType.Float16) || tune.epilogue?.some((exp) => exp.dtype === DType.Float16)) {
		if (!device.features.has("shader-f16")) throw new Error("WebGPU device does not support shader-f16 feature");
		emit("enable f16;");
	}
	emit(headerWgsl);
	const distinctOps = mapSetUnion(tune.exp.distinctOps(), tune.epilogue?.distinctOps());
	if (distinctOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
	if (distinctOps.has(AluOp.Erf) || distinctOps.has(AluOp.Erfc)) emit(erfSrc);
	emit("");
	const usedArgs = Array.from({ length: nargs }, () => null);
	tune.exp.fold((exp) => {
		if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
	});
	tune.epilogue?.fold((exp) => {
		if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
	});
	for (let i = 0; i < nargs; i++) {
		const ty = dtypeToWgsl(usedArgs[i] ?? DType.Float32, true);
		emit(`@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`);
	}
	const resultTy = dtypeToWgsl(kernel.dtype, true);
	emit(`@group(0) @binding(${nargs}) var<storage, read_write> result : array<${resultTy}>;`);
	const workgroupSize = findPow2(tune.threadCount, 256);
	const gridSize = Math.ceil(tune.threadCount / workgroupSize);
	const [gridX, gridY] = calculateGrid(gridSize);
	emit("", `@compute @workgroup_size(${workgroupSize})`, "fn main(@builtin(global_invocation_id) id : vec3<u32>) {", pushIndent);
	if (gridY === 1) emit(`if (id.x >= ${tune.threadCount}) { return; }`, "let gidx: i32 = i32(id.x);");
	else {
		const sizeX = gridX * workgroupSize;
		emit(`if (${sizeX} * id.y + id.x >= ${tune.threadCount}) { return; }`, `let gidx: i32 = i32(${sizeX} * id.y + id.x);`);
	}
	let gensymCount = 0;
	const gensym = () => `alu${gensymCount++}`;
	const isGensym = (text) => text.match(/^alu[0-9]+$/);
	if (args.length > 0) emit(args.map((arg) => `_ = &${arg};`).join(" "));
	const references = /* @__PURE__ */ new Map();
	const seen = /* @__PURE__ */ new Set();
	const countReferences = (exp) => {
		references.set(exp, (references.get(exp) ?? 0) + 1);
		if (!seen.has(exp)) {
			seen.add(exp);
			for (const src of exp.src) countReferences(src);
		}
	};
	const expContext = /* @__PURE__ */ new Map();
	const gen = (exp) => {
		if (expContext.has(exp)) return expContext.get(exp);
		const { op, src, dtype, arg } = exp;
		let source = "";
		if (op === AluOp.Const) return constToWgsl(dtype, arg);
		else if (op === AluOp.Special) return arg[0];
		else if (op === AluOp.Variable) return arg;
		else if (op === AluOp.GlobalIndex) {
			source = `${args[arg[0]]}[${strip1(gen(src[0]))}]`;
			if (dtype === DType.Bool) source = `(${source} != 0)`;
		} else if (op === AluOp.Threefry2x32) {
			const x = gensym();
			const [k0, k1, c0, c1] = src.map((x$1) => strip1(gen(x$1)));
			emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
			if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
			else if (arg === 0) source = `${x}.x`;
			else if (arg === 1) source = `${x}.y`;
			else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
		} else if (op === AluOp.Reciprocal && src[0].op === AluOp.Sqrt) {
			const a = gen(src[0].src[0]);
			source = `inverseSqrt(${a})`;
		} else if (op === AluOp.Cmpne) {
			const a = gen(src[0]);
			const b = gen(src[1]);
			if (isFloatDtype(src[0].dtype)) {
				const x = isGensym(a) ? a : gensym();
				if (x !== a) emit(`let ${x} = ${a};`);
				source = `(${x} != ${b} || min(${x}, ${dtypeToWgsl(src[0].dtype)}(inf())) != ${x})`;
			} else source = `(${a} != ${b})`;
		} else if (op === AluOp.Erf || op === AluOp.Erfc) {
			const a = gen(src[0]);
			source = translateErfToWgsl(op, a, dtype);
		} else if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
			const a = gen(src[0]);
			const b = gen(src[1]);
			source = translateAluOpToWgsl(op, [a, b], dtype, src[0].dtype) ?? "";
		} else if (AluGroup.Unary.has(op)) {
			const a = gen(src[0]);
			source = translateAluOpToWgsl(op, [a], dtype) ?? "";
		} else if (op === AluOp.Where) source = translateAluOpToWgsl(op, [
			gen(src[0]),
			gen(src[1]),
			gen(src[2])
		], dtype) ?? "";
		if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
		const typeName = dtypeToWgsl(dtype);
		if ((references.get(exp) ?? 0) > 1) {
			const name = gensym();
			expContext.set(exp, name);
			emit(`let ${name}: ${typeName} = ${strip1(source)};`);
			return name;
		} else {
			expContext.set(exp, source);
			return source;
		}
	};
	if (!re) {
		countReferences(tune.exp);
		let rhs = strip1(gen(tune.exp));
		if (resultTy !== dtypeToWgsl(tune.exp.dtype)) rhs = `${resultTy}(${rhs})`;
		emit(`result[gidx] = ${rhs};`);
	} else {
		if ((tune.size.groups ?? 1) > 1) throw new Error("WebGPU backend does not support group optimization yet");
		const unroll = tune.size.unroll ?? 1;
		const upcast = tune.size.upcast ?? 1;
		const acc = [...Array(upcast)].map((_, i) => `acc${i}`);
		for (let i = 0; i < upcast; i++) emit(`var ${acc[i]}: ${dtypeToWgsl(re.dtype)} = ${constToWgsl(re.dtype, re.identity)};`);
		emit(`for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`, pushIndent);
		const exps = [];
		const cache = /* @__PURE__ */ new Map();
		for (let up = 0; up < upcast; up++) {
			exps.push([]);
			for (let un = 0; un < unroll; un++) {
				const exp = tune.exp.substitute({
					upcast: AluExp.i32(up),
					unroll: AluExp.i32(un)
				});
				exps[up].push(exp.simplify(cache));
				countReferences(exps[up][un]);
			}
		}
		const items = exps.map((ar) => ar.map(gen).map(strip1));
		for (let i = 0; i < upcast; i++) {
			let rhs = items[i][0];
			for (let j = 1; j < unroll; j++) if (re.op === AluOp.Add) rhs = `${rhs} + ${items[i][j]}`;
			else if (re.op === AluOp.Mul) rhs = `${rhs} * ${items[i][j]}`;
			else if (re.op === AluOp.Min) rhs = re.dtype === DType.Bool ? `(${rhs} && ${items[i][j]})` : `min(${rhs}, ${items[i][j]})`;
			else if (re.op === AluOp.Max) rhs = re.dtype === DType.Bool ? `(${rhs} || ${items[i][j]})` : `max(${rhs}, ${items[i][j]})`;
			else throw new Error(`Unsupported reduction op: ${re.op}`);
			if (re.op === AluOp.Add) emit(`${acc[i]} += ${rhs};`);
			else if (re.op === AluOp.Mul) emit(`${acc[i]} *= ${rhs};`);
			else if (re.op === AluOp.Min) if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} && ${rhs};`);
			else emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
			else if (re.op === AluOp.Max) if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} || ${rhs};`);
			else emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
			else throw new Error(`Unsupported reduction op: ${re.op}`);
		}
		emit(popIndent, "}");
		expContext.clear();
		references.clear();
		seen.clear();
		const outputIdxExps = [];
		const fusionExps = [];
		for (let i = 0; i < upcast; i++) {
			const exp = tune.outputIdxExp.substitute({ upcast: AluExp.i32(i) });
			outputIdxExps.push(exp.simplify(cache));
			countReferences(outputIdxExps[i]);
			fusionExps.push(tune.epilogue.substitute({
				acc: AluExp.variable(re.dtype, acc[i]),
				upcast: AluExp.i32(i)
			}).simplify(cache));
			countReferences(fusionExps[i]);
		}
		for (let i = 0; i < upcast; i++) {
			const index = strip1(gen(outputIdxExps[i]));
			let rhs = strip1(gen(fusionExps[i]));
			if (resultTy !== dtypeToWgsl(fusionExps[i].dtype)) rhs = `${resultTy}(${rhs})`;
			emit(`result[${index}] = ${rhs};`);
		}
	}
	emit(popIndent, "}");
	return {
		code: getCode(),
		numInputs: nargs,
		numOutputs: 1,
		hasUniform: false,
		passes: [{ grid: [gridX, gridY] }]
	};
}
/**
* Generate a WGSL shader for native scan with inlined body kernel.
*
* CRITICAL INVARIANT: This shader is only correct for per-element-independent kernels.
* Each GPU thread i operates exclusively on carry[i] and xs[iter, i] — no cross-thread
* communication occurs. This invariant is enforced at JIT compile time: only elementwise
* kernels (no cross-element dependencies) qualify for native scan fusion.
*
* Without this invariant, the lack of global barriers between iterations would cause
* data races. WGSL barriers are workgroup-scoped only, not global across all threads.
*
* This is a convenience wrapper around nativeScanMultiShaderSource for single-kernel scans.
*/
function nativeScanShaderSource(device, params) {
	const { length, numConsts, constSizes, carrySizes, xsStrides, ysStrides, bodyKernel, numCarry, reverse } = params;
	if (numCarry !== 1 || ysStrides.length !== 1) throw new Error("Native scan: only single carry/output supported");
	const step = {
		kernel: bodyKernel,
		inputs: [],
		outputCarryIdx: 0,
		outputSize: bodyKernel.size
	};
	return nativeScanMultiShaderSource(device, {
		length,
		numConsts,
		constSizes,
		numCarry,
		carrySizes,
		numX: xsStrides.length,
		xsStrides,
		numY: ysStrides.length,
		ysStrides,
		steps: [step],
		reverse
	});
}
/**
* Generate WGSL expression code for a scan body.
* Handles the input layout: [consts..., carry..., xs...]
* - gid < numConsts: constant buffers (no iteration offset)
* - gid < numConsts + numCarry: carry buffers
* - gid >= numConsts + numCarry: xs buffers (with iteration offset via dataIdx)
*/
function genScanExpressionWithRidx(exp, dtype, numConsts, numCarry) {
	const gen = (e) => {
		const { op, src, dtype: eDtype, arg } = e;
		if (op === AluOp.GlobalIndex) {
			const gid = arg[0];
			const idxCode = gen(src[0]);
			if (gid < numConsts) return `const${gid}[${idxCode}]`;
			else if (gid < numConsts + numCarry) {
				const carryIdx = gid - numConsts;
				return `carry${carryIdx}[${idxCode}]`;
			} else {
				const xIdx = gid - numConsts - numCarry;
				const stride = arg[1];
				return `xs${xIdx}[i32(dataIdx) * ${stride} + ${idxCode}]`;
			}
		}
		if (op === AluOp.Const) return constToWgsl(eDtype, arg);
		if (op === AluOp.Special) {
			const name = Array.isArray(arg) ? arg[0] : arg;
			if (name === "gidx") return "gidx";
			if (name === "ridx") return "ridx";
			return name;
		}
		if (op === AluOp.Variable) {
			if (arg === "acc") return "acc";
			if (arg === "gidx") return "gidx";
			if (arg === "ridx") return "ridx";
			return arg;
		}
		if (op === AluOp.Erf || op === AluOp.Erfc) return translateErfToWgsl(op, gen(src[0]), eDtype);
		if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
			const result = translateAluOpToWgsl(op, [gen(src[0]), gen(src[1])], eDtype, src[0].dtype);
			if (result) return result;
		}
		if (AluGroup.Unary.has(op)) {
			const result = translateAluOpToWgsl(op, [gen(src[0])], eDtype);
			if (result) return result;
		}
		if (op === AluOp.Where) {
			const result = translateAluOpToWgsl(op, [
				gen(src[0]),
				gen(src[1]),
				gen(src[2])
			], eDtype);
			if (result) return result;
		}
		throw new Error(`genScanExpressionWithRidx: unsupported op ${AluOp[op]}`);
	};
	return strip1(gen(exp));
}
/**
* Generate a WGSL shader for native scan with multiple kernel steps.
*
* Each kernel step can have a different size, so we use conditional execution.
* Kernels are assumed to be independent (each writes to its own carry buffer),
* so no workgroup barrier is needed between kernels.
*
* Buffer layout:
*   - binding 0..numConsts-1: constant buffers (read)
*   - binding numConsts..numConsts+numCarry-1: initCarry buffers (read)
*   - binding numConsts+numCarry..: xs buffers (read)
*   - binding ...: carryOut buffers (read_write)
*   - binding ...: ysStacked buffers (write)
*/
function nativeScanMultiShaderSource(device, params) {
	const { length, numConsts, constSizes: _constSizes, carrySizes, xsStrides: _xsStrides, ysStrides, steps, numCarry, numX, numY, reverse } = params;
	const dtype = steps[0]?.kernel.dtype ?? DType.Float32;
	const resultTy = dtypeToWgsl(dtype, true);
	const elemSize = byteWidth(dtype);
	const maxKernelSize = Math.max(...steps.map((s) => s.kernel.size), 1);
	const { emit, pushIndent, popIndent, getCode } = createShaderEmitter();
	if (dtype === DType.Float16) {
		if (!device.features.has("shader-f16")) throw new Error("WebGPU device does not support shader-f16 feature");
		emit("enable f16;");
	}
	emit(headerWgsl);
	const allDistinctOps = /* @__PURE__ */ new Set();
	for (const step of steps) {
		const tune = tuneNullopt(step.kernel);
		for (const [op] of tune.exp.distinctOps()) allDistinctOps.add(op);
		if (tune.epilogue) for (const [op] of tune.epilogue.distinctOps()) allDistinctOps.add(op);
	}
	if (allDistinctOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
	if (allDistinctOps.has(AluOp.Erf) || allDistinctOps.has(AluOp.Erfc)) emit(erfSrc);
	emit("");
	let bindingIdx = 0;
	for (let i = 0; i < numConsts; i++) emit(`@group(0) @binding(${bindingIdx++}) var<storage, read> const${i}: array<${resultTy}>;`);
	for (let i = 0; i < numCarry; i++) emit(`@group(0) @binding(${bindingIdx++}) var<storage, read> initCarry${i}: array<${resultTy}>;`);
	for (let i = 0; i < numX; i++) emit(`@group(0) @binding(${bindingIdx++}) var<storage, read> xs${i}: array<${resultTy}>;`);
	for (let i = 0; i < numCarry; i++) emit(`@group(0) @binding(${bindingIdx++}) var<storage, read_write> carry${i}: array<${resultTy}>;`);
	for (let i = 0; i < numY; i++) emit(`@group(0) @binding(${bindingIdx++}) var<storage, read_write> ys${i}: array<${resultTy}>;`);
	const workgroupSize = Math.min(Math.max(maxKernelSize, 1), 256);
	const [gridX, gridY] = calculateGrid(Math.ceil(Math.max(maxKernelSize, 1) / workgroupSize));
	emit("", `@compute @workgroup_size(${workgroupSize})`, "fn main(@builtin(global_invocation_id) id: vec3<u32>) {", pushIndent);
	emit(`let gidx = i32(id.x);`);
	emit("");
	emit("// Initialize carry from initCarry");
	for (let i = 0; i < numCarry; i++) {
		const carrySize = carrySizes[i] / elemSize;
		emit(`if (gidx < ${carrySize}) {`);
		emit(pushIndent);
		emit(`carry${i}[gidx] = initCarry${i}[gidx];`);
		emit(popIndent, "}");
	}
	emit("");
	emit(`// Main scan loop over ${length} iterations`);
	emit(`for (var iter: u32 = 0u; iter < ${length}u; iter++) {`, pushIndent);
	if (reverse) emit(`let dataIdx = ${length - 1}u - iter;`);
	else emit(`let dataIdx = iter;`);
	for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
		const step = steps[stepIdx];
		const kernel = step.kernel;
		const tune = tuneNullopt(kernel);
		const carryIdx = step.outputCarryIdx;
		const kernelSize = kernel.size;
		const ysElemStride = ysStrides[carryIdx] / elemSize;
		emit("");
		emit(`// Step ${stepIdx}: kernel writes to carry${carryIdx}`);
		emit(`if (gidx < ${kernelSize}) {`);
		emit(pushIndent);
		const re = kernel.reduction;
		if (re) {
			const accTy = dtypeToWgsl(re.dtype, true);
			emit(`var acc: ${accTy} = ${constToWgsl(re.dtype, re.identity)};`);
			emit(`for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`, pushIndent);
			const expCode = genScanExpressionWithRidx(tune.exp, dtype, numConsts, numCarry);
			emit(`let val = ${expCode};`);
			if (re.op === AluOp.Add) emit(`acc = acc + val;`);
			else if (re.op === AluOp.Mul) emit(`acc = acc * val;`);
			else if (re.op === AluOp.Min) emit(`acc = min(acc, val);`);
			else if (re.op === AluOp.Max) emit(`acc = max(acc, val);`);
			else throw new Error(`Unsupported reduction op: ${re.op}`);
			emit(popIndent, "}");
			const epilogueCode = genScanExpressionWithRidx(tune.epilogue, dtype, numConsts, numCarry);
			emit(`let result_val_${stepIdx}: ${resultTy} = ${epilogueCode};`);
		} else {
			const expCode = genScanExpressionWithRidx(tune.exp, dtype, numConsts, numCarry);
			emit(`let result_val_${stepIdx}: ${resultTy} = ${expCode};`);
		}
		if (numY > 0 && carryIdx < numY) emit(`ys${carryIdx}[i32(dataIdx) * ${ysElemStride} + gidx] = result_val_${stepIdx};`);
		emit(`carry${carryIdx}[gidx] = result_val_${stepIdx};`);
		emit(popIndent, "}");
	}
	emit(popIndent, "}");
	emit(popIndent, "}");
	const numReadOnlyInputs = numConsts + numCarry + numX;
	const numReadWriteOutputs = numCarry + numY;
	return {
		code: getCode(),
		numInputs: numReadOnlyInputs,
		numOutputs: numReadWriteOutputs,
		hasUniform: false,
		passes: [{ grid: [gridX, gridY] }]
	};
}
function pipelineSubmit(device, pipelines, inputs, outputs) {
	const commandEncoder = device.createCommandEncoder();
	for (const { pipeline,...shader } of pipelines) {
		if (inputs.length !== shader.numInputs || outputs.length !== shader.numOutputs) throw new Error(`webgpu: expected ${shader.numInputs} inputs and ${shader.numOutputs} outputs, got ${inputs.length} inputs and ${outputs.length} outputs`);
		const filteredPasses = shader.passes.filter(({ grid }) => prod(grid) > 0);
		if (filteredPasses.length === 0) continue;
		const bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [...inputs.map((buffer, i) => ({
				binding: i,
				resource: { buffer }
			})), ...outputs.map((buffer, i) => ({
				binding: inputs.length + i,
				resource: { buffer }
			}))]
		});
		let uniformBindGroup = null;
		let uniformAlignment = 0;
		if (shader.hasUniform) {
			const uniforms = filteredPasses.map(({ uniform }) => uniform);
			const [uniformBuffer, alignment] = combineUniforms(device, uniforms);
			uniformAlignment = alignment;
			uniformBindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(1),
				entries: [{
					binding: 0,
					resource: {
						buffer: uniformBuffer,
						size: alignment
					}
				}]
			});
		}
		for (let i = 0; i < filteredPasses.length; i++) {
			const { grid } = filteredPasses[i];
			const passEncoder = commandEncoder.beginComputePass();
			passEncoder.setPipeline(pipeline);
			passEncoder.setBindGroup(0, bindGroup);
			if (uniformBindGroup) passEncoder.setBindGroup(1, uniformBindGroup, [i * uniformAlignment]);
			passEncoder.dispatchWorkgroups(grid[0], grid[1]);
			passEncoder.end();
		}
	}
	device.queue.submit([commandEncoder.finish()]);
}
function combineUniforms(device, uniforms) {
	for (const buf of uniforms) if (!buf || buf.byteLength === 0 || buf.byteLength !== uniforms[0].byteLength) throw new Error("webgpu: Uniform mismatch between shader passes");
	const minAlign = device.limits.minUniformBufferOffsetAlignment;
	const alignment = Math.ceil(uniforms[0].byteLength / minAlign) * minAlign;
	const buffer = device.createBuffer({
		size: alignment * uniforms.length,
		usage: GPUBufferUsage.UNIFORM,
		mappedAtCreation: true
	});
	const bufferMapped = new Uint8Array(buffer.getMappedRange());
	for (let i = 0; i < uniforms.length; i++) bufferMapped.set(uniforms[i], i * alignment);
	buffer.unmap();
	return [buffer, alignment];
}
/**
* A cache for compiled GPU compute pipelines, keyed by the shader source.
*
* This supports both async compilation (recommended) and a synchronous variant.
* If the pipeline is not in the cache, it will be compiled and added. For async
* compilation, only one compilation will be in progress at a time for a given
* shader source.
*/
var ShaderPipelineCache = class {
	cache;
	inProgress;
	constructor(device) {
		this.device = device;
		this.cache = /* @__PURE__ */ new Map();
		this.inProgress = /* @__PURE__ */ new Map();
	}
	#getLayout(shader) {
		if (shader.numInputs + shader.numOutputs > this.device.limits.maxStorageBuffersPerShaderStage) {
			const actual = shader.numInputs + shader.numOutputs;
			const max = this.device.limits.maxStorageBuffersPerShaderStage;
			throw new Error(`Too many buffers (${actual}) for WebGPU pipeline (max: ${max})`);
		}
		const bindGroupLayouts = [this.device.createBindGroupLayout({ entries: range(shader.numInputs + shader.numOutputs).map((i) => ({
			binding: i,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: i < shader.numInputs ? "read-only-storage" : "storage" }
		})) })];
		if (shader.hasUniform) bindGroupLayouts.push(this.device.createBindGroupLayout({ entries: [{
			binding: 0,
			visibility: GPUShaderStage.COMPUTE,
			buffer: {
				type: "uniform",
				hasDynamicOffset: true
			}
		}] }));
		return this.device.createPipelineLayout({ bindGroupLayouts });
	}
	async prepare(shader) {
		if (typeof globalThis.Deno !== "undefined") return this.prepareSync(shader);
		const existingPipeline = this.cache.get(shader.code);
		if (existingPipeline) return existingPipeline;
		const existingPromise = this.inProgress.get(shader.code);
		if (existingPromise) return await existingPromise;
		if (DEBUG >= 2) console.info("=========== WebGPU shader ===========\n" + shader.code);
		const shaderModule = this.device.createShaderModule({ code: shader.code });
		const promise = (async () => {
			this.device.pushErrorScope("validation");
			try {
				const pipeline$1 = await this.device.createComputePipelineAsync({
					layout: this.#getLayout(shader),
					compute: {
						module: shaderModule,
						entryPoint: "main"
					}
				});
				await this.device.popErrorScope();
				return pipeline$1;
			} catch (_error) {
				const scope = await this.device.popErrorScope();
				const emsg = await compileError(shaderModule, scope, shader.code);
				throw new Error(emsg);
			}
		})();
		this.inProgress.set(shader.code, promise);
		const pipeline = await promise;
		this.cache.set(shader.code, pipeline);
		return pipeline;
	}
	prepareSync(shader) {
		const existingPipeline = this.cache.get(shader.code);
		if (existingPipeline) return existingPipeline;
		if (DEBUG >= 2) console.info("=========== WebGPU shader ===========\n" + shader.code);
		const shaderModule = this.device.createShaderModule({ code: shader.code });
		this.device.pushErrorScope("validation");
		const pipeline = this.device.createComputePipeline({
			layout: this.#getLayout(shader),
			compute: {
				module: shaderModule,
				entryPoint: "main"
			}
		});
		const errorScopePromise = this.device.popErrorScope();
		if (errorScopePromise && typeof errorScopePromise.then === "function") errorScopePromise.then(async (scope) => {
			if (scope !== null) {
				const emsg = await compileError(shaderModule, scope, shader.code);
				console.error(emsg);
			}
		});
		this.cache.set(shader.code, pipeline);
		return pipeline;
	}
};
/** Gather information about a compilation error and format it. */
async function compileError(shaderModule, scope, code) {
	let message = `Failed to compile shader: ${scope ? scope.message : "(no error scope)"}`;
	const info = await shaderModule.getCompilationInfo();
	for (const msg of info.messages) message += `\n  [${msg.type} at ${msg.lineNum}:${msg.linePos}] ${msg.message}`;
	if (code) message += `\n\n${code}`;
	return message;
}

//#endregion
export { WebGPUBackend };