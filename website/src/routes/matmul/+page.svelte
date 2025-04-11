<script lang="ts">
  const n = 2048;

  const benchmarks = [
    "naive",
    "shmem-tiling-basic",
    "unroll-4x4-vec4",
    "tinygrad",
  ] as const;

  let result: Record<string, number> = $state({});

  const randomBuffer = new Float32Array(
    [...new Array(n * n)].map(() => Math.random()),
  );

  const naiveKernel = `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

// Define the dimensions for the square matrices.
const DIM : u32 = 2048u;

// Helper function for reading from matrix A.
fn mm_readA(row: u32, col: u32) -> f32 {
  // Matrix A is stored in row-major order.
  return A[row * DIM + col];
}

// Helper function for reading from matrix B.
fn mm_readB(row: u32, col: u32) -> f32 {
  // Matrix B is stored in row-major order.
  return B[row * DIM + col];
}

// Helper function for writing the computed result to matrix C.
fn mm_write(row: u32, col: u32, value: f32) {
  C[row * DIM + col] = value;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // Compute the output coordinates.
  let row: u32 = global_id.y;
  let col: u32 = global_id.x;

  // If the workgroup size oversubscribes the matrix dimensions, ensure we don't access out of bounds.
  if (row >= DIM || col >= DIM) {
    return;
  }

  var sum: f32 = 0.0;
  // Compute the dot product for the (row, col) position.
  for (var k: u32 = 0u; k < DIM; k = k + 1u) {
    sum = sum + mm_readA(row, k) * mm_readB(k, col);
  }

  // Write the computed sum to the output matrix.
  mm_write(row, col, sum);
}
`;

  const shmemTilingKernel = `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

// Declare workgroup (shared) memory tiles for A and B.
var<workgroup> Asub: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> Bsub: array<array<f32, TILE_SIZE>, TILE_SIZE>;

// Define matrix dimensions.
const M : u32 = 2048u;  // Number of rows of A and C.
const N : u32 = 2048u;  // Number of columns of B and C.
const K : u32 = 2048u;  // Number of columns of A and rows of B.
const TILE_SIZE : u32 = 16u; // Workgroup tile size.

// Helper functions similar to tfjs codegen.
fn mm_readA(row: u32, col: u32) -> f32 {
  // Reads the element at A[row, col].
  return A[row * K + col];
}

fn mm_readB(row: u32, col: u32) -> f32 {
  // Reads the element at B[row, col].
  return B[row * N + col];
}

fn mm_write(row: u32, col: u32, value: f32) {
  // Writes the computed value to C[row, col].
  C[row * N + col] = value;
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>) {
  // Compute the output coordinate [row, col] for this thread.
  let row: u32 = global_id.y;
  let col: u32 = global_id.x;

  var sum: f32 = 0.0;

  // Loop over the tiles along the shared dimension.
  // Since K=2048 and TILE_SIZE=16, there are exactly 2048/16 = 128 tiles.
  for (var t: u32 = 0u; t < (K / TILE_SIZE); t = t + 1u) {
    // Each thread loads one element of A and one element of B into shared memory.
    // For A, load element at (row, t*TILE_SIZE + local_x).
    let aRow: u32 = row;
    let aCol: u32 = t * TILE_SIZE + local_id.x;
    Asub[local_id.y][local_id.x] = mm_readA(aRow, aCol);

    // For B, load element at (t*TILE_SIZE + local_y, col).
    let bRow: u32 = t * TILE_SIZE + local_id.y;
    let bCol: u32 = col;
    Bsub[local_id.y][local_id.x] = mm_readB(bRow, bCol);

    // Ensure the full tile has been loaded before computation.
    workgroupBarrier();

    // Compute the partial dot product for the tile.
    for (var k_inner: u32 = 0u; k_inner < TILE_SIZE; k_inner = k_inner + 1u) {
      sum = sum + Asub[local_id.y][k_inner] * Bsub[k_inner][local_id.x];
    }

    // Synchronize before loading the next tile.
    workgroupBarrier();
  }

  // Write the computed value to the output matrix C.
  mm_write(row, col, sum);
}`;

  const unroll4x4Kernel = `
// WGSL port of tinygrad's Metal kernel (r_64_32_8_16_512_4_4_4)
// Fixed for a particular shape and tuning; note that all dimensions and shifts are hardcoded.

@group(0) @binding(0)
var<storage, read_write> data0: array<f32>;

@group(0) @binding(1)
var<storage, read> data1: array<f32>;

@group(0) @binding(2)
var<storage, read> data2: array<f32>;

@compute @workgroup_size(8, 16)
fn main(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  // Metal kernel parameters:
  //   grid dims: (gidx0: 32, gidx1: 64) from gid.x, gid.y
  //   local dims: (lidx0: 8, lidx1: 16) from lid.x, lid.y
  let gidx0 = i32(gid.x); // expected: 32
  let gidx1 = i32(gid.y); // expected: 64
  let lidx0 = i32(lid.x); // expected: 8
  let lidx1 = i32(lid.y); // expected: 16

  // Precompute some shifted values (same as in Metal kernel)
  let alu0 = lidx1 << 2;      // lidx1 * 4
  let alu1 = lidx0 << 13;     // lidx0 * 8192
  let alu2 = gidx0 << 6;      // gidx0 * 64
  let alu3 = gidx1 << 16;     // gidx1 * 65536

  // Initialize 16 accumulators
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;
  var acc4: f32 = 0.0;
  var acc5: f32 = 0.0;
  var acc6: f32 = 0.0;
  var acc7: f32 = 0.0;
  var acc8: f32 = 0.0;
  var acc9: f32 = 0.0;
  var acc10: f32 = 0.0;
  var acc11: f32 = 0.0;
  var acc12: f32 = 0.0;
  var acc13: f32 = 0.0;
  var acc14: f32 = 0.0;
  var acc15: f32 = 0.0;

  // Loop over the reduction dimension (512 iterations)
  for (var ridx0: i32 = 0; ridx0 < 512; ridx0 = ridx0 + 1) {
    // Compute indices into data1 and data2:
    let alu4 = (ridx0 << 2) + alu3 + alu1; // offset into data1
    // Load a float4 from data1 at alu4:
    let val0 = vec4<f32>(
      data1[alu4 + 0],
      data1[alu4 + 1],
      data1[alu4 + 2],
      data1[alu4 + 3]
    );

    let alu5 = (ridx0 << 13) + alu2 + alu0; // offset into data2
    let val1 = vec4<f32>(
      data2[alu5 + 0],
      data2[alu5 + 1],
      data2[alu5 + 2],
      data2[alu5 + 3]
    );

    // Next three float4 loads from data1, offset by 2048, 4096, 6144
    let val2 = vec4<f32>(
      data1[alu4 + 2048 + 0],
      data1[alu4 + 2048 + 1],
      data1[alu4 + 2048 + 2],
      data1[alu4 + 2048 + 3]
    );
    let val3 = vec4<f32>(
      data1[alu4 + 4096 + 0],
      data1[alu4 + 4096 + 1],
      data1[alu4 + 4096 + 2],
      data1[alu4 + 4096 + 3]
    );
    let val4 = vec4<f32>(
      data1[alu4 + 6144 + 0],
      data1[alu4 + 6144 + 1],
      data1[alu4 + 6144 + 2],
      data1[alu4 + 6144 + 3]
    );

    // Similarly, three loads from data2, offset by 2048, 4096, 6144
    let val5 = vec4<f32>(
      data2[alu5 + 2048 + 0],
      data2[alu5 + 2048 + 1],
      data2[alu5 + 2048 + 2],
      data2[alu5 + 2048 + 3]
    );
    let val6 = vec4<f32>(
      data2[alu5 + 4096 + 0],
      data2[alu5 + 4096 + 1],
      data2[alu5 + 4096 + 2],
      data2[alu5 + 4096 + 3]
    );
    let val7 = vec4<f32>(
      data2[alu5 + 6144 + 0],
      data2[alu5 + 6144 + 1],
      data2[alu5 + 6144 + 2],
      data2[alu5 + 6144 + 3]
    );

    // Accumulate: note the pattern of multiplying components of the loaded vectors
    acc0  = acc0  + (val0.x * val1.x) + (val0.y * val5.x) + (val0.z * val6.x) + (val0.w * val7.x);
    acc1  = acc1  + (val2.x * val1.x) + (val2.y * val5.x) + (val2.z * val6.x) + (val2.w * val7.x);
    acc2  = acc2  + (val3.x * val1.x) + (val3.y * val5.x) + (val3.z * val6.x) + (val3.w * val7.x);
    acc3  = acc3  + (val4.x * val1.x) + (val4.y * val5.x) + (val4.z * val6.x) + (val4.w * val7.x);

    acc4  = acc4  + (val0.x * val1.y) + (val0.y * val5.y) + (val0.z * val6.y) + (val0.w * val7.y);
    acc5  = acc5  + (val2.x * val1.y) + (val2.y * val5.y) + (val2.z * val6.y) + (val2.w * val7.y);
    acc6  = acc6  + (val3.x * val1.y) + (val3.y * val5.y) + (val3.z * val6.y) + (val3.w * val7.y);
    acc7  = acc7  + (val4.x * val1.y) + (val4.y * val5.y) + (val4.z * val6.y) + (val4.w * val7.y);

    acc8  = acc8  + (val0.x * val1.z) + (val0.y * val5.z) + (val0.z * val6.z) + (val0.w * val7.z);
    acc9  = acc9  + (val2.x * val1.z) + (val2.y * val5.z) + (val2.z * val6.z) + (val2.w * val7.z);
    acc10 = acc10 + (val3.x * val1.z) + (val3.y * val5.z) + (val3.z * val6.z) + (val3.w * val7.z);
    acc11 = acc11 + (val4.x * val1.z) + (val4.y * val5.z) + (val4.z * val6.z) + (val4.w * val7.z);

    acc12 = acc12 + (val0.x * val1.w) + (val0.y * val5.w) + (val0.z * val6.w) + (val0.w * val7.w);
    acc13 = acc13 + (val2.x * val1.w) + (val2.y * val5.w) + (val2.z * val6.w) + (val2.w * val7.w);
    acc14 = acc14 + (val3.x * val1.w) + (val3.y * val5.w) + (val3.z * val6.w) + (val3.w * val7.w);
    acc15 = acc15 + (val4.x * val1.w) + (val4.y * val5.w) + (val4.z * val6.w) + (val4.w * val7.w);
  }

  // Compute final offset for writing results.
  let alu23 = alu0 + alu1 + alu2 + alu3;

  // Write results: four float4 values stored in data0 at offsets alu23, alu23+2048, alu23+4096, alu23+6144
  {
    let idx = alu23;
    let res = vec4<f32>(acc0, acc4, acc8, acc12);
    data0[idx + 0] = res.x;
    data0[idx + 1] = res.y;
    data0[idx + 2] = res.z;
    data0[idx + 3] = res.w;
  }
  {
    let idx = alu23 + 2048;
    let res = vec4<f32>(acc1, acc5, acc9, acc13);
    data0[idx + 0] = res.x;
    data0[idx + 1] = res.y;
    data0[idx + 2] = res.z;
    data0[idx + 3] = res.w;
  }
  {
    let idx = alu23 + 4096;
    let res = vec4<f32>(acc2, acc6, acc10, acc14);
    data0[idx + 0] = res.x;
    data0[idx + 1] = res.y;
    data0[idx + 2] = res.z;
    data0[idx + 3] = res.w;
  }
  {
    let idx = alu23 + 6144;
    let res = vec4<f32>(acc3, acc7, acc11, acc15);
    data0[idx + 0] = res.x;
    data0[idx + 1] = res.y;
    data0[idx + 2] = res.z;
    data0[idx + 3] = res.w;
  }
}`;

  const tinygradKernel = `@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn r_64_32_8_16_512_4_4_4(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = lidx1;
  var precast1 = (bitcast<u32>(precast0)<<2u);
  var cast0 = bitcast<i32>(precast1);
  var precast2 = lidx0;
  var precast3 = (bitcast<u32>(precast2)<<13u);
  var cast1 = bitcast<i32>(precast3);
  var precast4 = gidx0;
  var precast5 = (bitcast<u32>(precast4)<<6u);
  var cast2 = bitcast<i32>(precast5);
  var precast6 = gidx1;
  var precast7 = (bitcast<u32>(precast6)<<16u);
  var cast3 = bitcast<i32>(precast7);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var precast8 = ridx0;
    var cast4 = bitcast<u32>(precast8);
    var precast9 = (cast4<<2u);
    var precast10 = (cast4<<13u);
    var alu0 = (bitcast<i32>(precast9)+cast3+cast1);
    var val0 = data1[alu0];
    var alu1 = (bitcast<i32>(precast10)+cast2+cast0);
    var val1 = data2[alu1];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+2048)];
    var val6 = data1[(alu0+2049)];
    var val7 = data1[(alu0+2050)];
    var val8 = data1[(alu0+2051)];
    var val9 = data1[(alu0+4096)];
    var val10 = data1[(alu0+4097)];
    var val11 = data1[(alu0+4098)];
    var val12 = data1[(alu0+4099)];
    var val13 = data1[(alu0+6144)];
    var val14 = data1[(alu0+6145)];
    var val15 = data1[(alu0+6146)];
    var val16 = data1[(alu0+6147)];
    var val17 = data2[(alu1+1)];
    var val18 = data2[(alu1+2)];
    var val19 = data2[(alu1+3)];
    var val20 = data2[(alu1+2048)];
    var val21 = data2[(alu1+2049)];
    var val22 = data2[(alu1+2050)];
    var val23 = data2[(alu1+2051)];
    var val24 = data2[(alu1+4096)];
    var val25 = data2[(alu1+4097)];
    var val26 = data2[(alu1+4098)];
    var val27 = data2[(alu1+4099)];
    var val28 = data2[(alu1+6144)];
    var val29 = data2[(alu1+6145)];
    var val30 = data2[(alu1+6146)];
    var val31 = data2[(alu1+6147)];
    acc0 = (acc0+(val0*val1)+(val2*val20)+(val3*val24)+(val4*val28));
    acc1 = (acc1+(val5*val1)+(val6*val20)+(val7*val24)+(val8*val28));
    acc2 = (acc2+(val9*val1)+(val10*val20)+(val11*val24)+(val12*val28));
    acc3 = (acc3+(val13*val1)+(val14*val20)+(val15*val24)+(val16*val28));
    acc4 = (acc4+(val0*val17)+(val2*val21)+(val3*val25)+(val4*val29));
    acc5 = (acc5+(val5*val17)+(val6*val21)+(val7*val25)+(val8*val29));
    acc6 = (acc6+(val9*val17)+(val10*val21)+(val11*val25)+(val12*val29));
    acc7 = (acc7+(val13*val17)+(val14*val21)+(val15*val25)+(val16*val29));
    acc8 = (acc8+(val0*val18)+(val2*val22)+(val3*val26)+(val4*val30));
    acc9 = (acc9+(val5*val18)+(val6*val22)+(val7*val26)+(val8*val30));
    acc10 = (acc10+(val9*val18)+(val10*val22)+(val11*val26)+(val12*val30));
    acc11 = (acc11+(val13*val18)+(val14*val22)+(val15*val26)+(val16*val30));
    acc12 = (acc12+(val0*val19)+(val2*val23)+(val3*val27)+(val4*val31));
    acc13 = (acc13+(val5*val19)+(val6*val23)+(val7*val27)+(val8*val31));
    acc14 = (acc14+(val9*val19)+(val10*val23)+(val11*val27)+(val12*val31));
    acc15 = (acc15+(val13*val19)+(val14*val23)+(val15*val27)+(val16*val31));
  }
  var alu19 = (cast0+cast1+cast2+cast3);
  data0[alu19] = acc0;
  data0[(alu19+1)] = acc4;
  data0[(alu19+2)] = acc8;
  data0[(alu19+3)] = acc12;
  data0[(alu19+2048)] = acc1;
  data0[(alu19+2049)] = acc5;
  data0[(alu19+2050)] = acc9;
  data0[(alu19+2051)] = acc13;
  data0[(alu19+4096)] = acc2;
  data0[(alu19+4097)] = acc6;
  data0[(alu19+4098)] = acc10;
  data0[(alu19+4099)] = acc14;
  data0[(alu19+6144)] = acc3;
  data0[(alu19+6145)] = acc7;
  data0[(alu19+6146)] = acc11;
  data0[(alu19+6147)] = acc15;
}`;

  async function bench(variant: string) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      alert("WebGPU not supported");
      return;
    }
    const device = await adapter.requestDevice({
      requiredFeatures: ["timestamp-query"], // TODO
      requiredLimits: {
        maxComputeInvocationsPerWorkgroup:
          adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
        maxComputeWorkgroupStorageSize:
          adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    if (!device) {
      alert("Failed to create device");
      return;
    }

    const usage =
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST;
    const a = device.createBuffer({ size: n * n * 4, usage });
    const b = device.createBuffer({ size: n * n * 4, usage });
    const c = device.createBuffer({ size: n * n * 4, usage });
    const staging = device.createBuffer({
      size: n * n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    device.queue.writeBuffer(a, 0, randomBuffer);
    device.queue.writeBuffer(b, 0, randomBuffer);
    // await device.queue.onSubmittedWorkDone();

    try {
      if (variant === "naive") {
        const pipeline = await device.createComputePipelineAsync({
          compute: {
            module: device.createShaderModule({ code: naiveKernel }),
            entryPoint: "main",
          },
          layout: "auto",
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: a } },
            { binding: 1, resource: { buffer: b } },
            { binding: 2, resource: { buffer: c } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(n / 16, n / 16);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
      } else if (variant === "shmem-tiling-basic") {
        const pipeline = await device.createComputePipelineAsync({
          compute: {
            module: device.createShaderModule({ code: shmemTilingKernel }),
            entryPoint: "main",
          },
          layout: "auto",
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: a } },
            { binding: 1, resource: { buffer: b } },
            { binding: 2, resource: { buffer: c } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(n / 16, n / 16);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
      } else if (variant === "unroll-4x4-vec4") {
        const pipeline = await device.createComputePipelineAsync({
          compute: {
            module: device.createShaderModule({ code: unroll4x4Kernel }),
            entryPoint: "main",
          },
          layout: "auto",
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: c } },
            { binding: 1, resource: { buffer: a } },
            { binding: 2, resource: { buffer: b } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(32, 64);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
      } else if (variant === "tinygrad") {
        const pipeline = await device.createComputePipelineAsync({
          compute: {
            module: device.createShaderModule({ code: tinygradKernel }),
            entryPoint: "r_64_32_8_16_512_4_4_4",
          },
          layout: "auto",
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 1, resource: { buffer: c } },
            { binding: 2, resource: { buffer: a } },
            { binding: 3, resource: { buffer: b } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(32, 64);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
      } else {
        throw new Error("Unknown variant");
      }

      const start = performance.now();
      const commandEncoder = device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(c, 0, staging, 0, n * n * 4);
      device.queue.submit([commandEncoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ, 0, n * n * 4);
      const buf = new Float32Array(staging.getMappedRange());
      console.log(buf[0], buf[1], buf[2], buf[3], buf[n * n - 1]);
      staging.unmap(); // Do not need to actually read it.

      const time = performance.now() - start;
      result[variant] = time / 1000; // seconds
    } finally {
      a.destroy();
      b.destroy();
      c.destroy();
      staging.destroy();
    }
  }

  async function tfjsbench() {
    const tf = await import("@tensorflow/tfjs");
    await import("@tensorflow/tfjs-backend-webgpu");
    await tf.setBackend("webgpu");

    const a = tf.tensor(randomBuffer, [n, n]);
    const b = tf.tensor(randomBuffer, [n, n]);
    const start = performance.now();
    const c = tf.matMul(a, b);
    const ar = await c.data();
    console.log(ar[0], ar[1], ar[2], ar[3], ar[n * n - 1]);
    const time = performance.now() - start;
    result["tfjs"] = time / 1000; // seconds

    a.dispose();
    b.dispose();
    c.dispose();
  }
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">matmul benchmark</h1>

  <p class="mb-4">
    Running a few different WebGPU matmul programs on {n}x{n} matrices.
  </p>

  <div class="flex gap-x-4 mb-4">
    {#each benchmarks as variant}
      <button
        class="border px-2 hover:bg-gray-100 active:scale-95"
        onclick={() => bench(variant)}
      >
        {variant}
      </button>
    {/each}
    <button
      class="border px-2 hover:bg-gray-100 active:scale-95"
      onclick={tfjsbench}
    >
      tfjs-matmul
    </button>
  </div>

  {#each Object.entries(result) as [variant, time]}
    <div>
      <span class="font-bold">{variant}:</span>
      {time.toFixed(3)} seconds,
      {((2 * n * n * n) / 1e9 / time).toFixed(2)} GFLOP/s
    </div>
  {/each}
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border px-2 hover:bg-gray-100 active:scale-95;
  }
</style>
