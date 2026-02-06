import { DEBUG } from "./backend-BLx6HGHC.js";

//#region src/backend/webgpu/scan-wrapper.ts
/**
* Parse buffer declarations from WGSL source.
* Matches: @group(G) @binding(B) var<storage, ACCESS> NAME: array<TYPE>;
*/
function parseBufferBindings(code) {
	const regex = /@group\((\d+)\)\s+@binding\((\d+)\)\s+var<storage,\s*(read|read_write)>\s+(\w+)\s*:\s*array<([^>]+)>/g;
	const bindings = [];
	let match;
	while ((match = regex.exec(code)) !== null) bindings.push({
		group: parseInt(match[1]),
		binding: parseInt(match[2]),
		name: match[4],
		access: match[3],
		type: match[5]
	});
	return bindings;
}
/**
* Identify which bindings need offsets based on routine-to-scan mapping.
*
* Uses the routineInputJitIds and routineOutputJitIds to determine which
* routine bindings correspond to xs/ys (need offsets) vs const/carry (no offset).
*/
function getOffsetBindings(bindings, scanInfo) {
	const sorted = [...bindings].sort((a, b) => a.binding - b.binding);
	const inputs = sorted.filter((b) => b.access === "read");
	sorted.filter((b) => b.access === "read_write");
	const xsBindings = [];
	const ysBindings = [];
	const xsStart = scanInfo.numConsts + scanInfo.numCarry;
	for (let i = 0; i < inputs.length; i++) if (i < scanInfo.routineInputJitIds.length) {
		const jitId = scanInfo.routineInputJitIds[i];
		if (jitId >= xsStart) xsBindings.push(inputs[i]);
	}
	return {
		xsBindings,
		ysBindings
	};
}
/**
* Generate the ScanOffsets uniform struct declaration.
* Only includes xs/ys bindings that need offsets.
*/
function generateOffsetsStruct(xsBindings, ysBindings) {
	const fields = [];
	for (const b of xsBindings) fields.push(`  ${b.name}_offset: u32,`);
	for (const b of ysBindings) fields.push(`  ${b.name}_offset: u32,`);
	if (fields.length === 0) return "struct ScanOffsets { _pad: u32, }";
	return `struct ScanOffsets {\n${fields.join("\n")}\n}`;
}
/**
* Transform array accesses to include offset for specific bindings.
*/
function transformArrayAccesses(code, bindings) {
	let result = code;
	for (const binding of bindings) {
		const name = binding.name;
		const pattern = new RegExp(`\\b${name}\\s*\\[`, "g");
		let match;
		const replacements = [];
		while ((match = pattern.exec(result)) !== null) {
			const startBracket = match.index + match[0].length - 1;
			let depth = 1;
			let i = startBracket + 1;
			while (i < result.length && depth > 0) {
				if (result[i] === "[") depth++;
				if (result[i] === "]") depth--;
				i++;
			}
			if (depth !== 0) {
				if (DEBUG >= 1) console.warn(`Unmatched bracket in array access: ${name}`);
				continue;
			}
			const endBracket = i - 1;
			const indexExpr = result.substring(startBracket + 1, endBracket);
			const newAccess = `${name}[${name}_offset + (${indexExpr})]`;
			replacements.push({
				start: match.index,
				end: i,
				text: newAccess
			});
		}
		for (let r = replacements.length - 1; r >= 0; r--) {
			const rep = replacements[r];
			result = result.substring(0, rep.start) + rep.text + result.substring(rep.end);
		}
	}
	return result;
}
/**
* Find the start of the main() function body.
* Handles various WGSL main function signatures, including multi-line.
*/
function findMainBodyStart(code) {
	const mainMatch = code.match(/fn\s+main\s*\([\s\S]*?\)\s*(?:->[\s\S]*?)?\{/);
	if (!mainMatch || mainMatch.index === void 0) throw new Error("Could not find main() function in shader");
	return mainMatch.index + mainMatch[0].length;
}
/**
* Generate offset variable declarations to inject at start of main().
*/
function generateOffsetDeclarations(xsBindings, ysBindings) {
	const decls = [];
	for (const b of xsBindings) decls.push(`  let ${b.name}_offset = scan_offsets.${b.name}_offset;`);
	for (const b of ysBindings) decls.push(`  let ${b.name}_offset = scan_offsets.${b.name}_offset;`);
	return decls.join("\n");
}
/**
* Transform a routine shader for scan-aware dispatch.
*
* Only transforms xs/ys buffer accesses (not carry/consts) based on scan signature.
*
* @param shaderInfo The original shader info from createRoutineShader
* @param scanInfo Scan signature info identifying which bindings are xs/ys
* @returns A new ShaderInfo with transformed code and uniform binding for offsets
*/
function wrapRoutineForScan(shaderInfo, scanInfo) {
	const { code } = shaderInfo;
	const allBindings = parseBufferBindings(code);
	if (allBindings.length === 0) throw new Error("No buffer bindings found in shader");
	const { xsBindings, ysBindings } = getOffsetBindings(allBindings, scanInfo);
	if (xsBindings.length === 0 && ysBindings.length === 0) return shaderInfo;
	const offsetsStruct = generateOffsetsStruct(xsBindings, ysBindings);
	const uniformBinding = `@group(1) @binding(0) var<uniform> scan_offsets: ScanOffsets;`;
	const lastBindingMatch = [...code.matchAll(/@group\(0\)\s+@binding\(\d+\)\s+var<[^>]+>\s+\w+\s*:\s*[^;]+;/g)];
	if (lastBindingMatch.length === 0) throw new Error("No group(0) bindings found");
	const lastBinding = lastBindingMatch[lastBindingMatch.length - 1];
	const insertPoint = lastBinding.index + lastBinding[0].length;
	let newCode = code.substring(0, insertPoint);
	newCode += `\n\n${offsetsStruct}\n${uniformBinding}\n`;
	newCode += code.substring(insertPoint);
	const bindingsToTransform = [...xsBindings, ...ysBindings];
	newCode = transformArrayAccesses(newCode, bindingsToTransform);
	const mainStart = findMainBodyStart(newCode);
	const offsetDecls = generateOffsetDeclarations(xsBindings, ysBindings);
	if (offsetDecls) newCode = newCode.substring(0, mainStart) + "\n" + offsetDecls + "\n" + newCode.substring(mainStart);
	return {
		...shaderInfo,
		code: newCode,
		hasUniform: true
	};
}
/**
* Create a combined uniform buffer with offsets for ALL iterations.
* Each iteration's offsets are padded to minUniformBufferOffsetAlignment.
*
* This allows using dynamic uniform offsets for efficient iteration.
*
* @returns The combined buffer and the alignment (bytes between iterations)
*/
function createAllIterationsOffsetsBuffer(numX, numY, length, xsElemStrides, ysElemStrides, minAlignment, reverse) {
	const offsetsPerIter = numX + numY;
	const bytesPerIter = offsetsPerIter * 4;
	const alignment = Math.ceil(bytesPerIter / minAlignment) * minAlignment;
	const totalBytes = alignment * length;
	const buffer = new Uint8Array(totalBytes);
	const view = new DataView(buffer.buffer);
	for (let iter = 0; iter < length; iter++) {
		const baseOffset = iter * alignment;
		const xsIter = reverse ? length - 1 - iter : iter;
		for (let i = 0; i < numX; i++) view.setUint32(baseOffset + i * 4, xsIter * xsElemStrides[i], true);
		for (let i = 0; i < numY; i++) view.setUint32(baseOffset + (numX + i) * 4, iter * ysElemStrides[i], true);
	}
	return {
		buffer,
		alignment
	};
}

//#endregion
export { createAllIterationsOffsetsBuffer, wrapRoutineForScan };