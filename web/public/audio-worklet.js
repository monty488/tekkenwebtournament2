// Plays the runtime's stereo output ring (runtime/apu/output.cpp) straight out of wasm
// memory: { u32 write, u32 read, u32 capacity, u32 pad, f32 data[capacity * 2] }, indices in
// frames, wrapping freely. The runtime writes; this reads.
class TTT2Output extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { buffer, ptr, capacity } = options.processorOptions;
    this.indices = new Uint32Array(buffer, ptr, 4);
    this.data = new Float32Array(buffer, ptr + 16, capacity * 2);
    this.mask = capacity - 1;
  }

  process(inputs, outputs) {
    const [left, right] = outputs[0];
    const write = Atomics.load(this.indices, 0);
    let read = Atomics.load(this.indices, 1);
    const available = (write - read) >>> 0;
    const n = Math.min(left.length, available);
    for (let i = 0; i < n; i++, read++) {
      const at = (read & this.mask) * 2;
      left[i] = this.data[at];
      right[i] = this.data[at + 1];
    }
    // An underrun plays silence (the arrays start zeroed).
    Atomics.store(this.indices, 1, read >>> 0);
    return true;
  }
}

registerProcessor('ttt2-output', TTT2Output);
