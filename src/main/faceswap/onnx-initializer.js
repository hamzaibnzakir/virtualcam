'use strict';

// Minimal ONNX (protobuf) reader: pulls one initializer tensor out of a model file.
// The face swapper keeps its identity-mapping matrix ("emap") as an initializer that
// the graph itself never uses, so ONNX Runtime can't hand it to us.

const ONNX_FLOAT = 1;
const ONNX_FLOAT16 = 10;

function readVarint(buf, pos) {
  let result = 0;
  let shift = 0;
  let byte;
  do {
    if (pos >= buf.length) throw new Error('Truncated varint');
    byte = buf[pos++];
    // Multiply instead of shifting so values above 2^31 stay correct.
    result += (byte & 0x7f) * 2 ** shift;
    shift += 7;
  } while (byte & 0x80);
  return [result, pos];
}

// Calls onField(fieldNumber, wireType, start, end) for every field in buf[start:end].
function walk(buf, start, end, onField) {
  let pos = start;
  while (pos < end) {
    let key;
    [key, pos] = readVarint(buf, pos);
    const field = Math.floor(key / 8);
    const wire = key & 7;
    if (wire === 0) {
      const begin = pos;
      [, pos] = readVarint(buf, pos);
      onField(field, wire, begin, pos);
    } else if (wire === 1) {
      onField(field, wire, pos, pos + 8);
      pos += 8;
    } else if (wire === 2) {
      let len;
      [len, pos] = readVarint(buf, pos);
      onField(field, wire, pos, pos + len);
      pos += len;
    } else if (wire === 5) {
      onField(field, wire, pos, pos + 4);
      pos += 4;
    } else {
      throw new Error(`Unsupported protobuf wire type ${wire}`);
    }
  }
}

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

function parseTensor(buf, start, end) {
  const tensor = { name: '', dims: [], dataType: 0, raw: null, floats: [], int32: [] };
  walk(buf, start, end, (field, wire, s, e) => {
    if (field === 1) {
      // dims: packed (wire 2) or single varint (wire 0)
      if (wire === 2) {
        let p = s;
        while (p < e) {
          let v;
          [v, p] = readVarint(buf, p);
          tensor.dims.push(v);
        }
      } else {
        tensor.dims.push(readVarint(buf, s)[0]);
      }
    } else if (field === 2) {
      tensor.dataType = readVarint(buf, s)[0];
    } else if (field === 4 && wire === 2) {
      for (let p = s; p < e; p += 4) tensor.floats.push(buf.readFloatLE(p));
    } else if (field === 5 && wire === 2) {
      let p = s;
      while (p < e) {
        let v;
        [v, p] = readVarint(buf, p);
        tensor.int32.push(v);
      }
    } else if (field === 8) {
      tensor.name = buf.toString('utf8', s, e);
    } else if (field === 9) {
      tensor.raw = buf.subarray(s, e);
    }
  });
  return tensor;
}

function toFloat32(tensor) {
  const count = tensor.dims.reduce((a, b) => a * b, 1);
  const out = new Float32Array(count);
  if (tensor.dataType === ONNX_FLOAT) {
    if (tensor.raw) {
      for (let i = 0; i < count; i += 1) out[i] = tensor.raw.readFloatLE(i * 4);
    } else {
      out.set(tensor.floats.slice(0, count));
    }
  } else if (tensor.dataType === ONNX_FLOAT16) {
    if (tensor.raw) {
      for (let i = 0; i < count; i += 1) out[i] = halfToFloat(tensor.raw.readUInt16LE(i * 2));
    } else {
      for (let i = 0; i < count; i += 1) out[i] = halfToFloat(tensor.int32[i] & 0xffff);
    }
  } else {
    throw new Error(`Unsupported tensor data type ${tensor.dataType}`);
  }
  return out;
}

// Returns every initializer's name and shape plus a loader for its data.
function listInitializers(buf) {
  const found = [];
  walk(buf, 0, buf.length, (field, wire, s, e) => {
    if (field !== 7 || wire !== 2) return; // ModelProto.graph
    walk(buf, s, e, (gField, gWire, gs, ge) => {
      if (gField !== 5 || gWire !== 2) return; // GraphProto.initializer
      found.push({ start: gs, end: ge });
    });
  });
  return found.map(({ start, end }) => {
    const t = parseTensor(buf, start, end);
    return { name: t.name, dims: t.dims, dataType: t.dataType, read: () => toFloat32(t) };
  });
}

// The swapper's emap is the last initializer and is 512x512.
function extractEmap(buf) {
  const inits = listInitializers(buf);
  for (let i = inits.length - 1; i >= 0; i -= 1) {
    const init = inits[i];
    if (init.dims.length === 2 && init.dims[0] === 512 && init.dims[1] === 512) {
      return init.read();
    }
  }
  throw new Error('This face swap model has no 512x512 identity matrix');
}

module.exports = { extractEmap, listInitializers, halfToFloat };
