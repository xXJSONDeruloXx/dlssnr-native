"""Read the pinned user DLL without loading or executing its code."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct

MODEL_SHA256 = 'dcc0dc2414aedec4a8e084647070383be068554042587180c20c784d4772d36f'
WEIGHTS_SHA256 = '6bf8dc931ef3ccffe18c82de26ab374156e7f19539ffcf8eabaa25dca5cf15ab'
MODEL_VERSION = '310.8.0.0'


@dataclass(frozen=True)
class Tensor:
    name: str
    data: bytes


def read_model(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError('Unsupported NVIDIA DLL hash')
    resource = memoryview(data)[18129248:18129248 + 147695410]
    labels = list(re.finditer(rb'block[0-9]+\.layer[0-9]+\.(?:layer|blend_scale)', resource))
    if len(labels) != 153:
        raise ValueError('Unexpected tensor inventory')
    tensors, index, cursor = [], bytearray(), 0
    for label in labels:
        name = label.group()
        size = struct.unpack_from('<Q', resource, label.end() + 16)[0]
        start = label.end() + 28
        if start+size > len(resource):
            raise ValueError('Invalid tensor boundary')
        payload = bytes(resource[start:start+size])
        tensors.append(Tensor(name.decode(), payload))
        index += bytes([len(name)]) + name + struct.pack('<QQ', cursor, size)
        cursor += size
    digest = hashlib.sha256(struct.pack('<8sII', b'DLSSNRW1', len(tensors), 16+len(index)))
    digest.update(index)
    for tensor in tensors:
        digest.update(tensor.data)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError('Tensor payload digest mismatch')
    return tensors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dll', type=Path)
    args = parser.parse_args()
    tensors = read_model(args.dll)
    print(json.dumps({'version':MODEL_VERSION, 'tensors':len(tensors),
                      'bytes':sum(len(t.data) for t in tensors),
                      'sha256':WEIGHTS_SHA256, 'inference_executed':False}))
