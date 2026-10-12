"""Package existing data for browsers isolated by anonymous-hosting sandboxes.

Classic scripts can load these static payloads without CORS or network APIs.
Base64 preserves the already-anonymized files byte for byte; it is not a
replacement for auditing the underlying JSON and decompressed response text.
"""
from pathlib import Path
import base64
import json

ROOT = Path(__file__).resolve().parents[1]


def build_payloads():
    docs = ROOT / 'docs'
    sources = sorted((docs / 'data').rglob('*.json'))
    sources += sorted((docs / 'data').rglob('*.json.gz'))
    for path in sources:
        key = path.relative_to(docs).as_posix()
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        body = f'globalThis.PRISM_DATA.register({json.dumps(key)},{json.dumps(encoded)});\n'
        path.with_name(path.name + '.js').write_text(body, encoding='ascii')
    print(f'Packaged {len(sources)} byte-preserving browser data payloads.')


if __name__ == '__main__':
    build_payloads()
