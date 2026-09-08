"""Package tracked source and explicit native build products."""
import gzip
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parent
name = 'dlssnr-native-0.1.0-preview.1-linux-x86_64'
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
files = [p for p in files if p]
files += ['native/build/libdlssnr-native.so', 'native/build/dlssnr-inspect', 'native/build/matrix-layout.spv']
for binary in ('native/build/libdlssnr-native.so', 'native/build/dlssnr-inspect'):
    if (root / binary).read_bytes()[:4] != b'\x7fELF':
        raise SystemExit('A Linux ELF build is required: ' + binary)
dist = root / 'dist'
dist.mkdir(exist_ok=True)
archive = dist / (name + '.tar.gz')
with archive.open('wb') as output, gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as gz:
    with tarfile.open(fileobj=gz, mode='w') as tar:
        for path in sorted(files):
            data = (root / path).read_bytes()
            info = tarfile.TarInfo(name + '/' + path)
            info.size = len(data)
            info.mode = 0o755 if path == 'native/build/dlssnr-inspect' else 0o644
            tar.addfile(info, io.BytesIO(data))
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
(dist / 'SHA256SUMS').write_text(sha + '  ' + archive.name + '\n')
print(archive)
print(sha)
