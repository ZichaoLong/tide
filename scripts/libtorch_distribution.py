"""Read an explicit standalone LibTorch prefix without importing Python Torch."""
from pathlib import Path
import re
from build_environment import digest


def standalone(prefix, abi):
    prefix = Path(prefix).resolve()
    config = prefix / 'share/cmake/Torch/TorchConfig.cmake'
    version_file = config.with_name('TorchConfigVersion.cmake')
    version = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', version_file.read_text())
    if not version:
        raise ValueError('standalone TorchConfigVersion.cmake has no PACKAGE_VERSION')
    flags = re.findall(r'_GLIBCXX_USE_CXX11_ABI=([01])', config.read_text())
    if flags and any(int(flag) != abi for flag in flags):
        raise ValueError('requested ABI differs from standalone TorchConfig.cmake')
    files = [config, version_file, prefix/'lib/libtorch.so', prefix/'lib/libtorch_cpu.so', prefix/'lib/libc10.so']
    return dict(torch=version[1], torch_dir=str(config.parent), prefix=str(prefix),
                cxx11_abi=bool(abi), cuda=None, discovery='standalone-prefix',
                distribution_sha256={str(p.relative_to(prefix)): digest(p) for p in files})
