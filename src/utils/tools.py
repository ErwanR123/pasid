import fsspec
from fsspec.implementations.arrow import HadoopFileSystem


def patch_fsspec():
    if "viewfs" not in fsspec.registry or "viewfs" not in fsspec.available_protocols():
        fsspec.register_implementation("viewfs", HadoopFileSystem)
