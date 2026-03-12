from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

_import_structure = {
    "configuration_nova": ["NovaConfig"],
    "processing_nova": ["NovaProcessor"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_nova"] = [
        "NovaForConditionalGeneration",
        "NovaModel",
        "NovaPreTrainedModel",
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_nova"] = ["NovaImageProcessor"]

if TYPE_CHECKING:
    from .configuration_nova import NovaConfig
    from .processing_nova import NovaProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_nova import (
            NovaForConditionalGeneration,
            NovaModel,
            NovaPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_nova import NovaImageProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
