from __future__ import annotations

"""Legacy compatibility shim for PyTorch runtime paths."""

from .cli.pytorch import (  # noqa: F401
    _build_arg_parser,
    _distill_model_from_file_impl,
    distill_model_from_file,
    handle_distill_request,
    handle_train_request,
    load_bundle,
    main,
    predict_rows,
    save_bundle,
    train_model_from_file,
)
