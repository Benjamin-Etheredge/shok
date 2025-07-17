import importlib


def test_import_all_symbols():
    module = importlib.import_module("shok.utils.functions")
    for name in module.__all__:
        assert hasattr(module, name), f"{name} not found in shok.utils.functions"


def test_symbol_types():
    module = importlib.import_module("shok.utils.functions")
    assert callable(getattr(module, "SoftRound", None)), "SoftRound should be callable"
    assert callable(getattr(module, "PassRound", None)), "PassRound should be callable"
    assert callable(getattr(module, "ScaleGrad", None)), "ScaleGrad should be callable"
