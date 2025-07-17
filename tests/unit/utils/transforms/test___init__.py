import importlib


def test_import_all_symbols():
    module = importlib.import_module("shok.utils.transforms")
    for name in module.__all__:
        assert hasattr(module, name), f"{name} not found in shok.utils.transforms"
