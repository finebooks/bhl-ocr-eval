"""The generated axis catalogue must stay byte-for-byte current with the registry."""
import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_generated_axis_catalogue_is_current():
    script = ROOT / "scripts" / "gen_axis_catalogue.py"
    spec = importlib.util.spec_from_file_location("gen_axis_catalogue", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT / "AXES.md").read_text(encoding="utf-8") == module.render()
