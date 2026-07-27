"""Tests for CLI formatters."""

import json
from astroml.cli_llm.formatters import output_result


class TestFormatters:
    def test_output_result_json(self, capsys):
        data = {"key": "value", "num": 42}
        output_result(data, as_json=True)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_output_result_text(self, capsys):
        output_result("hello", as_json=False)
        captured = capsys.readouterr()
        assert "hello" in captured.out
