"""Tests for normalize_response_format() in rojnik.llm.schemas."""


import pytest
from pydantic import BaseModel

from rojnik.llm.schemas import normalize_response_format


class TestNormalizeResponseFormat:
    def test_none_returns_none(self):
        assert normalize_response_format(None) is None

    def test_dict_passthrough(self):
        fmt = {"type": "json_object"}
        result = normalize_response_format(fmt)
        assert result is fmt  # exact same object, not a copy

    def test_json_schema_dict_passthrough(self):
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": "Foo", "strict": True, "schema": {}},
        }
        assert normalize_response_format(fmt) is fmt

    def test_pydantic_model_produces_json_schema_dict(self):
        class MyOutput(BaseModel):
            answer: str
            count: int

        result = normalize_response_format(MyOutput)
        assert result is not None
        assert result["type"] == "json_schema"
        js = result["json_schema"]
        assert js["name"] == "MyOutput"
        assert js["strict"] is True
        schema = js["schema"]
        assert schema["type"] == "object"
        assert "answer" in schema["properties"]
        assert "count" in schema["properties"]

    def test_pydantic_model_name_matches_class_name(self):
        class SomeSpecificOutputShape(BaseModel):
            value: float

        result = normalize_response_format(SomeSpecificOutputShape)
        assert result is not None
        assert result["json_schema"]["name"] == "SomeSpecificOutputShape"

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported response_format"):
            normalize_response_format("json_object")  # type: ignore[arg-type]

    def test_invalid_int_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported response_format"):
            normalize_response_format(42)  # type: ignore[arg-type]
