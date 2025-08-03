from dnnlib.util import replace_placeholders

#----------------------------------------------------------------------------
# Compare on a variety of different datatypes

def test_nested_dict_replacement():
    data = {
        "a": {
            "b": "__fill_in_later_name",
            "c": {"d": "__fill_in_later_age"}
        }
    }

    fill_ins = {"name": "Alice", "age": 30}
    expected = {
        "a": {
            "b": "Alice",
            "c": {"d": 30}
        }
    }
    assert replace_placeholders(data, fill_ins) == expected

def test_list_handling():
    data = {
        "items": [
            "__fill_in_later_first",
            {"nested": "__fill_in_later_second"},
            [{"deep": "__fill_in_later_third"}]
        ]
    }
    fill_ins = {"first": 1, "second": 2, "third": 3}
    expected = {
        "items": [
            1,
            {"nested": 2},
            [{"deep": 3}]
        ]
    }
    assert replace_placeholders(data, fill_ins) == expected

def test_mixed_types():
    data = {
        "name": "__fill_in_later_name",
        "age": 25,
        "scores": [80, "__fill_in_later_math", 90],
        "metadata": {
            "temp": "__fill_in_later_temp",
            "active": False
        }
    }
    fill_ins = {"name": "Bob", "math": 95, "temp": 36.6}
    expected = {
        "name": "Bob",
        "age": 25,
        "scores": [80, 95, 90],
        "metadata": {
            "temp": 36.6,
            "active": False
        }
    }
    assert replace_placeholders(data, fill_ins) == expected

def test_deeply_nested():
    data = {"a": [1, {"b": [{"c": {"d": "__fill_in_later_value"}}]}]}
    fill_ins = {"value": "deep"}
    expected = {"a": [1, {"b": [{"c": {"d": "deep"}}]}]}
    assert replace_placeholders(data, fill_ins) == expected

#----------------------------------------------------------------------------
# Test that empty fill-ins leave data unaltered

def test_empty_fill_ins_leave_data_unaltered():
    data = {"key": "__fill_in_later_missing", "other": "unchanged"}
    fill_ins = {}  # No replacement provided
    expected = {"key": "__fill_in_later_missing", "other": "unchanged"}
    assert replace_placeholders(data, fill_ins) == expected

#----------------------------------------------------------------------------
# Test that it works with custom prefixes

def test_custom_prefix():
    data = {"key": "!!replace_me!!test"}
    fill_ins = {"test": "success"}
    expected = {"key": "success"}
    assert replace_placeholders(data, fill_ins, placeholder_prefix="!!replace_me!!") == expected
