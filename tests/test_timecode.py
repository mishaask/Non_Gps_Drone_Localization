from gps_ex1.io.timecode import seconds_to_timecode


def test_seconds_to_timecode_basic():
    assert seconds_to_timecode(0) == "00:00:00.000"
    assert seconds_to_timecode(2.002) == "00:00:02.002"
    assert seconds_to_timecode(65.5) == "00:01:05.500"
    assert seconds_to_timecode(3661.234) == "01:01:01.234"


def test_seconds_to_timecode_empty_values():
    assert seconds_to_timecode(None) == ""
    assert seconds_to_timecode("") == ""
    assert seconds_to_timecode(-1) == ""
