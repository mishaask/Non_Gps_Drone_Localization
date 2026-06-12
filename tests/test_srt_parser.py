from gps_ex1.io.srt_parser import parse_srt_text, valid_gps_records


def test_parse_dji_srt_block():
    text = """1
00:00:00,000 --> 00:00:00,033
<font size=\"28\">SrtCnt : 1, DiffTime : 33ms
2024-06-19 11:01:31.198
[iso : 110] [shutter : 1/3000.0] [fnum : 170] [focal_len : 240] [latitude: 32.102538] [longitude: 35.209675] [rel_alt: 11.400 abs_alt: 696.441] </font>
"""
    records = parse_srt_text(text)
    assert len(records) == 1
    record = records[0]
    assert record.index == 1
    assert record.start_s == 0.0
    assert record.end_s == 0.033
    assert record.latitude == 32.102538
    assert record.longitude == 35.209675
    assert record.rel_alt == 11.4
    assert record.abs_alt == 696.441
    assert record.focal_len == 240.0
    assert record.has_valid_gps


def test_filter_zero_gps():
    text = """1
00:00:00,000 --> 00:00:00,033
[latitude: 0.000000] [longitude: 0.000000] [rel_alt: 0.000 abs_alt: 685.041]

2
00:00:00,033 --> 00:00:00,066
[latitude: 32.1] [longitude: 35.2] [rel_alt: 1.000 abs_alt: 686.000]
"""
    records = parse_srt_text(text)
    valid = valid_gps_records(records)
    assert len(valid) == 1
    assert valid[0].index == 2
