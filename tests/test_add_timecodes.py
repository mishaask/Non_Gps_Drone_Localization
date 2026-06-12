import csv

from gps_ex1.tools.add_timecodes import add_timecodes_to_csv


def test_add_timecodes_to_csv(tmp_path):
    src = tmp_path / "pred.csv"
    out = tmp_path / "pred_timecoded.csv"
    with src.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["query_time_s", "matched_reference_time_s", "pred_latitude"])
        writer.writeheader()
        writer.writerow({"query_time_s": "65.500", "matched_reference_time_s": "2.002", "pred_latitude": "32.1"})

    count = add_timecodes_to_csv(src, out)
    assert count == 1

    with out.open("r", newline="", encoding="utf-8") as fp:
        row = next(csv.DictReader(fp))

    assert row["query_timecode"] == "00:01:05.500"
    assert row["matched_reference_timecode"] == "00:00:02.002"
