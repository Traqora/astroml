import json
from pathlib import Path

import pytest

from astroml.ingestion.ledger_reader import LedgerBatchReader, LedgerReader, Page


@pytest.fixture
def ledger_dir(tmp_path):
    """Create a temporary directory with sample ledger files."""
    ledgers = [
        {"sequence": 100, "paging_token": "100_0", "total_coins": "100000000000"},
        {"sequence": 101, "paging_token": "101_0", "total_coins": "100000000001"},
        {"sequence": 102, "paging_token": "102_0", "total_coins": "100000000002"},
        {"sequence": 103, "paging_token": "103_0", "total_coins": "100000000003"},
        {"sequence": 104, "paging_token": "104_0", "total_coins": "100000000004"},
    ]
    for ledger in ledgers:
        f = tmp_path / f"ledger_{ledger['sequence']}.json"
        f.write_text(json.dumps(ledger))
    return tmp_path


@pytest.fixture
def ledger_dir_jsonl(tmp_path):
    """Create a temporary directory with JSONL ledger files."""
    ledgers = [
        {"sequence": 200, "paging_token": "200_0"},
        {"sequence": 201, "paging_token": "201_0"},
        {"sequence": 202, "paging_token": "202_0"},
    ]
    f = tmp_path / "ledger_batch.jsonl"
    f.write_text("\n".join(json.dumps(ledger) for ledger in ledgers))
    return tmp_path


class TestLedgerReader:
    def test_list_ledger_files(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        files = reader.list_ledger_files()
        assert len(files) == 5
        assert reader._extract_sequence(files[0].name) == 100

    def test_list_ledger_files_with_range(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        files = reader.list_ledger_files(start_seq=101, end_seq=103)
        assert len(files) == 3
        seqs = [reader._extract_sequence(f.name) for f in files]
        assert seqs == [101, 102, 103]

    def test_stream_file(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        records = list(reader.stream_file(ledger_dir / "ledger_100.json"))
        assert len(records) == 1
        assert records[0]["sequence"] == 100

    def test_stream_range(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        records = list(reader.stream_range(start_seq=101, end_seq=103))
        assert len(records) == 3
        assert [r["sequence"] for r in records] == [101, 102, 103]

    def test_stream_all(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        records = list(reader.stream_all())
        assert len(records) == 5

    def test_stream_file_not_found(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        with pytest.raises(FileNotFoundError):
            list(reader.stream_file(ledger_dir / "ledger_999.json"))

    def test_read_page_first_page(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        page = reader.read_page(page=1, page_size=2)
        assert isinstance(page, Page)
        assert len(page.records) == 2
        assert page.total == 5
        assert page.page == 1
        assert page.has_next is True
        assert page.has_prev is False

    def test_read_page_middle_page(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        page = reader.read_page(page=2, page_size=2)
        assert len(page.records) == 2
        assert page.has_next is True
        assert page.has_prev is True

    def test_read_page_last_page(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        page = reader.read_page(page=3, page_size=2)
        assert len(page.records) == 1
        assert page.has_next is False
        assert page.has_prev is True

    def test_read_page_invalid_page(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        with pytest.raises(ValueError, match="Page number must be >= 1"):
            reader.read_page(page=0, page_size=10)

    def test_read_page_invalid_page_size(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        with pytest.raises(ValueError, match="Page size must be >= 1"):
            reader.read_page(page=1, page_size=0)

    def test_count(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        assert reader.count() == 5
        assert reader.count(start_seq=101, end_seq=103) == 3

    def test_read_ledger_found(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        ledger = reader.read_ledger(102)
        assert ledger is not None
        assert ledger["sequence"] == 102

    def test_read_ledger_not_found(self, ledger_dir):
        reader = LedgerReader(str(ledger_dir))
        ledger = reader.read_ledger(999)
        assert ledger is None
    def test_read_ledger_partitioned_layout(self, tmp_path):
        # Place a ledger inside a bucket directory and ensure the reader finds it.
        bucket = tmp_path / 'ledger_bucket_00000000'
        bucket.mkdir(parents=True)
        (bucket / 'ledger_102.json').write_text('{"sequence": 102}')
        reader = LedgerReader(str(tmp_path))
        ledger = reader.read_ledger(102)
        assert ledger is not None
        assert ledger["sequence"] == 102

    def test_list_ledger_files_partitioned_layout(self, tmp_path):
        bucket = tmp_path / 'ledger_bucket_00010000'
        bucket.mkdir(parents=True)
        (bucket / 'ledger_10001.json').write_text('{"sequence": 10001}')
        (bucket / 'ledger_10002.json').write_text('{"sequence": 10002}')
        reader = LedgerReader(str(tmp_path))
        files = reader.list_ledger_files()
        assert len(files) == 2
        assert reader.count() == 2


    def test_empty_directory(self, tmp_path):
        reader = LedgerReader(str(tmp_path))
        assert reader.list_ledger_files() == []
        assert reader.count() == 0
        assert list(reader.stream_all()) == []

    def test_nonexistent_directory(self, tmp_path):
        reader = LedgerReader(str(tmp_path / "nonexistent"))
        assert reader.list_ledger_files() == []
        assert reader.count() == 0

    def test_extract_sequence_valid(self):
        assert LedgerReader._extract_sequence("ledger_42.json") == 42
        assert LedgerReader._extract_sequence("ledger_42.jsonl") == 42

    def test_extract_sequence_invalid(self):
        assert LedgerReader._extract_sequence("not_a_ledger.json") is None
        assert LedgerReader._extract_sequence("ledger_abc.json") is None

    def test_jsonl_streaming(self, ledger_dir_jsonl):
        reader = LedgerReader(str(ledger_dir_jsonl))
        records = list(reader.stream_file(ledger_dir_jsonl / "ledger_batch.jsonl"))
        assert len(records) == 3
        assert [r["sequence"] for r in records] == [200, 201, 202]


class TestLedgerBatchReader:
    def test_iter_batches(self, ledger_dir):
        batch_reader = LedgerBatchReader(str(ledger_dir), batch_size=2)
        batches = list(batch_reader.iter_batches())
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_iter_batches_with_range(self, ledger_dir):
        batch_reader = LedgerBatchReader(str(ledger_dir), batch_size=2)
        batches = list(batch_reader.iter_batches(start_seq=101, end_seq=103))
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_iter_batches_empty(self, tmp_path):
        batch_reader = LedgerBatchReader(str(tmp_path), batch_size=10)
        batches = list(batch_reader.iter_batches())
        assert batches == []
