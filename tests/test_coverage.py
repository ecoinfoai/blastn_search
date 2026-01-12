import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import io
import json
import yaml
import pandas as pd
from Bio import SeqIO
from src import blastn_search

class TestBlastnCoverage(unittest.TestCase):
    def setUp(self):
        # Setup common mock data
        self.mock_fasta = """>seq1
ATGCATGC
>seq2
CGTACGTA
"""
        self.mock_fasta_records = list(SeqIO.parse(io.StringIO(self.mock_fasta), "fasta"))

        # Patch logging to prevent clutter
        patcher = patch("src.blastn_search.logging")
        self.mock_logging = patcher.start()
        self.addCleanup(patcher.stop)

    def test_read_config_success(self):
        with patch("builtins.open", mock_open(read_data="user@example.com\nAPIKEY123\n")):
            with patch("os.path.exists", return_value=True):
                email, api_key = blastn_search.read_config("fake_config.txt")
                self.assertEqual(email, "user@example.com")
                self.assertEqual(api_key, "APIKEY123")

    def test_read_config_failure(self):
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(Exception):
                blastn_search.read_config("nonexistent.txt")

    @patch("src.blastn_search.NCBIWWW.qblast")
    @patch("src.blastn_search.NCBIXML.read")
    def test_blastn_success(self, mock_xml_read, mock_qblast):
        mock_qblast.return_value = io.StringIO("FAKE XML DATA")
        mock_record = MagicMock()
        mock_xml_read.return_value = mock_record

        result = blastn_search.blastn("ATGC")

        mock_qblast.assert_called_with("blastn", "nt", "ATGC")
        mock_xml_read.assert_called()
        self.assertEqual(result, mock_record)

    @patch("src.blastn_search.Entrez.efetch")
    @patch("src.blastn_search.Entrez.read")
    def test_get_taxa_info_success(self, mock_read, mock_efetch):
        # Mocking Entrez return structure
        mock_read.return_value = [
            {"LineageEx": [{"Rank": "species", "ScientificName": "Homo sapiens"}]}
        ]
        mock_efetch.return_value = io.StringIO("FAKE XML")

        # We must clear cache because other tests might have populated it
        blastn_search.get_taxa_info.cache_clear()

        result = blastn_search.get_taxa_info("9606")

        self.assertEqual(result, {"species": "Homo sapiens"})

    @patch("src.blastn_search.Entrez.efetch")
    def test_get_taxa_info_failure(self, mock_efetch):
        mock_efetch.side_effect = Exception("Network error")
        blastn_search.get_taxa_info.cache_clear()

        result = blastn_search.get_taxa_info("9999")

        self.assertIn("Error", result)

    def test_parse_results_no_alignments(self):
        mock_record = MagicMock()
        mock_record.alignments = []
        fasta = self.mock_fasta_records[0]

        results = blastn_search.parse_results(mock_record, fasta, 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["message"], "No results are available for this sequence.")

    @patch("src.blastn_search.get_taxa_info")
    def test_parse_results_with_hits(self, mock_get_taxa):
        mock_get_taxa.return_value = {"species": "Test Species"}

        mock_record = MagicMock()
        mock_alignment = MagicMock()
        mock_alignment.title = "gi|123|gb|ABC|taxid|999| Test Species Info"
        mock_alignment.length = 100
        mock_hsp = MagicMock()
        mock_hsp.expect = 0.0
        mock_hsp.identities = 90
        mock_hsp.align_length = 100
        mock_alignment.hsps = [mock_hsp]
        mock_record.alignments = [mock_alignment]

        fasta = self.mock_fasta_records[0]

        results = blastn_search.parse_results(mock_record, fasta, 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["taxa_info"], {"species": "Test Species"})

    def test_save_results_formats(self):
        results = [{"test": "data"}]

        with patch("builtins.open", mock_open()) as mock_file:
            # JSON
            with patch("json.dump") as mock_json:
                blastn_search.save_results(results, "json")
                mock_json.assert_called()

            # YAML
            with patch("yaml.dump") as mock_yaml:
                blastn_search.save_results(results, "yaml")
                mock_yaml.assert_called()

            # CSV
            with patch("pandas.DataFrame.to_csv") as mock_csv:
                blastn_search.save_results(results, "csv")
                mock_csv.assert_called()

            # TSV
            with patch("pandas.DataFrame.to_csv") as mock_tsv:
                blastn_search.save_results(results, "tsv")
                mock_tsv.assert_called()

    def test_save_results_invalid_format(self):
        with self.assertRaises(ValueError):
            blastn_search.save_results([], "xml")

    @patch("src.blastn_search.blastn")
    @patch("src.blastn_search.parse_results")
    def test_blastn_and_parse_flow(self, mock_parse, mock_blastn):
        mock_blastn.return_value = "record"
        mock_parse.return_value = [{"res": 1}]

        with patch("src.blastn_search.SeqIO.parse", return_value=self.mock_fasta_records):
            with patch("builtins.open", mock_open(read_data=self.mock_fasta)):
                with patch("os.path.exists", return_value=True):
                    with patch("src.blastn_search.save_results", return_value="out.json") as mock_save:
                        # Patch os.remove to avoid errors (though not needed now as we removed intermediate cleanup)
                        file_name = blastn_search.blastn_and_parse("test.fas", 1)
                        self.assertEqual(file_name, "out.json")
                        mock_save.assert_called()

    def test_blastn_and_parse_file_not_found(self):
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                blastn_search.blastn_and_parse("missing.fas", 1)

    @patch("src.blastn_search.blastn")
    def test_parse_and_save_results_exception_handling(self, mock_blastn):
        # Create a future that raises an exception
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)

        # We need to simulate a future that raises an exception when result() is called
        def raising_func(seq):
            raise Exception("Blast failed")

        future = executor.submit(raising_func, "ATGC")

        # We need a dict {future: fasta_record}
        fasta = self.mock_fasta_records[0]
        step_futures = {future: fasta}

        # Run parse_and_save_results
        results = []
        blastn_search.parse_and_save_results(step_futures, results, 1)

        # Verify logging was called
        self.mock_logging.error.assert_called()
        self.assertEqual(len(results), 0)
        executor.shutdown()

if __name__ == '__main__':
    unittest.main()
