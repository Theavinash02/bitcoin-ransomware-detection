import pytest

from ransomdet.config import ROOT
from ransomdet.readme import readme_block, results_block

REPORTS = ROOT / "reports"


@pytest.mark.skipif(not (REPORTS / "generalization.json").exists(), reason="reports not generated")
def test_readme_numbers_match_reports():
    assert readme_block(ROOT / "README.md") == results_block(REPORTS).strip()
