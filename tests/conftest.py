import pytest
from fastapi.testclient import TestClient

from pd_scoring.service.app import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def good_row():
    return {
            "issue_d": "2018-01-01",
            "revenue": 40000.0,
            "dti_n": 21.66,
            "loan_amnt": 8800,
            "fico_n": 667.0,
            "experience_c": 1,
            "zip_code": "806xx",
            "emp_length": "10+ years",
            "purpose": "debt_consolidation",
            "home_ownership_n": "MORTGAGE",
            "addr_state": "CO"
            }
