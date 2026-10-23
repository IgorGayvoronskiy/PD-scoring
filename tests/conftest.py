import pytest
from fastapi.testclient import TestClient

from pd_scoring.service.app import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def good_row():
    """Заявка только с обязательными признаками; необязательные не указаны."""
    return {
        "loan_amnt": 8800,
        "term": " 36 months",
        "emp_title": "teacher",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 40000.0,
        "verification_status": "Verified",
        "issue_d": "2018-01-01",
        "purpose": "debt_consolidation",
        "zip_code": "806xx",
        "addr_state": "CO",
        "dti": 21.66,
        "delinq_2yrs": 0,
        "earliest_cr_line": "Aug-2003",
        "fico_range_low": 665,
        "fico_range_high": 669,
        "open_acc": 8,
        "pub_rec": 0,
        "revol_bal": 12000,
        "revol_util": 54.3,
        "total_acc": 20,
        "application_type": "Individual",
        "acc_now_delinq": 0,
        "delinq_amnt": 0,
    }