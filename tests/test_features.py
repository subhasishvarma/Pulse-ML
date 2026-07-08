import pandas as pd
import pytest

from src.features.build_features import (
    compute_application_features,
    compute_bureau_features,
    compute_previous_loan_features,
)


def test_credit_income_ratio_handles_zero_income():
    df = pd.DataFrame({
        "SK_ID_CURR": [1], "AMT_INCOME_TOTAL": [0],
        "AMT_CREDIT": [10000], "AMT_ANNUITY": [500], "DAYS_EMPLOYED": [-200],
    })
    result = compute_application_features(df)
    assert pd.isna(result["credit_income_ratio"].iloc[0])


def test_annuity_income_ratio_normal_case():
    df = pd.DataFrame({
        "SK_ID_CURR": [1], "AMT_INCOME_TOTAL": [100000],
        "AMT_CREDIT": [200000], "AMT_ANNUITY": [10000], "DAYS_EMPLOYED": [-500],
    })
    result = compute_application_features(df)
    assert result["annuity_income_ratio"].iloc[0] == pytest.approx(0.1)
    assert result["credit_income_ratio"].iloc[0] == pytest.approx(2.0)


def test_application_features_negative_days_employed_preserved():
    df = pd.DataFrame({
        "SK_ID_CURR": [1], "AMT_INCOME_TOTAL": [50000],
        "AMT_CREDIT": [100000], "AMT_ANNUITY": [5000], "DAYS_EMPLOYED": [-9999],
    })
    result = compute_application_features(df)
    assert result["days_employed"].iloc[0] == -9999


def test_bureau_features_no_prior_loans_gives_empty_frame():
    empty = pd.DataFrame(columns=["SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT_OVERDUE",
                                   "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"])
    result = compute_bureau_features(empty)
    assert len(result) == 0


def test_bureau_features_debt_to_credit_ratio_zero_credit_is_null():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1], "CREDIT_ACTIVE": ["Active", "Closed"],
        "DAYS_CREDIT_OVERDUE": [0, 5],
        "AMT_CREDIT_SUM": [0, 0], "AMT_CREDIT_SUM_DEBT": [100, 50],
    })
    result = compute_bureau_features(df)
    assert pd.isna(result["total_debt_to_credit_ratio"].iloc[0])


def test_bureau_features_counts_active_loans_correctly():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1], "CREDIT_ACTIVE": ["Active", "Active", "Closed"],
        "DAYS_CREDIT_OVERDUE": [0, 0, 0],
        "AMT_CREDIT_SUM": [1000, 1000, 1000], "AMT_CREDIT_SUM_DEBT": [0, 0, 0],
    })
    result = compute_bureau_features(df)
    assert result["num_active_loans"].iloc[0] == 2
    assert result["num_prior_loans"].iloc[0] == 3


def test_previous_loan_features_approval_rate_no_applications_is_null():
    empty = pd.DataFrame(columns=["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT"])
    result = compute_previous_loan_features(empty)
    assert len(result) == 0


def test_previous_loan_features_approval_rate_computed_correctly():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1, 1], "NAME_CONTRACT_STATUS": ["Approved", "Approved", "Refused", "Cancelled"],
        "AMT_APPLICATION": [1000, 2000, 1500, 500], "AMT_CREDIT": [1000, 2000, 1500, 500],
    })
    result = compute_previous_loan_features(df)
    assert result["approval_rate"].iloc[0] == pytest.approx(0.5)
    assert result["num_approved"].iloc[0] == 2
    assert result["num_refused"].iloc[0] == 1


def test_previous_loan_features_all_refused():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1], "NAME_CONTRACT_STATUS": ["Refused", "Refused"],
        "AMT_APPLICATION": [1000, 2000], "AMT_CREDIT": [1000, 2000],
    })
    result = compute_previous_loan_features(df)
    assert result["approval_rate"].iloc[0] == 0.0


def test_application_features_missing_income_column_raises():
    df = pd.DataFrame({"SK_ID_CURR": [1], "AMT_CREDIT": [1000]})
    with pytest.raises(KeyError):
        compute_application_features(df)
