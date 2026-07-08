"""
Data dictionary for every engineered feature. This is the single source of
truth for "what does this column mean" -- point interviewers here instead of
re-deriving it from code, and reuse it to auto-generate README tables.
"""

FEATURE_DEFINITIONS = {
    # -- features_application --
    "amt_income_total": "Applicant's declared total annual income.",
    "amt_credit": "Total credit amount requested on this application.",
    "amt_annuity": "Loan annuity (periodic repayment amount).",
    "days_employed": "Days employed at time of application (negative = days before application date).",
    "credit_income_ratio": "amt_credit / amt_income_total. Higher = more leveraged relative to income.",
    "annuity_income_ratio": "amt_annuity / amt_income_total. Higher = larger repayment burden relative to income.",

    # -- features_bureau_history --
    "num_prior_loans": "Count of credit bureau records (loans from other institutions) for this applicant.",
    "num_active_loans": "Count of bureau records with CREDIT_ACTIVE == 'Active'.",
    "avg_days_credit_overdue": "Mean days overdue across all bureau records (0 if never overdue).",
    "total_debt_to_credit_ratio": "Sum(AMT_CREDIT_SUM_DEBT) / Sum(AMT_CREDIT_SUM) across bureau records.",

    # -- features_previous_loans --
    "num_previous_applications": "Count of previous loan applications with this lender.",
    "num_approved": "Count of previous applications with NAME_CONTRACT_STATUS == 'Approved'.",
    "num_refused": "Count of previous applications with NAME_CONTRACT_STATUS == 'Refused'.",
    "approval_rate": "num_approved / num_previous_applications. NULL if no previous applications.",
    "avg_amt_application": "Mean AMT_APPLICATION across previous applications.",
}
