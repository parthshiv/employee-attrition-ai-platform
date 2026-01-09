import pandas as pd

def engineer_features(df):
    """
    This function converts raw HR data into
    meaningful, behavior-based features.

    It is reused across:
    - model training
    - prediction
    - monitoring
    - explanations
    """

    # Always work on a copy to avoid modifying original data
    df = df.copy()

    # -----------------------------
    # FEATURE 1: TENURE BUCKET
    # -----------------------------
    # Convert continuous years_at_company
    # into human-understandable career stages, use cut()
    df["tenure_bucket"] = pd.cut(
        df["years_at_company"], # column from data
        bins=[0, 2, 5, 100], # it makes ranges like 0-2, 2-5, 5-100
        labels=["new", "mid", "senior"] # 0-2=new, 2-5=mid, 5-100=senior
    )

    # Convert categorical labels into numeric columns, use get_dummies()
    df = pd.get_dummies(
        df,
        columns=["tenure_bucket"],
        drop_first=True #To avoid dummy variable trap, reduces redundancy,improves numerical stability
    )
    
    # -----------------------------
    # FEATURE 2: SALARY RATIO
    # -----------------------------
    # formula: employee_salary / company_average_salary     
    # | salary_ratio | Meaning       |
    # | ------------ | ------------- |
    # | < 1.0        | Underpaid     |
    # | = 1.0        | Average       |
    # | > 1.0        | Above average |
    
    df['salary_ratio'] = df['salary'] / df['salary'].mean() #employee_salary / company_average_salary
    # df['salary'].mean(): Computes average salary of the entire dataset/whole comuln 'salary' of data


    return df
