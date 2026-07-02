# =============================================================
# FILE: data/generate_data.py
#
# PURPOSE:
#   Generate large, realistic dummy data for WealthWise Capital.
#   A wealth management firm with 50,000 clients.
#   This data feeds every other component — ETL, ML, APIs.
#
# OUTPUT:
#   data/clients.csv        (50,000 rows)
#   data/portfolios.csv     (~200,000 rows)
#   data/transactions.csv   (~500,000 rows)
#   data/fund_nav.csv       (~73,000 rows: 100 funds × 730 days)
#
# HOW TO RUN:
#   pip install pandas numpy faker
#   python data/generate_data.py
# =============================================================

# ── IMPORTS ──────────────────────────────────────────────────

import pandas as pd        # pandas: creates DataFrame (like Excel in Python)
                           # DataFrame = table with rows and columns
                           # We use it to build and save our CSV files

import numpy as np         # numpy: fast math library
                           # np.random.choice() = pick random item from list
                           # np.random.normal() = generate bell-curve numbers
                           # np.random.randint() = random whole number

import random              # Python's built-in random module
                           # random.uniform(a, b) = random decimal between a and b
                           # random.choice(list) = pick one item from list

from faker import Faker    # Faker: generates fake names, emails, phone numbers
                           # Saves us from making up 50,000 names manually

from datetime import date, timedelta, datetime
                           # date: represents a calendar date (2024-01-15)
                           # timedelta: represents a duration (30 days)
                           # datetime: date + time (2024-01-15 10:30:00)

import os                  # os: interact with the file system
                           # os.makedirs() creates folders if they don't exist

# ── CONFIGURATION ─────────────────────────────────────────────

# Faker with Indian locale — generates Indian names, phone numbers
fake = Faker('en_IN')

# Set a "seed" so the data is the same every time we run this script.
# Without a seed, you'd get different data each run.
# Seed = starting point for the random number generator.
random.seed(42)
np.random.seed(42)

# Total number of clients to generate
NUM_CLIENTS = 50_000       # Python allows underscores in numbers for readability
                           # 50_000 = 50000

# Output directory — where CSV files will be saved
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# exist_ok=True means: don't crash if the folder already exists


# ── REFERENCE DATA ─────────────────────────────────────────────
# These are the "lookup lists" — real values we pick from randomly

CITIES = [
    # Tier 1 — most wealth management clients are in metros
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    # Tier 2 — growing HNI (High Net Worth Individual) population
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kochi", "Chandigarh", "Indore", "Coimbatore"
]

# Weight = probability of each city
# Mumbai has 25% chance, Delhi 20%, etc.
# Must sum to 1.0
CITY_WEIGHTS = [
    0.25, 0.20, 0.15, 0.08, 0.07,
    0.06, 0.05, 0.04, 0.02, 0.02,
    0.01, 0.01, 0.01, 0.01, 0.02
]

# Risk profiles — how comfortable is the client with losing money?
RISK_PROFILES = ["Conservative", "Moderate", "Aggressive"]
# 30% Conservative (older, retired, risk-averse)
# 45% Moderate (working professionals, balanced)
# 25% Aggressive (young, high income, risk-tolerant)
RISK_WEIGHTS   = [0.30, 0.45, 0.25]

# KYC = Know Your Customer — regulatory requirement in India
KYC_STATUSES   = ["VERIFIED", "PENDING", "REJECTED"]
KYC_WEIGHTS    = [0.85, 0.12, 0.03]  # 85% verified (most clients pass KYC)

# Advisors — 50,000 clients, ~50 advisors = 1000 clients per advisor
ADVISOR_IDS = [f"ADV{i:03d}" for i in range(1, 51)]
# f"ADV{i:03d}" = format i as 3-digit padded number
# i=1 → "ADV001", i=50 → "ADV050"

# Mutual funds available on the platform
# Each fund has: name, type, base_nav (starting price), amc (fund house)
FUNDS = [
    # Equity funds — high risk, high return
    {"name": "Mirae Asset Large Cap",       "type": "Equity", "base_nav": 85.0,  "amc": "Mirae Asset",  "category": "Large Cap"},
    {"name": "Axis Bluechip Fund",          "type": "Equity", "base_nav": 52.0,  "amc": "Axis",         "category": "Large Cap"},
    {"name": "HDFC Flexi Cap Fund",         "type": "Equity", "base_nav": 120.0, "amc": "HDFC",         "category": "Flexi Cap"},
    {"name": "Parag Parikh Flexi Cap",      "type": "Equity", "base_nav": 67.0,  "amc": "PPFAS",        "category": "Flexi Cap"},
    {"name": "SBI Small Cap Fund",          "type": "Equity", "base_nav": 145.0, "amc": "SBI",          "category": "Small Cap"},
    {"name": "Nippon Small Cap",            "type": "Equity", "base_nav": 98.0,  "amc": "Nippon",       "category": "Small Cap"},
    {"name": "Kotak Emerging Equity",       "type": "Equity", "base_nav": 110.0, "amc": "Kotak",        "category": "Mid Cap"},
    {"name": "DSP Midcap Fund",             "type": "Equity", "base_nav": 88.0,  "amc": "DSP",          "category": "Mid Cap"},
    {"name": "ICICI Pru Technology Fund",   "type": "Equity", "base_nav": 175.0, "amc": "ICICI Pru",    "category": "Sectoral"},
    {"name": "Quant Active Fund",           "type": "Equity", "base_nav": 510.0, "amc": "Quant",        "category": "Multi Cap"},
    # Debt funds — low risk, stable returns
    {"name": "HDFC Short Term Debt",        "type": "Debt",   "base_nav": 25.0,  "amc": "HDFC",         "category": "Short Duration"},
    {"name": "ICICI Pru Corporate Bond",    "type": "Debt",   "base_nav": 28.0,  "amc": "ICICI Pru",    "category": "Corporate Bond"},
    {"name": "Axis Banking & PSU Debt",     "type": "Debt",   "base_nav": 22.0,  "amc": "Axis",         "category": "Banking & PSU"},
    {"name": "Kotak Gilt Fund",             "type": "Debt",   "base_nav": 88.0,  "amc": "Kotak",        "category": "Gilt"},
    {"name": "SBI Magnum Medium Duration",  "type": "Debt",   "base_nav": 42.0,  "amc": "SBI",          "category": "Medium Duration"},
    # Hybrid funds — mix of equity + debt
    {"name": "ICICI Pru Balanced Advantage","type": "Hybrid", "base_nav": 58.0,  "amc": "ICICI Pru",    "category": "Balanced Advantage"},
    {"name": "HDFC Balanced Advantage",     "type": "Hybrid", "base_nav": 350.0, "amc": "HDFC",         "category": "Balanced Advantage"},
    {"name": "Kotak Equity Hybrid",         "type": "Hybrid", "base_nav": 44.0,  "amc": "Kotak",        "category": "Aggressive Hybrid"},
    {"name": "Mirae Asset Hybrid Equity",   "type": "Hybrid", "base_nav": 24.0,  "amc": "Mirae Asset",  "category": "Aggressive Hybrid"},
    {"name": "Canara Robeco Hybrid",        "type": "Hybrid", "base_nav": 112.0, "amc": "Canara Robeco","category": "Aggressive Hybrid"},
    # Gold funds — inflation hedge
    {"name": "Nippon Gold BeES",            "type": "Gold",   "base_nav": 56.0,  "amc": "Nippon",       "category": "Gold ETF"},
    {"name": "HDFC Gold Fund",              "type": "Gold",   "base_nav": 20.0,  "amc": "HDFC",         "category": "Gold Fund of Fund"},
    {"name": "SBI Gold Fund",              "type": "Gold",   "base_nav": 19.0,  "amc": "SBI",           "category": "Gold Fund of Fund"},
]

# Group funds by type for easy lookup
# Result: {"Equity": [fund1, fund2, ...], "Debt": [...], ...}
FUNDS_BY_TYPE = {}
for fund in FUNDS:
    fund_type = fund["type"]
    if fund_type not in FUNDS_BY_TYPE:
        FUNDS_BY_TYPE[fund_type] = []
    FUNDS_BY_TYPE[fund_type].append(fund)

# Fund names as a simple list (used for quick random selection)
FUND_NAMES = [f["name"] for f in FUNDS]


# ── HELPER FUNCTIONS ──────────────────────────────────────────

def random_date(start_year: int, end_year: int) -> date:
    """
    Generate a random date between Jan 1 of start_year
    and Dec 31 of end_year.

    Args:
        start_year: earliest year (e.g., 2019)
        end_year:   latest year  (e.g., 2023)

    Returns:
        A random date object

    Why: We need clients to have different onboarding dates
         spread realistically over 4-5 years.
    """
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    delta = end - start          # timedelta: number of days between dates
    random_days = random.randint(0, delta.days)  # pick a random day count
    return start + timedelta(days=random_days)    # add days to start date


def get_fund_allocation(risk_profile: str) -> list:
    """
    Return a list of fund types matching the client's risk profile.

    Conservative → mostly Debt + some Gold + tiny Equity
    Moderate     → balanced mix
    Aggressive   → mostly Equity + some Hybrid

    Args:
        risk_profile: "Conservative" | "Moderate" | "Aggressive"

    Returns:
        List of fund types like ["Equity", "Debt", "Debt", "Gold"]

    Why: A Conservative client should NOT hold mostly small-cap equity.
         This makes our data realistic for ML training.
    """
    if risk_profile == "Conservative":
        # 70% debt, 20% hybrid, 10% gold — very safe
        pool = ["Debt"] * 7 + ["Hybrid"] * 2 + ["Gold"] * 1
    elif risk_profile == "Moderate":
        # 40% equity, 30% hybrid, 20% debt, 10% gold
        pool = ["Equity"] * 4 + ["Hybrid"] * 3 + ["Debt"] * 2 + ["Gold"] * 1
    else:  # Aggressive
        # 70% equity, 20% hybrid, 10% debt
        pool = ["Equity"] * 7 + ["Hybrid"] * 2 + ["Debt"] * 1

    # Pick 3 to 6 fund types from the pool
    num_funds = random.randint(3, 6)
    return random.choices(pool, k=num_funds)  # choices allows duplicates


def get_aum_by_risk(risk_profile: str) -> float:
    """
    Generate a realistic AUM (Assets Under Management) value.

    Conservative clients tend to be older with more wealth.
    Aggressive clients are often younger with moderate wealth.

    np.random.lognormal() generates right-skewed distribution
    — most clients have moderate wealth, few have very high wealth.
    This is realistic: most people have ₹10-50 lakh, few have ₹5 crore+

    Returns:
        AUM in INR (Indian Rupees)
    """
    if risk_profile == "Conservative":
        # Older, higher average wealth: mean around ₹30 lakh
        # np.random.lognormal(mean, sigma) — sigma controls spread
        return round(np.random.lognormal(mean=14.8, sigma=0.8), 2)
    elif risk_profile == "Moderate":
        # Working professionals: mean around ₹20 lakh
        return round(np.random.lognormal(mean=14.5, sigma=0.9), 2)
    else:  # Aggressive
        # Younger HNIs: mean around ₹15 lakh but high variance
        return round(np.random.lognormal(mean=14.2, sigma=1.1), 2)


# ── GENERATOR 1: CLIENTS ──────────────────────────────────────

def generate_clients(n: int) -> pd.DataFrame:
    """
    Generate n client records.

    Args:
        n: number of clients to generate (50,000)

    Returns:
        pandas DataFrame with all client columns

    Why: This is the master table. Every other table
         references client_id from here.
    """
    print(f"Generating {n:,} clients...")

    # Lists to collect data — faster than appending to DataFrame row by row
    rows = []

    for i in range(1, n + 1):
        # Progress update every 10,000 rows
        if i % 10_000 == 0:
            print(f"  ...{i:,} clients done")

        # client_id: CLT00001, CLT00002, ..., CLT50000
        # f-string with :05d = zero-pad to 5 digits
        client_id = f"CLT{i:05d}"

        # Pick risk profile using our probability weights
        risk_profile = np.random.choice(RISK_PROFILES, p=RISK_WEIGHTS)

        # fake.name() generates culturally appropriate Indian names
        # because we used Faker('en_IN') locale
        name = fake.name()

        # Generate email from the name (replace spaces with dots)
        email_name = name.lower().replace(" ", ".").replace("'", "")
        # Add a number to avoid duplicates
        email = f"{email_name}{random.randint(1, 999)}@{random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])}"

        # Indian phone numbers: 10 digits starting with 6-9
        phone = f"+91{random.randint(6000000000, 9999999999)}"

        # City picked by population weight (Mumbai most likely)
        city = np.random.choice(CITIES, p=CITY_WEIGHTS)

        # Age distribution: 25-75 years, most clients in 35-55 range
        # np.random.normal(mean, std) = bell curve
        age = int(np.clip(np.random.normal(42, 12), 25, 75))
        # np.clip() ensures age stays between 25 and 75

        # AUM based on risk profile
        aum = get_aum_by_risk(risk_profile)

        # Onboarding date: between 2019 and 2023
        onboarded_date = random_date(2019, 2023)

        # Assign to one of 50 advisors
        advisor_id = random.choice(ADVISOR_IDS)

        # KYC status (85% verified)
        kyc_status = np.random.choice(KYC_STATUSES, p=KYC_WEIGHTS)

        rows.append({
            "client_id":      client_id,
            "name":           name,
            "email":          email,
            "phone":          phone,
            "city":           city,
            "age":            age,
            "risk_profile":   risk_profile,
            "aum":            aum,
            "onboarded_date": onboarded_date,
            "advisor_id":     advisor_id,
            "kyc_status":     kyc_status,
        })

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(rows)
    print(f"  ✅ Clients: {len(df):,} rows")
    return df


# ── GENERATOR 2: PORTFOLIOS ───────────────────────────────────

def generate_portfolios(clients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate portfolio holdings for each client.

    Each client holds 3-6 mutual funds.
    Fund selection is based on their risk profile (realistic!).

    Args:
        clients_df: the DataFrame we just generated

    Returns:
        DataFrame with one row per fund holding
        (~200,000 rows for 50k clients with avg 4 funds each)
    """
    print("Generating portfolios...")
    rows = []
    portfolio_counter = 1   # for generating portfolio_id

    for idx, client in clients_df.iterrows():
        # Progress update
        if idx % 10_000 == 0 and idx > 0:
            print(f"  ...{idx:,} clients processed")

        # Get fund types appropriate for this client's risk profile
        fund_types = get_fund_allocation(client["risk_profile"])

        for fund_type in fund_types:
            # Pick a random fund of that type
            available_funds = FUNDS_BY_TYPE.get(fund_type, [])
            if not available_funds:
                continue
            fund = random.choice(available_funds)

            # portfolio_id: PRT000001, PRT000002, ...
            portfolio_id = f"PRT{portfolio_counter:06d}"
            portfolio_counter += 1

            # Number of units held (how many "shares" of the fund)
            # Varies by fund type and client AUM
            units = round(random.uniform(50, 1000), 3)

            # Buy price = a historical NAV (slightly lower than current)
            # We simulate that they bought it 1-3 years ago when it was cheaper
            discount_factor = random.uniform(0.7, 0.95)
            buy_price = round(fund["base_nav"] * discount_factor, 2)

            # Buy date: between client's onboarding and today
            onboarded = date.fromisoformat(str(client["onboarded_date"]))
            buy_date  = random_date(onboarded.year, 2024)

            # SIP = Systematic Investment Plan (monthly auto-investment)
            # 60% of holdings have an active SIP
            sip_active = random.random() < 0.60

            rows.append({
                "portfolio_id": portfolio_id,
                "client_id":    client["client_id"],
                "fund_name":    fund["name"],
                "fund_type":    fund["type"],
                "units":        units,
                "buy_price":    buy_price,
                "buy_date":     buy_date,
                "sip_active":   sip_active,
            })

    df = pd.DataFrame(rows)
    print(f"  ✅ Portfolios: {len(df):,} rows")
    return df


# ── GENERATOR 3: TRANSACTIONS ─────────────────────────────────

def generate_transactions(clients_df: pd.DataFrame,
                           funds: list) -> pd.DataFrame:
    """
    Generate transaction history for the past 12 months.

    Transaction types:
      BUY     — client invests a lump sum
      SELL    — client redeems (withdraws)
      SIP     — automated monthly investment
      DIVIDEND— fund pays dividend to client

    We also inject ~2% FRAUDULENT transactions:
      - Unusually large amounts
      - Odd hours (3am)
      - Multiple transactions same minute

    Args:
        clients_df: client DataFrame
        funds: list of fund dicts

    Returns:
        DataFrame with ~500,000 transaction rows
    """
    print("Generating transactions (this takes ~30 seconds)...")
    rows = []
    txn_counter = 1

    # Generate transactions for last 12 months
    today      = date.today()
    one_yr_ago = today - timedelta(days=365)

    # Pick a sample of clients who are "active" (not all 50k transact monthly)
    # 80% of clients had at least one transaction in the past year
    active_client_count = int(len(clients_df) * 0.80)
    active_clients = clients_df.sample(n=active_client_count, random_state=42)

    for idx, client in active_clients.iterrows():
        if idx % 10_000 == 0 and idx > 0:
            print(f"  ...{idx:,} clients processed for transactions")

        # Each active client makes 8-15 transactions per year
        num_transactions = random.randint(8, 15)

        for _ in range(num_transactions):
            txn_id = f"TXN{txn_counter:07d}"
            txn_counter += 1

            # Pick a random fund (doesn't have to be in their portfolio)
            fund = random.choice(funds)

            # Transaction type weighted probabilities
            # SIP is most common (monthly automatic)
            txn_type = np.random.choice(
                ["SIP", "BUY", "SELL", "DIVIDEND"],
                p=[0.55, 0.25, 0.15, 0.05]
            )

            # Random date in last 12 months
            days_ago  = random.randint(0, 365)
            txn_date  = datetime.combine(
                today - timedelta(days=days_ago),
                # Random time between 9am and 6pm (market hours)
                datetime.min.time()
            ).replace(
                hour=random.randint(9, 17),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )

            # NAV on transaction day
            # Slightly different from base NAV (markets fluctuate)
            nav_fluctuation = random.uniform(0.85, 1.20)
            nav_at_txn = round(fund["base_nav"] * nav_fluctuation, 2)

            # Amount based on transaction type
            if txn_type == "SIP":
                # SIP amounts: ₹1,000 to ₹50,000
                amount = round(random.choice([1000, 2000, 3000, 5000,
                                               10000, 25000, 50000]), 2)
            elif txn_type == "BUY":
                # Lump sum: ₹5,000 to ₹5,00,000
                amount = round(random.uniform(5000, 500000), 2)
            elif txn_type == "SELL":
                amount = round(random.uniform(5000, 200000), 2)
            else:  # DIVIDEND
                amount = round(random.uniform(500, 10000), 2)

            # Units = amount / NAV (how many fund units this amount buys)
            units = round(amount / nav_at_txn, 3)

            # Channel = how the transaction was placed
            channel = np.random.choice(
                ["APP", "WEB", "ADVISOR", "API"],
                p=[0.45, 0.30, 0.20, 0.05]
            )

            # Is this transaction fraudulent? (2% base rate)
            is_flagged = False

            rows.append({
                "txn_id":      txn_id,
                "client_id":   client["client_id"],
                "fund_name":   fund["name"],
                "txn_type":    txn_type,
                "amount":      amount,
                "units":       units,
                "nav_at_txn":  nav_at_txn,
                "txn_date":    txn_date,
                "channel":     channel,
                "is_flagged":  is_flagged,
            })

    # ── INJECT FRAUD TRANSACTIONS ──────────────────────────────
    # Inject ~500 fraudulent transactions so our fraud model has
    # positive examples to learn from.
    # Fraud patterns:
    #   1. Abnormally large amount (>₹10 lakh in one shot)
    #   2. Transaction at 2-4am (unusual time)
    #   3. API channel (programmatic, suspicious)
    print("  Injecting fraud transactions...")

    num_fraud = 500
    fraud_clients = clients_df.sample(n=num_fraud, random_state=99)

    for idx, client in fraud_clients.iterrows():
        txn_id = f"TXN{txn_counter:07d}"
        txn_counter += 1

        fund = random.choice(funds)
        nav_at_txn = round(fund["base_nav"] * random.uniform(0.9, 1.1), 2)

        # Fraud pattern: very large amount at night via API
        amount = round(random.uniform(1_000_000, 5_000_000), 2)  # ₹10-50 lakh
        units  = round(amount / nav_at_txn, 3)

        txn_date = datetime.combine(
            today - timedelta(days=random.randint(0, 365)),
            datetime.min.time()
        ).replace(
            hour=random.randint(2, 4),     # 2am-4am — very suspicious
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )

        rows.append({
            "txn_id":      txn_id,
            "client_id":   client["client_id"],
            "fund_name":   fund["name"],
            "txn_type":    "BUY",
            "amount":      amount,
            "units":       units,
            "nav_at_txn":  nav_at_txn,
            "txn_date":    txn_date,
            "channel":     "API",           # programmatic — suspicious
            "is_flagged":  True,            # ← FRAUD LABEL
        })

    df = pd.DataFrame(rows)
    # Shuffle so fraud rows aren't all at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  ✅ Transactions: {len(df):,} rows "
          f"({df['is_flagged'].sum()} fraud)")
    return df


# ── GENERATOR 4: FUND NAV HISTORY ─────────────────────────────

def generate_fund_nav(funds: list, days: int = 730) -> pd.DataFrame:
    """
    Generate 2 years of daily NAV history for all funds.

    How NAV changes daily:
      - Random walk: each day NAV = previous NAV × (1 + small random change)
      - Equity funds: more volatile (±2% daily)
      - Debt funds: very stable (±0.05% daily)
      - Gold: medium volatility (±1% daily)

    This simulates how real NAVs move over time.

    Args:
        funds: list of fund dicts (with base_nav)
        days:  how many days of history to generate (730 = 2 years)

    Returns:
        DataFrame with one row per fund per day
        (23 funds × 730 days = ~16,790 rows)
    """
    print(f"Generating {days} days of NAV history for {len(funds)} funds...")
    rows = []

    today     = date.today()
    start_date = today - timedelta(days=days)

    for fund in funds:
        current_nav = fund["base_nav"]

        # Volatility depends on fund type
        # sigma = standard deviation of daily % change
        if fund["type"] == "Equity":
            sigma = 0.012      # ±1.2% daily
            drift = 0.0004     # slight upward drift (markets grow over time)
        elif fund["type"] == "Debt":
            sigma = 0.0003     # ±0.03% daily (very stable)
            drift = 0.0002     # small positive drift
        elif fund["type"] == "Gold":
            sigma = 0.008      # ±0.8% daily
            drift = 0.0002
        else:  # Hybrid
            sigma = 0.007
            drift = 0.0003

        for day_num in range(days):
            nav_date = start_date + timedelta(days=day_num)

            # Skip weekends (markets closed Sat/Sun)
            # weekday(): 0=Monday, 5=Saturday, 6=Sunday
            if nav_date.weekday() >= 5:
                continue

            # Random walk formula:
            # new_nav = old_nav × (1 + drift + random_shock)
            daily_return = drift + np.random.normal(0, sigma)
            current_nav  = max(1.0, current_nav * (1 + daily_return))
            # max(1.0, ...) ensures NAV never goes below ₹1 (theoretical floor)

            rows.append({
                "fund_name": fund["name"],
                "nav":       round(current_nav, 4),
                "nav_date":  nav_date,
                "category":  fund["category"],
                "amc":       fund["amc"],
            })

    df = pd.DataFrame(rows)
    print(f"  ✅ Fund NAV: {len(df):,} rows")
    return df


# ── MAIN — ORCHESTRATE ALL GENERATORS ─────────────────────────

def main():
    """
    Main function — runs all generators in sequence and saves CSV files.

    Why separate functions?
      Each generator is independent and testable.
      We can regenerate only one table without touching others.
    """
    print("=" * 60)
    print("WealthWise Capital — Data Generator")
    print("=" * 60)

    # Step 1: Generate clients (everything else depends on this)
    clients_df = generate_clients(NUM_CLIENTS)

    # Step 2: Generate portfolios (needs client_id + risk_profile)
    portfolios_df = generate_portfolios(clients_df)

    # Step 3: Generate transactions (needs client_id + fund list)
    transactions_df = generate_transactions(clients_df, FUNDS)

    # Step 4: Generate NAV history (independent — just needs fund list)
    nav_df = generate_fund_nav(FUNDS, days=730)

    # ── SAVE TO CSV ───────────────────────────────────────────
    print("\nSaving CSV files...")

    clients_path = f"{OUTPUT_DIR}/clients.csv"
    clients_df.to_csv(clients_path, index=False)
    # index=False: don't write row numbers (0, 1, 2...) as a column
    print(f"  💾 Saved: {clients_path}")

    portfolios_path = f"{OUTPUT_DIR}/portfolios.csv"
    portfolios_df.to_csv(portfolios_path, index=False)
    print(f"  💾 Saved: {portfolios_path}")

    transactions_path = f"{OUTPUT_DIR}/transactions.csv"
    transactions_df.to_csv(transactions_path, index=False)
    print(f"  💾 Saved: {transactions_path}")

    nav_path = f"{OUTPUT_DIR}/fund_nav.csv"
    nav_df.to_csv(nav_path, index=False)
    print(f"  💾 Saved: {nav_path}")

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Clients:      {len(clients_df):>10,} rows")
    print(f"  Portfolios:   {len(portfolios_df):>10,} rows")
    print(f"  Transactions: {len(transactions_df):>10,} rows")
    print(f"  Fund NAV:     {len(nav_df):>10,} rows")
    print(f"\n  Risk breakdown:")
    risk_counts = clients_df["risk_profile"].value_counts()
    for profile, count in risk_counts.items():
        pct = count / len(clients_df) * 100
        print(f"    {profile:15s}: {count:,} ({pct:.1f}%)")
    print(f"\n  Total AUM: ₹{clients_df['aum'].sum() / 1e7:.1f} crore")
    print(f"  Avg AUM:   ₹{clients_df['aum'].mean() / 1e5:.1f} lakh")
    print(f"  Fraud txns: {transactions_df['is_flagged'].sum()}")
    print("=" * 60)


# ── ENTRY POINT ───────────────────────────────────────────────
# This block runs only when you execute the file directly:
#   python data/generate_data.py
#
# It does NOT run when another file imports this module.
# This is standard Python practice.
if __name__ == "__main__":
    main()