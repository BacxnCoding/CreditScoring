# generate_sme_synthetic.py
# Usage:
#   python project-root/scripts/generate_sme_synthetic.py --outdir ./project-root/data --firms 400 --months 18 --target_default 0.08 --seed 42
import argparse, os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd, DateOffset

def make_applicants(n, rng):
    sectors = ["manufacturing","services","trade","construction","agriculture"]
    sector_probs = [0.22,0.32,0.28,0.10,0.08]
    regions = ["Tunis","Sfax","Sousse","Nabeul","Monastir","Ariana"]
    ownerships = ["private","family","partnership","public"]
    firm_ids = [f"F{100000+i}" for i in range(n)]
    return pd.DataFrame({
        "firm_id": firm_ids,
        "sector": rng.choice(sectors, n, p=sector_probs),
        "age": rng.integers(1, 30, n),
        "region": rng.choice(regions, n),
        "ownership": rng.choice(ownerships, n, p=[0.6,0.25,0.1,0.05]),
        "employees": rng.integers(2, 120, n)
    })

def random_dates(n, start, end, rng):
    delta = (end - start).days
    offs = rng.integers(0, delta + 1, n, dtype=np.int32)
    return pd.to_datetime(start) + pd.to_timedelta(offs, unit="D")

def make_loans(applicants, start_month, today, rng):
    n = len(applicants)
    app_dates = random_dates(n, start_month, today, rng)
    amounts = np.round(np.exp(rng.normal(9.2, 0.7, n)) / 10, 2)  # ~ right-skewed TND
    tenor = rng.integers(6, 36, n)
    secured = rng.choice([0,1], n, p=[0.6,0.4])
    price = np.round(rng.uniform(7.0, 18.0, n), 2)
    return pd.DataFrame({
        "loan_id": [f"L{200000+i}" for i in range(n)],
        "firm_id": applicants["firm_id"].values,
        "app_date": pd.to_datetime(app_dates).date,
        "amount": amounts,
        "tenor": tenor,
        "secured": secured,
        "price": price,
        "decision": 1  # assume approved for synthetic set
    })

def make_monthly_data(firm_ids, months, rng):
    rows_t, rows_v = [], []
    for f in firm_ids:
        base_sales = max(6000.0, rng.normal(40000, 12000))
        vol = abs(rng.normal(0.2, 0.08))
        purchase_ratio = float(rng.uniform(0.45, 0.75))
        bal = float(rng.normal(10000, 5000))
        seasonality = np.sin(np.linspace(0, 2*np.pi, len(months), endpoint=False)) * float(rng.uniform(0.05, 0.15))
        for i, m in enumerate(months):
            seasonal = 1 + seasonality[i]
            sales = max(2000.0, rng.normal(base_sales*seasonal, base_sales*vol*0.6))
            purchases = max(1000.0, sales*purchase_ratio*float(rng.uniform(0.9,1.1)))
            inflow = sales * float(rng.uniform(0.8,1.05))
            outflow = purchases * float(rng.uniform(0.85,1.05))
            bal = max(0.0, bal + inflow - outflow + float(rng.normal(0,2000)))

            rows_t.append({
                "firm_id": f,
                "date": (m.to_timestamp() + MonthEnd(0)).date(),
                "inflow": round(inflow, 2),
                "outflow": round(outflow, 2),
                "balance": round(bal, 2),
            })
            tax_paid = max(0.0, 0.19 * max(0.0, sales - purchases))
            rows_v.append({
                "firm_id": f,
                "month": str(m),  # YYYY-MM
                "sales": round(sales, 2),
                "purchases": round(purchases, 2),
                "tax_paid": round(tax_paid, 2),
            })
    return pd.DataFrame(rows_t), pd.DataFrame(rows_v)

def attach_outcomes(loans, applicants, transactions, target_default, rng):
    # Aggregate simple cashflow features
    cf = transactions.groupby("firm_id").apply(lambda d: pd.Series({
        "avg_net_cf": (d["inflow"]-d["outflow"]).mean(),
        "avg_balance": d["balance"].mean()
    })).reset_index()

    df = loans.merge(applicants[["firm_id","age","sector"]], on="firm_id", how="left") \
              .merge(cf, on="firm_id", how="left")

    # Base linear index (log-odds)
    z = (-2.45
         - 0.00006*df["avg_net_cf"].fillna(0)
         - 0.00003*df["avg_balance"].fillna(0)
         + 0.00001*df["amount"]
         - 0.02*df["secured"]
         - 0.02*np.log1p(df["age"]))

    sector_adj = df["sector"].map({
        "manufacturing": -0.05, "services": 0.0, "trade": 0.12,
        "construction": 0.08, "agriculture": 0.05
    }).fillna(0.0)
    z = z + sector_adj

    # Intercept calibration to hit target default rate
    # We shift z by delta so mean(sigmoid(z+delta)) ~= target_default (clipped)
    def rate(delta):
        p = 1/(1+np.exp(-(z + delta)))
        p = np.clip(p, 0.01, 0.35)
        return float(p.mean())

    lo, hi = -5.0, 5.0
    for _ in range(40):  # binary search on delta
        mid = (lo+hi)/2
        r = rate(mid)
        if r > target_default:
            hi = mid
        else:
            lo = mid
    delta = (lo+hi)/2
    pd_prob = 1/(1+np.exp(-(z + delta)))
    pd_prob = np.clip(pd_prob, 0.01, 0.35)

    defaults = rng.binomial(1, pd_prob)
    loans = loans.copy()
    loans["outcome"] = 1 - defaults  # 1=repaid, 0=default
    # crude default timing after app_date
    ddays = rng.integers(60, 540, size=len(loans))
    loans["default_date"] = np.where(defaults==1,
                                     pd.to_datetime(loans["app_date"]) + pd.to_timedelta(ddays, unit="D"),
                                     pd.NaT)
    loans["default_date"] = pd.to_datetime(loans["default_date"]).dt.date
    return loans, float(pd_prob.mean())

def make_bureau(applicants, loans, rng):
    n = len(applicants)
    arrears = rng.poisson(0.3, n)
    dpd = rng.choice(["0","1-30","31-60","61-90","90+"], n, p=[0.7,0.18,0.07,0.03,0.02])
    # exposure proportional to requested amount
    exposure = np.round(loans["amount"].to_numpy() * rng.uniform(0.2,1.2, n), 2)
    return pd.DataFrame({
        "firm_id": applicants["firm_id"].values,
        "arrears_count": arrears,
        "DPD_buckets": dpd,
        "existing_exposure": exposure
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./project-root/data")
    ap.add_argument("--firms", type=int, default=400)
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--target_default", type=float, default=0.08, help="Average default rate (0–1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Dates
    today = pd.Timestamp.today().normalize()
    start_month = (today - DateOffset(months=args.months-1)).to_period("M").to_timestamp()
    months = pd.period_range(start=start_month, end=today, freq="M")

    # Tables
    applicants = make_applicants(args.firms, rng)
    loans = make_loans(applicants, start_month, today, rng)
    transactions, vat = make_monthly_data(applicants["firm_id"].values, months, rng)
    loans, realized_default = attach_outcomes(loans, applicants, transactions, args.target_default, rng)
    bureau = make_bureau(applicants, loans, rng)

    # Save
    paths = {
        "applicants.csv": applicants,
        "loans.csv": loans,
        "transactions.csv": transactions,
        "vat_invoices.csv": vat,
        "bureau.csv": bureau,
    }
    for name, df in paths.items():
        df.to_csv(os.path.join(args.outdir, name), index=False)

    print(f"Done. Wrote to {os.path.abspath(args.outdir)}")
    print(f"Firms: {len(applicants)} | Months: {args.months} | Transactions rows: {len(transactions)} | VAT rows: {len(vat)}")
    print(f"Target default rate: {args.target_default:.3f} | Realized: {realized_default:.3f}")

if __name__ == "__main__":
    main()
