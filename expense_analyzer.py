
"""
Usage:-

python expense_analyzer.py --generate            
python expense_analyzer.py --csv my_expenses.csv 
python expense_analyzer.py --generate --show 
python expense_analyzer.py --csv your_filename.csv --show    
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Settings you can tweak
# --------------------------------------------------------------------------
CURRENCY = "$"

# Monthly budget per category. Categories not listed here are reported as
# "no budget" instead of being flagged.
BUDGETS = {
    "Rent": 1200,
    "Groceries": 350,
    "Food": 250,
    "Transport": 120,
    "Utilities": 200,
    "Entertainment": 100,
    "Shopping": 150,
    "Health": 80,
}


def money(x: float) -> str:
    return f"{CURRENCY}{x:,.2f}"


# --------------------------------------------------------------------------
# 1. Mock data
# --------------------------------------------------------------------------
def generate_mock_data(path: Path, month: str | None = None, seed: int = 42) -> None:
    """Write a realistic (and slightly messy) month of expenses to `path`.

    The mess is intentional: inconsistent casing, stray spaces, a "$" inside an
    amount, a duplicate row, a missing amount and a bad date. That gives the
    cleaning step something real to do.
    """
    rng = np.random.default_rng(seed)

    # Default to the previous full calendar month
    period = pd.Period(month, "M") if month else pd.Timestamp.today().to_period("M") - 1
    days = pd.date_range(period.start_time, period.end_time.normalize())

    rows = []

    def add(day, desc, cat, amt):
        rows.append({"date": day, "description": desc, "category": cat, "amount": round(float(amt), 2)})

    # Fixed monthly bills
    add(days[0], "Monthly rent", "Rent", 1200)
    add(days[4], "Electricity bill", "Utilities", rng.uniform(55, 95))
    add(days[7], "Internet", "Utilities", 45)
    add(days[11], "Phone plan", "Utilities", 30)
    add(days[9], "Netflix", "Entertainment", 15.49)
    add(days[14], "Spotify", "Entertainment", 10.99)

    # Everyday spending: (category, [descriptions], daily probability, (min, max))
    habits = [
        ("Food", ["Coffee", "Lunch", "Takeout dinner", "Pizza", "Brunch"], 0.60, (5, 28)),
        ("Groceries", ["Supermarket", "Farmers market", "Corner store"], 0.22, (20, 95)),
        ("Transport", ["Bus pass top-up", "Taxi", "Fuel", "Metro"], 0.35, (3, 35)),
        ("Entertainment", ["Cinema", "Game purchase", "Concert", "Bowling"], 0.12, (10, 60)),
        ("Shopping", ["Clothes", "Books", "Home goods", "Electronics"], 0.10, (20, 140)),
        ("Health", ["Pharmacy", "Gym day pass", "Doctor co-pay"], 0.05, (12, 75)),
    ]
    for day in days:
        for cat, descs, prob, (lo, hi) in habits:
            if rng.random() < prob:
                add(day, rng.choice(descs), cat, rng.uniform(lo, hi))

    # One big weekend so the "top spending days" report has a clear winner
    spike = days[min(19, len(days) - 1)]
    add(spike, "Concert tickets", "Entertainment", 140)
    add(spike, "Dinner out", "Food", 85)
    add(spike, "New sneakers", "Shopping", 120)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["amount"] = df["amount"].astype(object)  # allow strings like "$45.50" below

    # ---- Make it messy on purpose ----
    df.loc[3, "category"] = df.loc[3, "category"].lower()          # "food"
    df.loc[8, "category"] = "  " + df.loc[8, "category"] + "  "    # stray spaces
    df.loc[12, "amount"] = f"${df.loc[12, 'amount']}"              # "$12.34"
    df.loc[20, "amount"] = None                                    # missing amount
    df.loc[25, "date"] = "not a date"                              # bad date
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)          # exact duplicate row

    df.to_csv(path, index=False)
    print(f"Mock data written to {path}  ({len(df)} rows, {period})")


# --------------------------------------------------------------------------
# 2. Load & clean
# --------------------------------------------------------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    """Read the CSV and return a tidy DataFrame (date, description, category, amount)."""
    df = pd.read_csv(path, dtype=str)
    start_rows = len(df)

    # Normalise column names: " Amount " -> "amount"
    df.columns = df.columns.str.strip().str.lower()
    missing = {"date", "description", "category", "amount"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    # Text columns: trim whitespace, consistent Title Case, blank -> "Uncategorized"
    df["description"] = df["description"].fillna("").str.strip()
    df["category"] = (
        df["category"].fillna("").str.strip().str.title().replace("", "Uncategorized")
    )

    # Amount: strip "$", ",", spaces ... then convert; anything unparseable -> NaN
    df["amount"] = pd.to_numeric(
        df["amount"].str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
    )

    # Date: unparseable -> NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop unusable rows, duplicates, and non-positive amounts (refunds etc.)
    bad = df["date"].isna() | df["amount"].isna()
    n_bad = int(bad.sum())
    df = df[~bad]

    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    n_nonpos = int((df["amount"] <= 0).sum())
    df = df[df["amount"] > 0]

    print(
        f"Cleaning: {start_rows} rows read -> {len(df)} kept "
        f"({n_bad} invalid, {n_dupes} duplicate, {n_nonpos} non-positive removed)"
    )
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Analysis (groupby / aggregation)
# --------------------------------------------------------------------------
def spending_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Total, count, average and share of spending for each category."""
    out = (
        df.groupby("category")["amount"]
        .agg(total="sum", transactions="count", average="mean")
        .sort_values("total", ascending=False)
    )
    out["percent"] = out["total"] / out["total"].sum() * 100
    # Round only the money columns; keep percent at full precision and round it
    # at display time (rounding twice, e.g. 16.7466 -> 16.75 -> 16.8, is wrong).
    return out.round({"total": 2, "average": 2})


def daily_totals(df: pd.DataFrame) -> pd.Series:
    """Total spend per calendar day, including days with no spending (as 0)."""
    daily = df.groupby(df["date"].dt.normalize())["amount"].sum()
    return daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0.0)


def top_spending_days(daily: pd.Series, n: int = 3) -> pd.Series:
    return daily.nlargest(n)


def budget_report(by_cat: pd.DataFrame, budgets: dict) -> pd.DataFrame:
    """Compare actual spending with the budget for each category."""
    report = by_cat[["total"]].copy()
    report["budget"] = report.index.map(budgets)  # NaN if category has no budget
    report["remaining"] = report["budget"] - report["total"]
    report["status"] = np.select(
        [report["budget"].isna(), report["remaining"] < 0],
        ["no budget", "OVER"],
        default="ok",
    )
    return report.round(2)


# --------------------------------------------------------------------------
# 4. Charts (matplotlib)
# --------------------------------------------------------------------------
def plot_category_pie(by_cat: pd.DataFrame, outfile: Path, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.Set3(np.linspace(0, 1, len(by_cat)))

    wedges, _, autotexts = ax.pie(
        by_cat["total"],
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",  # hide labels on tiny slices
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)

    # Legend with the actual amounts is clearer than labels on the slices
    labels = [f"{cat}  -  {money(row.total)}" for cat, row in by_cat.iterrows()]
    ax.legend(wedges, labels, title="Category", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Spending by Category{title_suffix}\nTotal: {money(by_cat['total'].sum())}", fontsize=14)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")
    return fig


def plot_daily_trend(daily: pd.Series, top_days: pd.Series, outfile: Path, excluded=()) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(daily.index, daily.values, marker="o", markersize=4, linewidth=1.5,
            color="#4C72B0", label="Daily spend")
    ax.plot(daily.index, daily.rolling(7, min_periods=1).mean(), linewidth=2.5,
            color="#DD8452", linestyle="--", label="7-day average")

    # Highlight and label the top days
    ax.scatter(top_days.index, top_days.values, s=120, color="#C44E52", zorder=5,
               label="Top spending days")
    for day, amt in top_days.items():
        ax.annotate(f"{day:%b %d}\n{money(amt)}", (day, amt),
                    textcoords="offset points", xytext=(0, 14), ha="center", fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85},
                    zorder=6)

    note = f"  (excluding: {', '.join(excluded)})" if excluded else ""
    ax.set_title(f"Daily Spending Trend{note}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Amount spent ({CURRENCY})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 31, 3)))
    ax.margins(y=0.25)  # headroom for the labels
    ax.set_xlim(daily.index.min() - pd.Timedelta(days=1), daily.index.max() + pd.Timedelta(hours=12))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    return fig


def plot_budget_vs_actual(report: pd.DataFrame, outfile: Path) -> None:
    data = report.dropna(subset=["budget"])
    x = np.arange(len(data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, data["budget"], width, label="Budget", color="#9ecae1")
    actual_colors = np.where(data["status"] == "OVER", "#C44E52", "#55A868")
    ax.bar(x + width / 2, data["total"], width, label="Actual", color=actual_colors)

    ax.set_xticks(x, data.index, rotation=30, ha="right")
    ax.set_ylabel(f"Amount ({CURRENCY})")
    ax.set_title("Budget vs Actual (red = over budget)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    return fig


# --------------------------------------------------------------------------
# 5. Text report
# --------------------------------------------------------------------------
def print_report(df, by_cat, top_days, report, daily_df, excluded=()) -> None:
    line = "=" * 62
    print(f"\n{line}\n EXPENSE SUMMARY   {df['date'].min():%d %b %Y} - {df['date'].max():%d %b %Y}\n{line}")
    print(f" Total spent:        {money(df['amount'].sum())}")
    print(f" Transactions:       {len(df)}")
    print(f" Average per day:    {money(daily_totals(df).mean())}")

    print(f"\n SPENDING BY CATEGORY\n{'-' * 62}")
    for cat, r in by_cat.iterrows():
        print(f" {cat:<15}{money(r.total):>12}  {r.percent:>5.1f}%   ({int(r.transactions)} txns)")

    note = f"  (excluding: {', '.join(excluded)})" if excluded else ""
    print(f"\n TOP {len(top_days)} SPENDING DAYS{note}\n{'-' * 62}")
    for rank, (day, amt) in enumerate(top_days.items(), start=1):
        items = daily_df[daily_df["date"].dt.normalize() == day].sort_values("amount", ascending=False)
        biggest = ", ".join(f"{r.description} ({money(r.amount)})" for r in items.head(3).itertuples())
        print(f" {rank}. {day:%a %d %b}  {money(amt):>10}   {biggest}")

    print(f"\n BUDGET CHECK\n{'-' * 62}")
    for cat, r in report.iterrows():
        if r.status == "no budget":
            print(f" {cat:<15}{money(r.total):>12}   (no budget set)")
        else:
            flag = "OVER by " + money(-r.remaining) if r.status == "OVER" else money(r.remaining) + " left"
            print(f" {cat:<15}{money(r.total):>12} / {money(r.budget):<10} {flag}")
    print(line)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Personal Expense & Budget Analyzer")
    parser.add_argument("--csv", type=Path, default=Path("expenses.csv"), help="path to expenses CSV")
    parser.add_argument("--generate", action="store_true", help="create a mock dataset at --csv first")
    parser.add_argument("--month", help="month for mock data, e.g. 2026-09 (default: last month)")
    parser.add_argument("--outdir", type=Path, default=Path("reports"), help="folder for charts/CSV output")
    parser.add_argument("--top", type=int, default=3, help="how many top spending days to list")
    parser.add_argument("--exclude", nargs="*", default=["Rent"], metavar="CATEGORY",
                        help="categories to leave out of the daily trend / top days "
                             "(default: Rent). Pass --exclude with no values to include everything")
    parser.add_argument("--show", action="store_true", help="display charts in windows as well")
    args = parser.parse_args()

    if args.generate:
        generate_mock_data(args.csv, month=args.month)
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found. Use --generate to create a mock dataset.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(args.csv)
    by_cat = spending_by_category(df)
    report = budget_report(by_cat, BUDGETS)

    # Daily views can leave out big fixed bills (like rent) so the everyday
    # spending pattern is visible. Category totals and budgets still include them.
    excluded = [c.strip().title() for c in args.exclude]
    daily_df = df[~df["category"].isin(excluded)]
    daily = daily_totals(daily_df)
    top_days = top_spending_days(daily, args.top)

    print_report(df, by_cat, top_days, report, daily_df, excluded)

    # Save tables
    by_cat.to_csv(args.outdir / "category_summary.csv")
    report.to_csv(args.outdir / "budget_report.csv")

    # Save charts
    month_label = f" ({df['date'].min():%B %Y})"
    plot_category_pie(by_cat, args.outdir / "category_pie.png", month_label)
    plot_daily_trend(daily, top_days, args.outdir / "daily_trend.png", excluded)
    plot_budget_vs_actual(report, args.outdir / "budget_vs_actual.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()











