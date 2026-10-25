"""Convert the raw scraped Pune listings into the schema train.py expects.

Input : data/Pune_property_data.csv   (raw)
Output: data/pune_house_data.csv      (size, total_sqft, bath, balcony, price, site_location)
"""
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/Pune_property_data.csv")
OUT = Path("data/pune_house_data.csv")


def parse_area(value):
    """'1,611 sq ft' -> 1611.0 (anything not in sq ft is dropped)."""
    s = str(value).lower().replace(",", "").strip()
    m = re.fullmatch(r"([\d.]+)\s*sq\s*ft", s)
    return float(m.group(1)) if m else None


def main():
    df = pd.read_csv(RAW)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    print(f"Raw rows: {len(df)}")

    # 1. Remove blank rows and header rows repeated inside the file
    df = df[df["bhk"].notna() & (df["bhk"] != "bhk")]
    print(f"After removing blank/repeated-header rows: {len(df)}")

    # 2. Remove exact duplicate listings
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)}")

    # 3. Keep apartments only (plots have no BHK; villas/houses price very differently)
    df = df[df["bhk"].str.contains("BHK Apartment", na=False)]
    print(f"Apartments only: {len(df)}")

    out = pd.DataFrame({
        "size": df["bhk"].str.extract(r"(\d+)\s*BHK")[0] + " BHK",
        "total_sqft": df["area"].apply(parse_area),
        "bath": pd.to_numeric(df["bathroom"], errors="coerce"),
        "balcony": pd.to_numeric(df["balconies"], errors="coerce"),   # NaN -> median in train.py
        "price": pd.to_numeric(df["price"], errors="coerce") / 100000,  # rupees -> lakhs
        "site_location": df["locality"].str.strip(),
    })

    out = out.dropna(subset=["size", "total_sqft", "bath", "price", "site_location"])
    print(f"After dropping unparseable rows: {len(out)}")

    # 4. Drop obviously bad values
    out = out[(out["price"] >= 5) & (out["total_sqft"].between(200, 10000))]
    print(f"Final rows: {len(out)}")

    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT}")
    print(out.describe().round(1).T[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
