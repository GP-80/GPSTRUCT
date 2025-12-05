# us_density_choropleth_conus_rail_leaders_pretty_legend_v5.py
# ------------------------------------------------------------
# Adds: labels for the LIGHTEST class are colored with Alabama's fill color.
# Everything else (layout, legend, rails, title/footer) unchanged.
# ------------------------------------------------------------

import math
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import rgb2hex

# ---------------------- CONFIG ----------------------
CSV_INPUT   = "output.csv"  # A=state name or abbrev, D=density
SHP_URL     = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
OUTPUT_PNG  = "us_state_density_quantiles_blackbg_labels_20m.png"
K_BINS      = 7

DROP_CODES  = {"AK","HI","PR","GU","VI","MP","AS"}
EAST_RAIL_STATES = ["DC","DE","MD","NJ","CT","RI","MA","NH"]
VT_STRAIGHT_UP = True

TITLE = "Contiguous U.S. — Population Density, people per km²"
FOOTNOTE = "GP80 2025 — US Albers Equal Area — US Census Bureau"
# ----------------------------------------------------


def load_states(url: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(url)
    g = g[~g["STUSPS"].isin(DROP_CODES)].copy()
    return g.to_crs("EPSG:5070")


def load_density(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"A": str})
    if "A" not in df.columns or "D" not in df.columns:
        raise ValueError("CSV must have columns 'A' (state) and 'D' (density).")
    df = df.rename(columns={"A": "state", "D": "density"})
    df["state"] = df["state"].str.strip()
    df["density"] = pd.to_numeric(df["density"], errors="coerce")
    if df["density"].dropna().empty:
        raise ValueError("No valid numeric densities found in column 'D'.")
    q99 = df["density"].quantile(0.99)
    df["density_capped"] = np.minimum(df["density"], q99)
    return df[["state", "density", "density_capped"]]


def choose_merge_key(states: gpd.GeoDataFrame, df: pd.DataFrame) -> str:
    csv_vals = set(df["state"].str.upper())
    name_overlap = len(csv_vals & set(states["NAME"].str.upper()))
    abbr_overlap = len(csv_vals & set(states["STUSPS"].str.upper()))
    return "NAME" if name_overlap >= abbr_overlap else "STUSPS"


# ---------- Pretty legend helpers ----------
def _nice_floor(x: float) -> float:
    if x == 0:
        return 0.0
    s = 1 if x > 0 else -1
    x = abs(x)
    e = math.floor(math.log10(x))
    for m in (5, 2, 1):
        cand = m * 10**e
        if cand <= x:
            return s * cand
    return s * 10**(e - 1)

def _nice_ceil(x: float) -> float:
    if x == 0:
        return 0.0
    s = 1 if x > 0 else -1
    x = abs(x)
    e = math.floor(math.log10(x))
    for m in (1, 2, 5):
        cand = m * 10**e
        if cand >= x:
            return s * cand
    return s * 10**(e + 1)

def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    s = f"{x:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    if "." in s:
        a, b = s.split(".")
        return f"{int(a):,}.{b}"
    return f"{int(s):,}"

def classify_quantiles_with_pretty_labels(gdf: gpd.GeoDataFrame, k: int) -> list[str]:
    """
    Sets gdf['class_idx'] with exact quantile codes.
    Returns outward-rounded 1–2–5 legend labels (pretty only; bins unchanged).
    """
    cats = pd.qcut(gdf["density_capped"], q=k, duplicates="drop")
    gdf["class_idx"] = cats.cat.codes
    ivs = list(cats.cat.categories)

    pretty = []
    last_hi = None
    for iv in ivs:
        lo_true, hi_true = float(iv.left), float(iv.right)
        lo = _nice_floor(lo_true)
        hi = _nice_ceil(hi_true)
        if last_hi is not None and lo <= last_hi:
            lo = last_hi
        if lo >= hi:
            hi = lo + max(hi_true - lo_true, 1e-9)
        pretty.append((lo, hi))
        last_hi = hi

    labels = [f"{_fmt_num(lo)}–{_fmt_num(hi)}" for lo, hi in pretty]
    return labels
# ------------------------------------------


def representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    return gdf.representative_point()


def distribute_1d(sorted_targets, min_gap):
    out = []
    last = None
    for t in sorted_targets:
        y = t if last is None else max(t, last + min_gap)
        out.append(y)
        last = y
    return out


def make_plot(gdf: gpd.GeoDataFrame, labels: list[str], out_png: str, title: str):
    k = gdf["class_idx"].max() + 1
    cmap = plt.get_cmap("Blues", k)  # discrete light->dark

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150, facecolor="black")
    ax.set_facecolor("black")

    # Choropleth fills (quantiles)
    gdf.plot(column="density_capped", scheme="Quantiles", k=k, cmap=cmap, linewidth=0, legend=False, ax=ax)
    # Boundaries & silhouette
    gdf.boundary.plot(ax=ax, color="#bfbfbf", linewidth=0.6)
    gpd.GeoSeries([gdf.unary_union], crs=gdf.crs).boundary.plot(ax=ax, color="#e0e0e0", linewidth=1.0)

    # Legend (manual using pretty labels)
    handles = [Patch(facecolor=cmap(i), edgecolor="#ffffff", label=labels[i]) for i in range(k)]
    leg = ax.legend(
        handles=handles, title="People per km²",
        loc="lower right", frameon=True, framealpha=0.85,
        facecolor="black", edgecolor="#aaaaaa", fontsize=8, title_fontsize=9
    )
    plt.setp(leg.get_title(), color="white")
    for t in leg.get_texts():
        t.set_color("white")

    # Representative points for label anchors
    rp = representative_points(gdf)
    gdf["_x"], gdf["_y"] = rp.x.values, rp.y.values

    # --- Alabama fill color (to use for all LOWEST-CLASS labels) ---
    al_row = gdf[gdf["STUSPS"] == "AL"]
    if not al_row.empty and not pd.isna(al_row.iloc[0]["class_idx"]):
        al_idx = int(al_row.iloc[0]["class_idx"])
        al_color = rgb2hex(cmap(al_idx))
    else:
        al_color = rgb2hex(cmap(0))  # fallback

    # ---- East Coast rail positions ----
    xmin, ymin, xmax, ymax = gdf.total_bounds
    rail_x = xmax + 150_000.0  # ~150 km offshore

    rail_rows = gdf[gdf["STUSPS"].isin(EAST_RAIL_STATES)].copy().sort_values("_y", ascending=True)
    desired_y = rail_rows["_y"].to_list()
    slot_y = distribute_1d(desired_y, min_gap=120_000.0)
    rail_rows["_slot_y"] = slot_y
    rail_rows["_slot_x"] = rail_x
    rail_pos = {r["STUSPS"]: (r["_slot_x"], r["_slot_y"]) for _, r in rail_rows.iterrows()}

    # 1) Rail states with leaders
    for _, r in rail_rows.iterrows():
        lbl_color = al_color if r["class_idx"] == 0 else "white"
        ax.text(r["_slot_x"], r["_slot_y"], r["STUSPS"], ha="left", va="center",
                color=lbl_color, fontsize=8.5, fontweight="bold")
        ax.plot([r["_x"], r["_slot_x"]], [r["_y"], r["_slot_y"]],
                color="white", linewidth=0.8)

    # 2) VT straight-up leader (optional)
    if VT_STRAIGHT_UP:
        vt = gdf[gdf["STUSPS"] == "VT"]
        if not vt.empty:
            r = vt.iloc[0]
            lbl_color = al_color if r["class_idx"] == 0 else "white"
            x0, y0 = r["_x"], r["_y"]
            x1, y1 = x0, y0 + 150_000.0
            ax.plot([x0, x1], [y0, y1], color="white", linewidth=0.8)
            ax.text(x1, y1, "VT", ha="center", va="bottom",
                    color=lbl_color, fontsize=8.5, fontweight="bold")

    # 3) Everyone else: label at representative point
    excluded = set(rail_pos.keys()) | ({"VT"} if VT_STRAIGHT_UP else set())
    remain = gdf[~gdf["STUSPS"].isin(excluded)]
    for _, r in remain.iterrows():
        lbl_color = al_color if r["class_idx"] == 0 else "white"
        ax.text(r["_x"], r["_y"], r["STUSPS"], ha="center", va="center",
                color=lbl_color, fontsize=8.5, fontweight="bold")

    # Title and footer
    ax.set_title(title, fontsize=13, fontweight="bold", pad=32, color="white", loc="center")
    ax.text(
        0.01, 0.01, FOOTNOTE,
        transform=ax.transAxes,
        color="#aaaaaa", fontsize=7, ha="left", va="bottom", style="italic"
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote: {out_png}")
    plt.close(fig)


def main():
    # Load & merge
    states = load_states(SHP_URL)
    df = load_density(CSV_INPUT)

    merge_key = choose_merge_key(states, df)
    if merge_key == "NAME":
        states["_MERGE_KEY"] = states["NAME"].str.upper()
        df["_MERGE_KEY"] = df["state"].str.upper()
    else:
        states["_MERGE_KEY"] = states["STUSPS"].str.upper()
        df["_MERGE_KEY"] = df["state"].str.upper()

    gdf = states.merge(df, on="_MERGE_KEY", how="left")

    if gdf.empty:
        raise RuntimeError("Merged GeoDataFrame is empty. Check the input CSV and shapefile schema.")

    # Classify (sets class_idx) & build pretty labels
    labels = classify_quantiles_with_pretty_labels(gdf, k=K_BINS)

    # Plot & save
    make_plot(gdf, labels, OUTPUT_PNG, TITLE)


if __name__ == "__main__":
    main()
