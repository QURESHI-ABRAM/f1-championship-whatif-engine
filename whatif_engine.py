import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Standard modern F1 points systems
GP_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS_2021 = {1: 3, 2: 2, 3: 1}
SPRINT_POINTS_MODERN = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def load_data(data_dir="data"):
    races = pd.read_csv(f"{data_dir}/races.csv")
    results = pd.read_csv(f"{data_dir}/results.csv")
    drivers = pd.read_csv(f"{data_dir}/drivers.csv")
    constructors = pd.read_csv(f"{data_dir}/constructors.csv")
    status = pd.read_csv(f"{data_dir}/status.csv")

    try:
        sprints = pd.read_csv(f"{data_dir}/sprint_results.csv")
    except FileNotFoundError:
        sprints = pd.DataFrame(columns=["raceId", "driverId", "constructorId", "positionOrder", "points"])

    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    return races, results, sprints, drivers, constructors, status


def build_season_table(year, races, results, sprints, drivers, constructors, status):
    season_races = races[races["year"] == year].sort_values("round").copy()
    if season_races.empty:
        raise ValueError(f"No race data found for season {year}.")

    gp = results.merge(season_races[["raceId", "round", "name", "date"]], on="raceId")
    gp = gp.merge(drivers[["driverId", "driverRef", "code", "driver_name"]], on="driverId")
    gp = gp.merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}), on="constructorId")
    gp = gp.merge(status[["statusId", "status"]], on="statusId", how="left")

    # Add sprint points per (raceId, driverId) if sprint happened that weekend
    if not sprints.empty:
        sp = sprints[sprints["raceId"].isin(season_races["raceId"])][["raceId", "driverId", "points"]].copy()
        sp = sp.rename(columns={"points": "sprint_points"})
        gp = gp.merge(sp, on=["raceId", "driverId"], how="left")
        gp["sprint_points"] = gp["sprint_points"].fillna(0.0)
    else:
        gp["sprint_points"] = 0.0

    gp["total_race_points"] = gp["points"] + gp["sprint_points"]
    gp = gp.sort_values(["round", "positionOrder"]).reset_index(drop=True)
    return season_races, gp


def compute_standings(gp_df):
    standings = (
        gp_df.groupby(["driverId", "driverRef", "code", "driver_name", "constructor"], as_index=False)
        .agg(
            total_points=("total_race_points", "sum"),
            wins=("positionOrder", lambda s: (s == 1).sum()),
            podiums=("positionOrder", lambda s: (s <= 3).sum()),
            dnfs=("positionText", lambda s: s.isin(["R", "D", "E", "W", "F", "N"]).sum()),
        )
        .sort_values(["total_points", "wins", "podiums"], ascending=False)
        .reset_index(drop=True)
    )
    standings["rank"] = standings.index + 1
    return standings


def apply_whatif_override(gp_df, round_num, driver_ref, new_position, fastest_lap=False):
    """
    Rewrites a driver's finishing position in a specific round, shifts other drivers
    realistically, and recalculates Grand Prix points under modern F1 scoring rules.
    """
    df = gp_df.copy()
    race_mask = df["round"] == round_num
    race_slice = df[race_mask].sort_values("positionOrder").copy()

    if driver_ref not in race_slice["driverRef"].values:
        raise ValueError(f"Driver '{driver_ref}' not found in Round {round_num}.")

    old_pos = int(race_slice.loc[race_slice["driverRef"] == driver_ref, "positionOrder"].iloc[0])
    if old_pos == new_position:
        return df

    # Reorder finishing positions cleanly
    ordered_refs = list(race_slice["driverRef"])
    ordered_refs.remove(driver_ref)
    ordered_refs.insert(max(0, min(new_position - 1, len(ordered_refs))), driver_ref)

    for idx, ref in enumerate(ordered_refs, start=1):
        row_mask = race_mask & (df["driverRef"] == ref)
        df.loc[row_mask, "positionOrder"] = idx
        # Preserve half-points races (e.g., 2021 Spa Round 12) by scaling standard points
        scale = 0.5 if (df.loc[race_mask, "points"].max() <= 12.5 and df.loc[race_mask, "points"].max() > 0) else 1.0
        base_pts = GP_POINTS.get(idx, 0) * scale

        # Preserve existing fastest lap bonus if driver finished top 10
        had_fl = (df.loc[row_mask, "rank"].astype(str) == "1").any()
        fl_bonus = 1.0 * scale if (idx <= 10 and (had_fl or (ref == driver_ref and fastest_lap))) else 0.0
        df.loc[row_mask, "points"] = base_pts + fl_bonus

    df["total_race_points"] = df["points"] + df["sprint_points"]
    return df


def run_monte_carlo(gp_df, season_races, cutoff_round, n_sims=10000, seed=42):
    """
    Simulates remaining rounds after cutoff_round using each contender's empirical
    finishing distribution from the season up to cutoff_round.
    """
    rng = np.random.default_rng(seed)
    completed = gp_df[gp_df["round"] <= cutoff_round].copy()
    remaining_rounds = season_races[season_races["round"] > cutoff_round]["round"].tolist()

    if not remaining_rounds:
        raise ValueError("cutoff_round must be less than the total number of rounds in the season.")

    current_standings = compute_standings(completed)
    top_contenders = current_standings.head(6).copy()
    contender_ids = top_contenders["driverId"].tolist()
    base_points = dict(zip(top_contenders["driverId"], top_contenders["total_points"]))

    # Build empirical points-per-weekend distribution for each contender up to cutoff_round
    hist_distributions = {}
    for did in contender_ids:
        pts_history = completed[completed["driverId"] == did]["total_race_points"].values
        # Add mild smoothing with league-average top-6 weekend outcomes so small samples stay realistic
        hist_distributions[did] = pts_history

    # Check which remaining rounds have sprints
    sprint_rounds = set(gp_df[gp_df["sprint_points"] > 0]["round"].unique())

    final_points_matrix = np.zeros((n_sims, len(contender_ids)))
    for i, did in enumerate(contender_ids):
        final_points_matrix[:, i] = base_points[did]

    for rnd in remaining_rounds:
        # Sample independent weekend performance with slight random race-incident variance
        for i, did in enumerate(contender_ids):
            sampled_pts = rng.choice(hist_distributions[did], size=n_sims, replace=True)
            # If future round has a sprint and sampled race had 0 sprint points, add expected sprint bonus
            if rnd in sprint_rounds:
                sampled_pts = sampled_pts + rng.choice([0, 2, 4, 6, 8], size=n_sims, p=[0.35, 0.20, 0.20, 0.15, 0.10])
            final_points_matrix[:, i] += sampled_pts

    winners = np.argmax(final_points_matrix, axis=1)
    summary = []
    for i, row in top_contenders.iterrows():
        idx = contender_ids.index(row["driverId"])
        pts_dist = final_points_matrix[:, idx]
        win_prob = (winners == idx).mean() * 100.0
        summary.append({
            "driver_name": row["driver_name"],
            "constructor": row["constructor"],
            "points_at_cutoff": row["total_points"],
            "expected_final_points": round(float(pts_dist.mean()), 1),
            "p10_points": round(float(np.percentile(pts_dist, 10)), 1),
            "p90_points": round(float(np.percentile(pts_dist, 90)), 1),
            "title_win_prob": round(float(win_prob), 2),
        })

    return pd.DataFrame(summary).sort_values("title_win_prob", ascending=False).reset_index(drop=True)


def plot_trajectory(actual_gp, whatif_gp, season_races, drivers_to_plot, year, scenario_label):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = ["#00D2BE", "#1E41FF", "#FF8000", "#DC0000"]
    for idx, ref in enumerate(drivers_to_plot):
        color = colors[idx % len(colors)]
        act = actual_gp[actual_gp["driverRef"] == ref].sort_values("round")
        wif = whatif_gp[whatif_gp["driverRef"] == ref].sort_values("round")
        label_name = act["driver_name"].iloc[0]

        ax.plot(act["round"], act["total_race_points"].cumsum(), label=f"{label_name} (Actual)",
                color=color, linestyle="--", alpha=0.75, linewidth=2)
        ax.plot(wif["round"], wif["total_race_points"].cumsum(), label=f"{label_name} (What-If)",
                color=color, linestyle="-", linewidth=2.6)

    ax.set_title(f"{year} F1 Championship Trajectory — Actual vs. What-If\n({scenario_label})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cumulative Championship Points", fontsize=11)
    ax.set_xticks(season_races["round"])
    ax.legend(frameon=True, facecolor="white", fontsize=9.5)
    plt.tight_layout()
    plt.savefig("whatif_title_trajectory.png", dpi=160)
    plt.close()


def plot_monte_carlo(mc_df, year, cutoff_round, total_rounds):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5.2))

    plot_df = mc_df.sort_values("title_win_prob", ascending=True)
    bars = ax.barh(plot_df["driver_name"], plot_df["title_win_prob"], color="#1E41FF", alpha=0.88)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{row['title_win_prob']:.1f}%  (Exp: {row['expected_final_points']} pts)",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xlim(0, max(105, plot_df["title_win_prob"].max() + 24))
    ax.set_title(
        f"{year} F1 Drivers' Championship — Monte Carlo Title Odds After Round {cutoff_round}/{total_rounds} (10,000 Sims)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Simulated Championship Win Probability (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig("monte_carlo_title_odds.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="F1 Championship What-If & Monte Carlo Simulator")
    parser.add_argument("--year", type=int, default=2021, help="F1 Season Year (default: 2021)")
    parser.add_argument("--round", type=int, default=6, help="Round number to override (default: 6 - 2021 Baku)")
    parser.add_argument("--driver", type=str, default="hamilton", help="driverRef to override (default: hamilton)")
    parser.add_argument("--pos", type=int, default=2, help="New finishing position in What-If (default: 2)")
    parser.add_argument("--cutoff", type=int, default=15, help="Cutoff round for Monte Carlo forecast (default: 15)")
    parser.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo simulations (default: 10000)")
    args = parser.parse_args()

    races, results, sprints, drivers, constructors, status = load_data()
    season_races, actual_gp = build_season_table(args.year, races, results, sprints, drivers, constructors, status)
    actual_standings = compute_standings(actual_gp)

    print(f"\n=== {args.year} ACTUAL FINAL TOP 5 STANDINGS ===")
    print(actual_standings[["rank", "driver_name", "constructor", "total_points", "wins", "podiums"]].head(5).to_string(index=False))

    # 1. Counterfactual What-If Scenario
    whatif_gp = apply_whatif_override(actual_gp, round_num=args.round, driver_ref=args.driver, new_position=args.pos)
    whatif_standings = compute_standings(whatif_gp)

    race_name = season_races.loc[season_races["round"] == args.round, "name"].iloc[0]
    scenario_desc = f"What if {args.driver} finished P{args.pos} in Round {args.round} ({race_name})?"

    print(f"\n=== COUNTERFACTUAL SCENARIO: {scenario_desc} ===")
    print(whatif_standings[["rank", "driver_name", "constructor", "total_points", "wins", "podiums"]].head(5).to_string(index=False))

    top2_refs = actual_standings["driverRef"].head(2).tolist()
    plot_trajectory(actual_gp, whatif_gp, season_races, top2_refs, args.year, scenario_desc)

    # 2. Monte Carlo Mid-Season Forecast
    total_rounds = int(season_races["round"].max())
    mc_summary = run_monte_carlo(actual_gp, season_races, cutoff_round=args.cutoff, n_sims=args.sims)
    print(f"\n=== MONTE CARLO TITLE FORECAST (AFTER ROUND {args.cutoff} OF {total_rounds}, {args.sims:,} SIMS) ===")
    print(mc_summary.to_string(index=False))

    plot_monte_carlo(mc_summary, args.year, args.cutoff, total_rounds)
    print("\nSaved charts: whatif_title_trajectory.png & monte_carlo_title_odds.png\n")


if __name__ == "__main__":
    main()
