# F1 Championship What-If & Monte Carlo Engine

A Python scenario analysis and Monte Carlo championship simulation engine built on historical Formula 1 race and sprint data. It evaluates counterfactual race outcomes ("What if a key DNF or position swap hadn't happened?") and forecasts mid-season championship win probabilities across remaining rounds.

## Features
- Exact Multi-Table Points Engine: Joins `results.csv`, `sprint_results.csv`, `races.csv`, `drivers.csv`, `constructors.csv`, and `status.csv` to reproduce official season standings (including Grand Prix points, sprint points, fastest-lap bonuses, and half-point races).
- Counterfactual "Rewrite History" Mode: Overrides a driver's finishing position in any chosen round, shifts affected drivers realistically, and recalculates cumulative points trajectories and final championship standings.
- Monte Carlo Championship Forecaster: Freezes standings at a specified cutoff round and runs 10,000 empirical simulations over remaining Grand Prix and Sprint weekends to estimate title win probabilities, expected final points, and 10th-90th percentile confidence intervals.
- Automated Visualizations: Exports `whatif_title_trajectory.png` and `monte_carlo_title_odds.png`.

## Example Usage
- Default (2021 title fight override + Round 15 Monte Carlo forecast):
  `python whatif_engine.py`
- Custom scenario (2024 season override + Round 16 Monte Carlo forecast):
  `python whatif_engine.py --year 2024 --round 11 --driver norris --pos 1 --cutoff 16 --sims 10000`

## Sample Output (2021 Season)
- Actual Final Standings: Max Verstappen (395.5 pts) vs. Lewis Hamilton (387.5 pts)
- Counterfactual (Hamilton P2 in Round 6 Azerbaijan GP): Lewis Hamilton (405.5 pts) vs. Max Verstappen (395.5 pts)
- Monte Carlo Title Odds (After Round 15/22, 10,000 sims): Lewis Hamilton 53.54% (Exp: 364.4 pts) | Max Verstappen 46.46% (Exp: 361.3 pts)
