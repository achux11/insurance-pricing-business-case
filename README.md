# P&C Insurance Pricing: Segment Performance & Rate Change Business Case

A model of auto insurance claim frequency (Poisson) and severity (Gamma), combined into an expected loss cost. It measures loss ratios by region and driver segment, then evaluates the business case for a risk-based rate change: the impact on premium, loss ratio and retention.

**Data:** freMTPL2 (French Motor Third-Party Liability), OpenML ids 41214 and 41215. The script downloads it automatically.

## Results
- **678K** policies, **36K** claims, **22** regions.
- **Loss ratio by region:** ranges from **24.6%** to **87.7%**.
- **Loss ratio by driver age:** **116.8%** for drivers aged 18-25, vs. 40-47% for other age groups.
- **Rate change (risk-based, capped at +/-15%):** premium **-7.7%**, loss ratio **46.1% to 48.1%**, at **82.6%** retention.
- **Recommendation:** targeted increases on unprofitable segments rather than a broad repricing.

**Note:** current pricing is simulated as a flat book premium adjusted by area, and retention is a simple assumption (each +1% of rate lowers renewal by 0.5 points from a base of 85%). These are stated assumptions, not observed data.

## Run
```bash
pip install -r requirements.txt
python insurance_pricing_project.py
```
The script writes `segment_performance.csv`.
