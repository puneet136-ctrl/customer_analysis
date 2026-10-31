# Customer Behaviour Lab — ASSP Mini Project

Interactive dashboard that turns e-commerce transactions into measurable customer behaviour:
distribution fitting, a 3-state Markov chain (Active / Regular / Inactive), hypothesis tests
(t, F, ANOVA + post-hoc), and a retention what-if simulator.

## 1. Setup (about 5 minutes)

```bash
# 1. Install Python 3.10+ from python.org, then in this folder:
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install packages
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```
The browser opens at http://localhost:8501. With "Built-in sample data" selected it works immediately.

## 2. Use the real dataset

1. Download **"Online Retail"** (UCI Machine Learning Repository) or **"Online Retail II"** (Kaggle).
2. In the sidebar choose **Upload my dataset** and upload the .xlsx or .csv.
3. Expected columns: InvoiceNo, CustomerID, InvoiceDate, Quantity, UnitPrice (Description and Country optional).
   Other common names (Order ID, Customer ID, Date, Price...) are auto-detected.
4. The app removes rows with no customer, returns/cancellations (Quantity <= 0) and zero prices.

Large files (500k rows) take ~20-40 s for the first load, then results are cached.

## 3. File map

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (7 tabs) |
| `stats_utils.py` | All statistics: fits, Markov chain, tests, simulation, Kaplan-Meier |
| `generate_sample_data.py` | Synthetic data in the UCI schema (`python generate_sample_data.py`) |
| `requirements.txt` | Dependencies |

## 4. Mapping to your presentation

| Slide | Dashboard tab | Function in stats_utils.py |
|---|---|---|
| 1 Measurement | Overview | `customer_table`, `lorenz_gini` |
| 2 Variable system | Overview / Lookup | `customer_table` |
| 3 Distributions | Distributions | `fit_lognormal`, `poisson_vs_negbin`, `fit_exponential_weibull` |
| 4 State space | States & Markov chain | `monthly_states`, `transition_matrix`, `steady_state`, `mean_first_passage` |
| 5 Inference | Hypothesis tests | `t_test`, `f_test`, `anova_suite`, `bootstrap_mean_ci` |
| Extra | What-if, Extras | `adjust_matrix`, `project`, `kaplan_meier` |

Note: your slide 4 table says Bull/Stable/Bear and the speaker notes mention price series and returns.
Those look left over from a finance template; change them to Active/Regular/Inactive.

## 5. Suggested build timeline

1. Day 1: install, run with sample data, read each tab.
2. Day 2: load the real dataset, tune the Active/Inactive thresholds, note the results.
3. Day 3: screenshots for every tab, fill in the transition matrix on slide 4.
4. Day 4: write the report (method, results, limitations); rehearse the demo.

## 6. Limitations to state in your report (examiners like honesty)

- Fitted distributions are approximations; KS p-values are optimistic when parameters are estimated.
- Large samples make almost any test "significant", so effect sizes (Cohen's d, eta-squared) are reported.
- The Markov property may be violated; the app tests it.
- State thresholds (30/90 days) are a modelling choice; the sidebar lets you check sensitivity.
- Revenue projections assume transition probabilities stay constant.

## 7. Likely viva questions

- Why lognormal for spend, and how do you know it fits? (QQ-plot, AIC comparison, KS)
- What does Var/Mean > 1 mean for a Poisson model? (overdispersion -> negative binomial)
- Why ANOVA instead of several t-tests? (family-wise error rate)
- What do the diagonal entries of the transition matrix represent? (retention)
- What is the steady-state vector and how is it computed? (pi P = pi, sum = 1)
- What happens if ANOVA assumptions fail? (log transform, Kruskal-Wallis)

## 8. Optional: put it online

Push the folder to GitHub, then deploy free on https://share.streamlit.io and share the link.
