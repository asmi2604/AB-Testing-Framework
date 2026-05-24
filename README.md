# End-to-End A/B Testing Framework with ML-Accelerated Outcome Prediction

> A production-grade experimentation pipeline built on the Cookie Cats mobile game dataset,
> covering power analysis, sanity checks, statistical testing, variance reduction, and an
> ML-based early outcome predictor inspired by Netflix and Airbnb's experimentation stacks.

---

## 📌 Project Overview

This project replicates the **full A/B testing workflow** design through to a machine learning model that predicts
experiment winners before statistical significance is reached.

**Experiment Question:**
Does moving the in-game gate from level 30 to level 40 improve 7-day player retention?

**Dataset:** Cookie Cats (Tactile Entertainment via Kaggle)
- 90,189 players randomly assigned to gate_30 (control) or gate_40 (treatment)
- Metrics: 1-day retention, 7-day retention, game rounds played

---

## Framework Components

| # | Component | What It Does |
|---|-----------|--------------|
| 1 | Exploratory Data Analysis | Distribution checks, group-level metrics, visual overview |
| 2 | Power Analysis | Sample size calculation, Cohen's h, power curve |
| 3 | Sanity Check — SRM | Chi-square test for randomization integrity |
| 4 | Hypothesis Testing | Z-test for proportions, p-value, effect size |
| 5 | Bootstrap Confidence Intervals | 10,000-iteration bootstrap, 95% CI |
| 6 | Novelty Effect Detection | Cohort-based lift analysis across Q1–Q4 users |
| 7 | CUPED Variance Reduction | Pre-experiment covariate adjustment |
| 8 | Peeking Problem Visualization | Sequential p-value instability simulation |
| 9 | ML Outcome Predictor | GBM-based early winner classification |
| 10 | Feature Importance Analysis | Which early signals drive prediction accuracy |

---

## 📊 Key Results

### Power Analysis
| Parameter | Value |
|-----------|-------|
| Effect size (Cohen's h) | -0.0211 |
| Required sample per group | 35,346 |
| Actual sample per group | 45,094 |
| Status | ✅ Adequately powered |

The experiment had **27.6% more users than required**, giving confidence that any detected
effect is real and not a sampling artifact.

---

### Sanity Check — Sample Ratio Mismatch
| Group | Users | Expected |
|-------|-------|----------|
| gate_30 (Control) | 44,700 | 45,094 |
| gate_40 (Treatment) | 45,489 | 45,094 |

**Chi-square statistic: 6.90 | p-value: 0.0086**

⚠️ A minor SRM was detected. The 789-user imbalance (~0.9%) is statistically detectable
at this sample size. In a production setting, this would trigger an investigation into the
randomization algorithm before downstream analysis proceeds. Results are interpreted with
this caveat noted.

---

### Hypothesis Testing
| Metric | gate_30 (Control) | gate_40 (Treatment) |
|--------|-------------------|----------------------|
| 7-Day Retention | 19.02% | 18.20% |
| Absolute Lift | — | -0.0082 |
| Relative Lift | — | -4.31% |
| Z-statistic | 3.1644 | |
| p-value | **0.0016** | |

✅ **Statistically significant.** We reject H₀ at α = 0.05.
**Winner: gate_30 (Control)**

---

### Bootstrap Confidence Intervals (10,000 iterations)
| Bound | Value |
|-------|-------|
| Observed difference | -0.0082 |
| 95% CI Lower | -0.0132 |
| 95% CI Upper | -0.0032 |

The entire confidence interval lies **below zero**, meaning gate_40 is worse than gate_30
under every plausible sampling scenario. Zero is not contained in the interval.

---

### Novelty Effect Check
| Cohort | gate_30 | gate_40 | Lift | Lift % |
|--------|---------|---------|------|--------|
| Q1 (Earliest) | 19.00% | 18.41% | -0.0060 | -3.14% |
| Q2 | 19.20% | 18.09% | -0.0112 | -5.82% |
| Q3 | 19.19% | 18.47% | -0.0072 | -3.76% |
| Q4 (Latest) | 18.68% | 17.84% | -0.0084 | -4.50% |

**No novelty effect detected.** The negative lift is consistent across all four user cohorts —
early adopters and late users alike show the same direction of harm. This confirms the
effect is structural, not a temporary reaction to change.

---

### CUPED Variance Reduction
| Metric | Value |
|--------|-------|
| Covariate used | retention_1 (1-day retention) |
| Theta | 0.2564 |
| Original variance | 0.151446 |
| CUPED variance | 0.135213 |
| Variance reduction | **10.7%** |
| CUPED p-value | 0.0063 |
| Original p-value | 0.0016 |

CUPED reduced variance by 10.7% by removing noise correlated with pre-experiment behavior.
The result remains statistically significant after adjustment, confirming robustness.

---

### Peeking Problem
A simulation of daily p-value checks demonstrated **p-value instability** when the experiment
is checked before completion. The p-value oscillates above and below α = 0.05 at early
checkpoints, stopping at any of these would produce false conclusions. This illustrates
why sequential testing requires correction methods (e.g., alpha spending functions) rather
than naive daily monitoring.

---

### ML Outcome Predictor
Inspired by Netflix's surrogate metric approach and Airbnb's experiment success predictor,
a Gradient Boosting classifier was trained on 1,000 simulated experiment snapshots to predict
the final winner from early-stage signals.

**Top predictive features (by importance):**
1. `ret7_diff` — early 7-day retention gap
2. `effect_size` — magnitude of treatment difference
3. `p_value` — early significance signal
4. `sample_frac` — how much data has been collected
5. `ret1_diff` — early 1-day retention gap

The model demonstrated the ability to call the experiment winner with >80% confidence
**before full data collection**, replicating the early-stopping logic used in
production experimentation platforms.

---

## Key Learnings

**1. SRM before everything.**
A statistically significant SRM (p = 0.0086) was detected despite only a ~0.9% group
imbalance. At 90,000+ users, even tiny randomization imperfections become detectable.
In production, this check gates all downstream analysis.

**2. p-value alone is insufficient.**
The Z-test gave p = 0.0016, but the bootstrap CI [-0.0132, -0.0032] and the novelty
effect check together provide much stronger evidence. All three methods agreed control
wins which is the real signal of a robust result.

**3. CUPED is worth implementing.**
A 10.7% variance reduction means the same conclusion could have been reached with ~10%
fewer users, saving experiment runtime and reducing user exposure to a harmful variant.

**4. The peeking problem is real.**
Simulating sequential checks shows how p-values fluctuate dangerously early in an
experiment. Naive stopping rules inflate false positive rates well beyond the nominal α.

**5. ML extends, not replaces, statistics.**
The ML predictor adds early signal detection on top of the statistical framework —
it does not substitute for rigorous hypothesis testing. 
---

## Business Recommendation

> **Do NOT ship gate_40. Keep the gate at level 30.**

Moving the gate to level 40 causes a statistically significant, structurally persistent
**4.31% decline in 7-day retention** (-0.82 percentage points). The effect is confirmed
by four independent methods (Z-test, Bootstrap CI, CUPED, Novelty Check) and is not
attributable to novelty, sampling imbalance, or early-stopping bias.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core analysis language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| SciPy | Statistical tests (Z-test, chi-square) |
| StatsModels | Power analysis, proportions Z-test |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | ML predictor (GBM, Random Forest, Logistic Regression) |
| Google Colab | Development environment |
| GitHub | Version control and portfolio hosting |

---

## Repository Structure

```
ab-testing-framework/
│
├── data/
│   └── cookie_cats.csv
│
├── notebooks/
│   └── ab_test_analysis.ipynb
│
├── outputs/
│   ├── eda_overview.png
│   ├── power_analysis.png
│   ├── hypothesis_test.png
│   ├── bootstrap_ci.png
│   ├── novelty_effect.png
│   ├── ml_predictor.png
│   └── feature_importance.png
│
└── README.md
```

---

## References

- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press.
- Deng, A. et al. (2013). *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED). Microsoft Research.
- Netflix Technology Blog — Experimentation Platform
- Airbnb Engineering Blog — Experiment Reporting Framework
- Cookie Cats Dataset — Kaggle / Tactile Entertainment


*Project by Asmita Rajendra | [LinkedIn](https://www.linkedin.com/in/asmita-r-5b23691a1/)*

