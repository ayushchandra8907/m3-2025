# M3 Challenge 2025 – Staying Cool as the World Heats Up

**Team #17621**  
**Semifinalists (7th–12th place out of 794 teams)**  
March 3, 2025

---

## Project Overview

Climate change is driving more intense heat waves that hit vulnerable communities hardest. Our team chose Memphis, Tennessee, as a case study to:

1. **Model indoor temperatures** in non‑air‑conditioned homes during a 24‑hour heat wave.  
2. **Predict peak energy demand** on the city’s power grid over the next 20 years, accounting for population changes, climate trends, and electric‑vehicle adoption.  
3. **Score neighborhood vulnerability** to heat waves and grid failures using weighted factors and Monte Carlo sensitivity analysis to ensure robustness under real‑world variability, then translate findings into actionable recommendations.

All results, discussion and full technical details are in the PDF report.

---

## Methods Overview

### Part I (First_Problem.py)
- **Data & Preprocessing:** Loaded outdoor temperature time series for a 24‑hour heat wave.  
- **Sinusoidal Fit:** Used a fixed‑frequency sinusoidal model (daily cycle) to approximate outdoor temperature.  
- **Heat‑Transfer Model:** Applied Newton’s Law of Cooling to couple outdoor and indoor temperatures, fitting the heat‑transfer coefficient for each house.  
- **Output & Visualization:** Extracted best‑fit parameters, plotted both observed and predicted indoor temperatures for four representative homes.

### Part II (Second_Problem.py)
- **Population & Climate Trends:** Fitted separate linear regression models to historical population data and cooling‑degree days.  
- **Per‑Person Consumption:** Modeled energy use per capita via linear regression on past energy records.  
- **Electric‑Vehicle Adoption:** Fitted a logistic curve to EV registration data to project penetration over time.  
- **Demand Forecast:** Combined forecasts (population × cooling × per‑person use × EV factor) to simulate weekly summer demand for the next 30 years. Plotted the resulting time series.

### Part III (Third_Problem.py)
- **Data Integration:** Merged neighborhood census and housing data into a single table.  
- **Factor Selection:** Identified six key drivers: median income, fraction aged 65+, predicted peak indoor temperature, access to healthcare (distance‑based), vehicle ownership rate, and workforce participation.  
- **Composite Scoring & Sensitivity Analysis:**  
  - Applied a tailored weighting scheme to combine factors into a baseline vulnerability metric.  
  - Conducted **1,000** Monte Carlo simulations per neighborhood—sampling each factor’s empirical distribution—to evaluate score stability and refine weights through correlational checks.  
- **Results & Analysis:**  
  - Generated a **bar graph** highlighting the top 10 most vulnerable and top 10 least vulnerable ZIP codes/neighborhoods.  
  - Used correlational analysis to validate that weights aligned with observed score variability.  
  - Developed actionable recommendations for resource allocation during heat waves based on the vulnerability rankings.

---

## Repository Contents

| File                         | Description                                                                                   | Technologies Used                                  |
|------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------|
| **official2025m3_17621.pdf** | Full team report (Executive Summary, Parts I–III, Conclusions, References, Code Appendix).     | —                                                  |
| **First_Problem.py**         | Part I: sinusoidal & heat‑transfer modeling of indoor temperatures.                           | Python, NumPy, SciPy, Matplotlib                   |
| **Second_Problem.py**        | Part II: linear & logistic regression forecasting of summer energy demand.                   | Python, NumPy, Pandas, scikit‑learn, Matplotlib     |
| **Third_Problem.py**         | Part III: Monte Carlo vulnerability scoring and correlational analysis across neighborhoods.   | Python, Pandas                                    |
| **README.md**                | This file.                                                                                    | —                                                  |
