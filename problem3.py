import pandas as pd

# Income, age, vehicle access spreadsheet
df1 = pd.read_csv("Untitled spreadsheet - Sheet1 (1).csv")  

# Work-related data spreadsheet
df2 = pd.read_csv("Untitled spreadsheet - Sheet1.csv") 

# Convert relevant columns in df2 (work data) to numeric
numeric_cols2 = [
    "Population age 16+ years old who work",
    "Primary mode of transportation to work (persons aged 16 years+): driving",
    "Primary mode of transportation to work (persons aged 16 years+): walking or public transit",
    "Primary mode of transportation to work (persons aged 16 years+):  other and work from home"
]

for col in numeric_cols2:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')

df = pd.merge(df1, df2, on=["Neighborhood", "ZIP code"], how="inner")

def normalize(data, min_val, max_val):
    """Normalize data using min-max scaling."""
    return (data - min_val) / (max_val - min_val) if max_val > min_val else 0

# Define min-max values for normalization
min_max_values['work'] = (
    df["Primary mode of transportation to work (persons aged 16 years+): walking or public transit"].min(),
    df["Primary mode of transportation to work (persons aged 16 years+): walking or public transit"].max()
)

# Apply normalization to Work Score (higher reliance on public transit = higher score)
df["Work_Score"] = df["Primary mode of transportation to work (persons aged 16 years+): walking or public transit"].apply(
    lambda x: normalize(x, *min_max_values['work'])
)
df["Income_Score"] = df["Median household income (in US dollars)"].apply(lambda x: 1 - normalize(x, *min_max_values['income'])) # Inverted (lower income = higher vulnerability)
df["Vehicle_Score"] = df["Households with no vehicles"].apply(lambda x: normalize(x, *min_max_values['vehicle']))
df["Age_Score"] = df["Households with one or more people 65 years and over"].apply(lambda x: normalize(x, *min_max_values['age']))

# Housing data spreadsheet
housing_df = pd.read_csv("queasion3 - Sheet1.csv")

# Calculate total number of homes by summing over construction periods
housing_df['TotalHomes'] = (housing_df["Homes built 2010 or later"] +
                            housing_df["Homes built 1990 to 2009"] +
                            housing_df["Homes built 1970 to 1989"] +
                            housing_df["Homes built 1950 to 1969"] +
                            housing_df["Homes built 1950 or earlier"])

# Compute weighted average construction year.
# Assume approximate mid-year values for each period:
# 2010 or later: 2015, 1990-2009: 2000, 1970-1989: 1980, 1950-1969: 1955, 1950 or earlier: 1940
housing_df['WeightedMidYear'] = (
    housing_df["Homes built 2010 or later"] * 2015 +
    housing_df["Homes built 1990 to 2009"] * 2000 +
    housing_df["Homes built 1970 to 1989"] * 1980 +
    housing_df["Homes built 1950 to 1969"] * 1955 +
    housing_df["Homes built 1950 or earlier"] * 1940
) / housing_df['TotalHomes']

# Compute building age (assuming current year is 2025)
housing_df['BuildingAge'] = 2025 - housing_df['WeightedMidYear']

# Compute average number of stories.
# Assume weights: Detached whole house = 1, Townhouse = 3, Apartments = 5, Mobile Homes/Other = 1.
housing_df['TotalHousingTypes'] = (housing_df["Detached whole house"] +
                                   housing_df["Townhouse"] +
                                   housing_df["Apartments"] +
                                   housing_df["Mobile Homes/Other"])
housing_df['AvgStories'] = (
    housing_df["Detached whole house"] * 1 +
    housing_df["Townhouse"] * 3 +
    housing_df["Apartments"] * 5 +
    housing_df["Mobile Homes/Other"] * 1
) / housing_df['TotalHousingTypes']

# Estimate greencover as the fraction of homes built 2010 or later (as a proxy for modern, green building practices)
housing_df['Greencover'] = housing_df["Homes built 2010 or later"] / housing_df['TotalHomes']

# For S (insulation quality parameter in Tphys), we use the BuildingAge.
# (Assumption: older buildings have lower S, causing higher temperatures.)
housing_df['S'] = housing_df['BuildingAge']

# Compute T_phys for each neighborhood using t and S.
housing_df['Tphys'] = housing_df.apply(lambda row: Tphys(t, row['S']), axis=1)

# Now compute the Temp Score using the provided formula:
# TempScore = Tphys + 1.4*shadelevel + T_change(greencover) + BuildingAge + 0.5*AvgStories + 0.1*(people/rooms)
housing_df['TempScore'] = (
    housing_df['Tphys'] +
    1.4 * shadelevel +
    T_change(housing_df['Greencover']) +
    housing_df['BuildingAge'] +
    0.5 * housing_df['AvgStories'] +
    0.1 * people_per_room
)

data = {
    "Neighborhood": [
        "Downtown / South Main Arts District / South Bluffs",
        "Lakeland / Arlington / Brunswick",
        "Collierville / Piperton",
        "Cordova, Zipcode 1",
        "Cordova, Zipcode 2",
        "Hickory Withe",
        "Oakland",
        "Rossville",
        "East Midtown / Central Gardens / Cooper Young",
        "Uptown / Pinch District",
        "South Memphis",
        "North Memphis / Snowden / New Chicago",
        "Hollywood / Hyde Park / Nutbush",
        "Coro Lake / White Haven",
        "East Memphis – Colonial Yorkshire",
        "Midtown / Evergreen / Overton Square",
        "East Memphis",
        "Windyke / Southwind",
        "South Forum / Washington Heights",
        "Frayser",
        "Egypt / Raleigh",
        "Bartlett, Zipcode 1",
        "Bartlett, Zipcode 2",
        "Bartlett, Zipcode 3",
        "Germantown, Zipcode 1",
        "Germantown, Zipcode 2",
        "South Riverdale"
    ],
    # Actual population values for each neighborhood.
    "Population": [11816, 43688, 56225, 44274, 37996, 7699, 12360, 3706, 22121, 4961, 21702, 14001, 18425, 43639, 42061, 15107, 26258, 42732, 5456, 39404, 43701, 20900, 38849, 30284, 25171, 16298, 23768],
    # Actual hospital counts for each neighborhood.
    "Hospital_Count": [4,1,0,1,0,0,0,0,3,1,0,1,0,0,0,0,0,2,0,0,2,2,0,0,2,0,0]
}

df3 = pd.DataFrame(data)

# Calculate the hospital ratio: number of hospitals per capita.
df3["Hospital_Ratio"] = df3["Hospital_Count"] / df3["Population"]

# Since higher hospital availability lowers vulnerability, we define a raw health score as the inverse.
#   Raw_Health_Score = 1 - Hospital_Ratio
# (Neighborhoods with a lower hospital-to-population ratio will have a higher raw health score.)
df3["Raw_Health_Score"] = 1 - df3["Hospital_Ratio"]

# Define a normalization function to scale values between 0 and 1.
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize the Raw Health Score to get the final Health Score.
df3["Health_Score"] = normalize(df3["Raw_Health_Score"])


# Add everything up to get the Vulnerability Score 
df["Vulnerability_Score"] = (
    0.2 * df["Income_Score"] +
    0.25 * df["Age_Score"] +
    0.2 * df["Temp_Score"] +
    0.15 * df3["Health_Score"] +
    0.1 * df["Vehicle_Score"] +
    0.1 * df["Work_Score"]
)


# Sensitivity Analysis Using Monte Carlo Simulation 

# Merge two dataframs into df
df = df.merge(df3[["Neighborhood", "Health_Score"]], on="Neighborhood", how="left")

# Define baseline weights (same order as the columns)
baseline_weights = np.array([0.2, 0.25, 0.2, 0.15, 0.1, 0.1])
weight_names = ["Income_Score", "Age_Score", "Temp_Score", "Health_Score", "Vehicle_Score", "Work_Score"]

# Number of Monte Carlo iterations
n_iterations = 1000
vuln_scores_all = np.zeros((n_iterations, len(df)))

# Variability range (±10%)
delta = 0.1

# Monte Carlo simulation
for i in range(n_iterations):
    # Generate random weights with ±10% variation
    random_weights = np.array([
        np.random.uniform(b * (1 - delta), b * (1 + delta))
        for b in baseline_weights
    ])
    # Normalize to sum to 1
    random_weights /= random_weights.sum()
    
    # Compute vulnerability score for each neighborhood
    vuln_score_iter = np.zeros(len(df))
    for j, col in enumerate(weight_names):
        vuln_score_iter += random_weights[j] * df[col]
    
    # Store results
    vuln_scores_all[i, :] = vuln_score_iter

# Compute mean and standard deviation of vulnerability scores
df["Vuln_MC_Mean"] = vuln_scores_all.mean(axis=0)
df["Vuln_MC_Std"] = vuln_scores_all.std(axis=0)

# Display results
print(df[["Neighborhood", "Vuln_MC_Mean", "Vuln_MC_Std"]])
