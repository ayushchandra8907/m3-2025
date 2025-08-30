import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Time data (Hours)
time_hours = np.arange(24)

temp_outdoor = np.array([
    85, 85, 84, 83, 83, 83, 84, 88, 91, 94, 96, 97,
    100, 100, 102, 102, 100, 99, 97, 94, 92, 91, 90, 89
])

# Define sinusoidal function with fixed frequency
def sinusoidal_model_fixed(x, A, C, D):
    B = np.pi / 12  # Fixed frequency corresponding to a 24-hour period
    return A * np.sin(B * x + C) + D

# Fit the model to data (only A, C, D are fitted)
def fit_sinusoidal_fixed(x, y):
    params, _ = curve_fit(sinusoidal_model_fixed, x, y, p0=[10, 0, 75])
    return params

A, C, D = fit_sinusoidal_fixed(time_hours, temp_outdoor)
B = np.pi / 12  # fixed frequency

plt.figure(figsize=(10, 5))
plt.plot(time_hours, sinusoidal_model_fixed(time_hours, A, C, D), label="Fitted Sinusoidal Model", color="red")
plt.scatter(time_hours, temp_outdoor, label="Outdoor Temperature", linestyle="dashed", color="black")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°F)")
plt.title("Predicted Outdoor Temperature Over 24 Hours")
plt.legend()
plt.grid()
plt.show()

np.save("sinusoidal_params.npy", [A, B, C, D])
print(f"Model parameters: A={A:.2f}, B={B:.4f}, C={C:.2f}, D={D:.2f}")


tau = 2.5
Aout = -9.07 # amplitude
b = 0.2618
c = 0.88
d = 92.04
T0 = 85 # this is the initial value that we were given in the data
k = 0.07

def Tout(t):
  return Aout*np.sin(b*t + c) + d

def Tphys(t, S):
  return Tout(t - tau) + (T0 - Tout(t - tau)) * np.exp(-k*t / S)

def Tchange(greencover):
  return 2*(0.7857 - 0.2*greencover + 0.6*(1-greencover))/(0.7857 - 0.5103)

def age(year):
  return 5 / (1 + np.exp(year - 1980))

def Tin(t, S, shadelevel, greencover, year, stories, people, rooms):
  if shadelevel.lower() == "not at all shady":
    shadelevel = 0
  elif shadelevel.lower() == "not very shady":
    shadelevel = 1
  elif shadelevel.lower() == "shady":
    shadelevel = 2
  elif shadelevel.lower() == "very shady":
    shadelevel = 3
  else:
    raise ValueError

  return Tphys(t, S) + 1.4 * shadelevel + Tchange(greencover) + age(year) + 0.5 * stories + 0.1 * people / rooms

plt.figure(figsize=(10, 5))
plt.plot(time_hours, Tin(time_hours, .88, "shady", .4829, 1953, 1, 3, 3), label="Home 1", color="blue")
plt.plot(time_hours, Tin(time_hours, .63, "not very shady", .1047, 1967, 1, 3, 2), label="Home 2", color="green")
plt.plot(time_hours, Tin(time_hours, .74, "not at all shady", .1047, 2003, 15, 2, 1), label="Home 3", color="black")
plt.plot(time_hours, Tin(time_hours, 2.78, "not at all shady", .0451, 1990, 2, 6, 5), label="Home 4", color="red")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°F)")
plt.title("Predicted Indoor Temperature Over 24 Hours")
plt.legend()
plt.grid()
plt.show()

# Sensitivity Analysis 

houses = {
    "Home 1": {"S": 0.88, "shadelevel": "shady",         "greencover": 0.4829, "year": 1953, "stories": 1,  "people": 3, "rooms": 3},
    "Home 2": {"S": 0.63, "shadelevel": "not very shady",  "greencover": 0.1047, "year": 1967, "stories": 1,  "people": 3, "rooms": 2},
    "Home 3": {"S": 0.74, "shadelevel": "not at all shady",  "greencover": 0.1047, "year": 2003, "stories": 15, "people": 2, "rooms": 1},
    "Home 4": {"S": 2.78, "shadelevel": "not at all shady",  "greencover": 0.0451, "year": 1990, "stories": 2,  "people": 6, "rooms": 5}
}

num_iterations = 5  # Number of jitter iterations

print("Average percent difference due to a ±5% random jitter for each house:")
for house_name, params in houses.items():
    # Compute the original prediction for the house
    original_prediction = Tin(time_hours, params["S"], params["shadelevel"],
        params["greencover"], 
        params["year"],
        params["stories"], 
        params["people"], 
        params["rooms"])

    percent_differences = []

    for _ in range(num_iterations):
        # Jitter each numeric parameter by a random factor between 0.95 and 1.05
        jittered_params = {
            "S": params["S"] * np.random.uniform(0.95, 1.05),
            "greencover": params["greencover"] * np.random.uniform(0.95, 1.05),
            "year": params["year"] * np.random.uniform(0.95, 1.05),
            "stories": params["stories"] * np.random.uniform(0.95, 1.05),
            "people": params["people"] * np.random.uniform(0.95, 1.05),
            "rooms": params["rooms"] * np.random.uniform(0.95, 1.05),
            "shadelevel": params["shadelevel"]  # remains unchanged
        }

        # Compute jittered prediction
        jittered_prediction = Tin(time_hours, jittered_params["S"], jittered_params["shadelevel"],
        jittered_params["greencover"], 
        jittered_params["year"],
        jittered_params["stories"], 
        jittered_params["people"], 
        jittered_params["rooms"])

        # Calculate the percent difference (averaged over time points)
        perc_diff = np.mean(np.abs(jittered_prediction - original_prediction) / np.abs(original_prediction) * 100)
        percent_differences.append(perc_diff)

    avg_percent_difference = np.mean(percent_differences)
    print(f"{house_name}: {avg_percent_difference:.2f}%")
