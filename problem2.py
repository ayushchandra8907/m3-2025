import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#determined from data from mathworks
use_per_person = [11097.340449586547, 11073.718272065424, 10941.574701014348, 10925.040055373136, 10867.72300030933, 10600.545657186545, 11065.241307350008, 10663.471039849586, 10123.375254923389, 10609.725421769803, 10647.66812003078]

years = range(2012, 2023)

#input data into np arrays for sk
x = np.array(years).reshape((-1, 1))
y = np.array(use_per_person)

model = LinearRegression()
model.fit(x, y)

#lin regression A and B terms
intercept = model.intercept_
slope = model.coef_

print(f"intercept: {intercept}")
print(f"slope: {slope}")

#regression values
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

#Plot data
plt.scatter(x, y,  color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption  Per Memphis Resident')
plt.ylim(0, 13000)

plt.legend()
plt.show()
