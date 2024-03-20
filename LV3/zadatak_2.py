import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

# a)
plt.hist(data['CO2 Emissions (g/km)'])
plt.title('CO2 emission histogram')
plt.ylabel('Number of cars')
plt.xlabel('Emissions (g/km)')

# b)
data.plot.scatter(x = 'Fuel Consumption City (L/100km)',
            y = 'CO2 Emissions (g/km)',
            c = 'Fuel Type')
plt.title('CO2 emissions vs Fuel consumption')

# c)
data.boxplot(column='Fuel Consumption City (L/100km)', by='CO2 Emissions (g/km)')


# d)
grouped_data = data.groupby('Fuel Type')
count_cars = grouped_data.size()
count_cars.plot.bar()
plt.title('Number of Cars per Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Number of Cars')


# e)
plt.figure()
plt.bar(data['Cylinders'], data['CO2 Emissions (g/km)'])
plt.title('CO2 Emissions per cylinder count')
plt.xlabel('Cylinder count')
plt.ylabel('CO2 Emissions (g/km)')

plt.show()