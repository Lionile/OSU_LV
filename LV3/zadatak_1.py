import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# a)
print("Data count: " + str(data.shape[0]))
print("Data types: " + str(data.dtypes))
print("Data duplicate count: " + str(data.duplicated().sum()))
data.drop_duplicates(inplace=True)
print("Data missing values count: " + str(data.isnull().sum().sum()))
data.dropna(inplace=True)

data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')
print("Data types: " + str(data.dtypes))


# b)
print("\nLowest fuel consumption:\n" + str(data.sort_values(by='Fuel Consumption City (L/100km)').head(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']]))
print("\nHighest fuel consumption:\n" + str(data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']]))


# c)
temp_data = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print("\nAverage CO2 emission for engine size between 2.5 and 3.5: " + str(temp_data['CO2 Emissions (g/km)'].mean()))


# d)
audi_data = data[data['Make'] == 'Audi']
print("\nAudi data count: " + str(audi_data.shape[0]))
print("4 cylinder Audi average C02 emission: " + str(audi_data[audi_data['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean()))


# e)
plt.bar(data['Cylinders'].value_counts().index, data['Cylinders'].value_counts().values)
plt.title('Car count per cylinder count')
plt.xlabel('Cylinder count')
plt.ylabel('Number of cars')

plt.figure()
plt.bar(data['Cylinders'].value_counts().index, data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean())
plt.title('Average CO2 emission per cylinder count')
plt.xlabel('Cylinder count')
plt.ylabel('Emissions (g/km)')


# f)
print("\nAverage diesel fuel consumption: " + str(data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].mean()))
print('Median diesel fuel consumption: ' + str(data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].median()))
print("\nAverage gasoline fuel consumption: " + str(data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)'].mean()))
print('Median gasoline fuel consumption: ' + str(data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)'].median()))


# g)
print('\nHighest 4 cylinder car fuel consumption: ' + str(data[(data['Cylinders'] == 4)].sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(1)[['Make', 'Model', 'Fuel Consumption City (L/100km)']]))


# h)
print('\nManual transmission car count: ' + str(data[data['Transmission'].str.startswith('M')]['Transmission'].count()))


# i)
print(data.corr(numeric_only=True))


plt.show()