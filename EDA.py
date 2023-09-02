import pandas as pd
from pandas_profiling import ProfileReport

dataset = pd.read_csv("House Price India.csv")
dataset.drop(['id', 'Date', 'Renovation Year', 'Lattitude',
              'Longitude', 'living_area_renov', 'lot_area_renov',
              'Number of schools nearby','waterfront present','number of views'],
             axis=1,
             inplace=True)

report  = ProfileReport(dataset,explorative=True)
report.to_file("EDA.html")
print(dataset.columns)