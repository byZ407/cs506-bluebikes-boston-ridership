# CS506 Final Project: Predicting Bluebikes Ridership in Boston Using Weather and Temportal Features

## Project Description
Bluebikes is an essential part of Boston's public transportation network. The city hosts more than 5,600 bikes and nearly 600 stations, providing students, residents, and visitors with a convenient and affordable way to travel between campuses or around the city. However, Bluebikes usage can vary depending on factors such as time of year, seasons, weather conditions, days of the week.  
The goal of this project is to understand these patterns and build models that can successfully predict Bluebikes ridership on a given day through analyzing histroical Bluebikes ridership data and weather data in Boston, using the full data science lifecycle that include data collection, data cleaning, feature extraction, data visualization, and model training.

## Project Timeline
- Setup and data collection (Week 1)
- Data cleaning and integration (Week 2)
- Feature extraction (Week 3)
- Preliminary data analysis and visualization (Week 4)
- Linear Regression modeling and evaluation (Week 5)
- Random Forest modeling and evaluation (Week 6)
- XGBoost modeling and evaluation (Week 7)
- Model comparison and refinement (Week 8)
- Results analysis and final visualizations (Week 9)
- Final report and presentation (Week 10)

## Project Goals
The goal of this project is to successfully predict the total number of Bluebikes rides on a given day based on the weather conditions and temporal context.
I will examine features such as:
- month
- season
- day of the week
- temperature
- precipitation
- snowfall

and investigate how they are associated with daily Bluebikes ridership.  
A potential secondary goal is to rank the features based on feature importance and find out which features have the strongest correlation with daily Bluebikes ridership. 

## Data Collection
### Bluebikes Ridership Data
Source: https://bluebikes.com/system-data  
Bluebikes directly publishes downloadable csv files of Bluebikes trip data each month, the data include:
- Bike Type & ID
- Trip Duration (seconds)
- Start Time and Date
- Stop Time and Date
- Start Station Name & ID & Latitude/Longitude
- End Station Name & ID & Latitude/Longitude
- User Type

### Boston Weather Data
Source: https://open-meteo.com/en/docs/historical-weather-api  
I will collect historical weather data for Boston using the Open-Meteo Historical Weather API, the data include:
- Temperature
- Rain
- Snowfall
- Precipitation
- Wind speed & direction
- Other weather conditions (humidity, cloud coverage etc.)

## Modeling Plan
Possible models include:
- Linear Regression
- Random Forest
- XGBoost

These are subject to change as we learn more of the data or explore new methods/knowledge in class.

## Visualization Plan
Possible visualizations include:
- Bluebikes ridership vs. temperature
- Bluebikes ridership vs. precipitation
- Average Bluebikes ridership by month/day of the week/season
- Ranked feature importance

These are subject to change as we learn more of the data or explore new methods/knowledge in class.
