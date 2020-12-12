# Bike-Sharing-Demand
https://www.kaggle.com/c/bike-sharing-demand
</br>Checking null variables in data</br>
<code>train_data.isnull().values.any()</code>
</br>Out: <code>False</code>
</br>Afterward I analized data using Tableau online</br>
![Months](https://github.com/RapidSwipe/Bike-Sharing-Demand/blob/main/months.png?raw=true)</br></br>
![Season](https://github.com/RapidSwipe/Bike-Sharing-Demand/blob/main/Season.png?raw=true)

</br></br>
# Analysing columns correlation</br>
![correlation](https://github.com/RapidSwipe/Bike-Sharing-Demand/blob/main/chart.png?raw=true)
</br></br>
Dropping column 'workingday' because it's highly correlated to 'day' column</br></br>
After many of tests turned out that the most efficient regressor is RandomForestRegressor</br>
<code>
from sklearn.ensemble import RandomForestRegressor</code></br>
<code>
regressor=RandomForestRegressor(n_estimators=350)</code>
</br></br>
After training and GridSearch optimization test set RMSLE equals to 
</br><code>0.3239540116411649</code>
</br>
Finally sumbmitting results:</br>
<code>sample_submission.csv</code>
