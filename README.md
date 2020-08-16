# Airbnbmaybenot
<img src="images/Airbnb_Lockup_Over_Gradient.png" width=350px align=right>
Authors | Kara Hodges, Grace Huefner, Thomas Knies, Laura Bullard, Lisa Reed-Preston

## Table of Contents
* [Objective](#Objective)
* [Technologies](#Technologies)
* [Process](#Process)
* [Visualization](#Visualization)

# Objective | 
Analyze Airbnb data in selected international cities in 2018-2019 and predict the potential sales for Airbnb and then compare to the actual 2019-2020 data.  Specifically we will be looking at at:
* New York - Lisa  (March 1, 2020 - first case)
* Hong Kong - Grace
* Venice - Kara
* Rio - Thomas
* Berlin - Laura

# Data | 
http://insideairbnb.com/get-the-data.html

# Technologies |
* Python/Pandas
* Tableau
* Machine Learning

# Process |
Download monthly “Listings.csv” from the relevant city. (4th in section) Include every available month from May 2018 - May 2020 starting from the equivalent month when Covid-19 was announced.  For instance, NYC was affected in March, therefore the dataset will include May 2018 - May 2019, June 2019 - May 2020. “Fiscal Year” - With drill down into months if possible

## Process historical data
Clean data with Pandas/Python
* Add column for month of data
* Create annual spreadsheet based on May 2018 - May 2019
* Removing extra columns
* LEAVE IN Id as Listing ID, Host ID, Longitude, Latitude, Neighbourhood (not neighbourhood group), Room Type, Price, Availability, Calculated Host Listing, Minimum Nights

Machine Learning Processing
* Feed May 2018 - May 2019 into Machine Learning to predict June 2019-May 2020
  Extracted data from csv
  ```
  raw_df = pd.read_csv("data/NYC_May18May19.csv")
  print(f"The dataset contains {len(raw_df)} Airbnb listings")
  pd.set_option('display.max_columns', len(raw_df.columns)) # To view all columns
  pd.set_option('display.max_rows', 100)
  raw_df.head(3)
  ```
  Replaced columns with f/t with 0/1
  ```
  raw_df.replace({'f': 0, 't': 1}, inplace=True)

  # Plotting the distribution of numerical and boolean categories
  raw_df.hist(figsize=(20,20));
  ```
  Visualized data for better understanding
  <img src="images/NYC_nightly prices.png" width=350px align=right>
  ```
  # Distribution of prices from $0 to $1000
  plt.figure(figsize=(20,4))
  raw_df.price.hist(bins=100, range=(0,1000))
  plt.margins(x=0)
  plt.axvline(200, color='orange', linestyle='--')
  plt.title("Airbnb advertised nightly prices in NYC up to $10,000", fontsize=16)
  plt.xlabel("Price ($)")
  plt.ylabel("Number of listings")
  plt.savefig('images/NYC_nightly prices.png')
  plt.show()
  ```
  Created heat map of correlations between features in the df. A figure size can optionally be set.
  ```
  def multi_collinearity_heatmap(df, figsize=(11,9)):
    # Set the style of the visualization
    sns.set(style="white")

    # Create a covariance matrix
    corr = df.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmax=corr[corr != 1.0].max().max());
  ```
  Transformed necessary columns
  ```
  transformed_df['date'] = pd.to_datetime(transformed_df['date'])
  transformed_df['day_of_week'] = transformed_df['date'].dt.day_name().map({"Monday":0, "Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6})
  transformed_df["new_year_day"] = transformed_df['date'].apply(lambda x: pd.Timestamp(year=x.year, month=1, day=1))
  transformed_df['day_of_the_year'] = transformed_df.apply(lambda x: (x['date'] - x['new_year_day']).days + 1, axis=1)
  transformed_df['Year'] = transformed_df.new_year_day.astype(str).str[:4]
  transformed_df['Year'] = transformed_df['Year'].astype(int)
  del transformed_df['date']
  del transformed_df['new_year_day']
  transformed_df.head()
  ```
  Used sklearn to process data for machine learning
  ```
  # Separating X and y
  X = transformed_df
  y = transformed_df.price

  # Scaling
  scaler = StandardScaler()
  X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))
  
  # Splitting into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  ```
  Trained data using six models, including RGBoost and Keras Sequential.  RGBoost was the most successful model.
  ```
  xgb_reg_start = time.time()

  xgb_reg = xgb.XGBRegressor()
  xgb_reg.fit(X_train, y_train)
  training_preds_xgb_reg = xgb_reg.predict(X_train)
  val_preds_xgb_reg = xgb_reg.predict(X_test)

  xgb_reg_end = time.time()

  print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes")
  print("\nTraining MSE:", round(mean_squared_error(y_train, training_preds_xgb_reg),4))
  print("Validation MSE:", round(mean_squared_error(y_test, val_preds_xgb_reg),4))
  print("\nTraining r2:", round(r2_score(y_train, training_preds_xgb_reg),4))
  print("Validation r2:", round(r2_score(y_test, val_preds_xgb_reg),4))
  ```
  Prepared Test data (June 2019-May 2020)
  ```
  transformed_test["new_year_day"] = transformed_test['date'].apply(lambda x: pd.Timestamp(year=x.year, month=1, day=1))
  transformed_test['day_of_the_year'] = transformed_test.apply(lambda x: (x['date'] - x['new_year_day']).days + 1, axis=1)
  transformed_test['date'] = pd.to_datetime(transformed_test['date'])
  transformed_test['day_of_week'] = transformed_test['date'].dt.day_name().map({"Monday":0, "Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}) 
  transformed_test['Year'] = transformed_test.new_year_day.astype(str).str[:4]
  transformed_test['Year'] = transformed_test['Year'].astype(int)
  del transformed_test['Unnamed: 0']
  transformed_test=transformed_test.drop(['new_year_day'], axis=1)
  del transformed_test['date']
  transformed_test = transformed_test[transformed_df.columns]
  ```
  Tested data with RGBoost
  ```
  transformed_test['Prediction'] = np.expm1(xgb_reg.predict(transformed_test[ft_weights_xgb_reg]))
  filename = 'data/NYC_prediction.csv'
  pd.DataFrame({'id': transformed_test.id, 'price': transformed_test.Prediction}).to_csv(filename, index=False)
  ```
* Compare results of Machine Learning June 2019-May 2020 to actual results

Tableau Visualizations
* Each city has a dashboard of 4 visualizations
* 5 dashboards combined to create Tableau story.


# Visualizations |

* Each city has a dashboard of 4 visualizations
* 5 dashboards combined to create Tableau story.
* Dashboard of comparisons between then, now and prediction of now

# Troubleshooting |
* Converting data types in Pandas was necessary before running Machine Learning Process.
* Machine learning process included many tweaks to improve accuracy percentage, correct column alignment, same row input information.

