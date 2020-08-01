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
* Tokyo - Grace
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
Clean data with Pandas/Python (7/30)
* Add column for month of data
* Create annual spreadsheet based on May - May
* Removing extra columns
* LEAVE IN Id as Listing ID, Host ID, Neighbourhood (not neighbourhood group), Room Type, Price, Availability, Calculated Host Listing, Minimum Nights (keep document formatting of names)

Machine Learning Processing (8/1)
* Feed May 2018 - May 2019 into Machine Learning to predict June 2019-May 2020
* Compare results of Machine Learning June 2019-May 2020 to actual results

Tableau Visualizations (8/8)
* Each city has a dashboard of 4 visualizations
* 5 dashboards combined to create Tableau story.

Big Data (8/11, 8/13)
Whatever this is!!!

Presentation Preparation and Loose Ends (8/15)

Presentation (8/18)


# Visualizations |

* Each city has a dashboard of 4 visualizations
* 5 dashboards combined to create Tableau story.
* Dashboard of comparisons between then, now and prediction of now

# Troubleshooting |
* Converting data types in Pandas was necessary before running Machine Learning Process.

