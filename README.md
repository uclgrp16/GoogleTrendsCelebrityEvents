
# GoogleTrendsCelebrityEvents

The project is largely based on the following repositories/codes:
1. [Stata Harmonization Code](https://github.com/jack-madison/Google-Trends). 
2. [Pytrends.daily_data.py](https://github.com/GeneralMills/pytrends/blob/master/pytrends/dailydata.py).
3. [Selenium Google Trends Scraping](https://ggiesa.wordpress.com/2018/05/15/scraping-google-trends-with-selenium-and-python/).


### Data Collection:
The code has **only** been tested on a Apple M2 Macbook running MacOS Sonoma. 

#### General Requirements
1. Install homebrew following these [instructions](https://brew.sh/).


#### Requirements if using Proxies:
1. Download NordVPN
   Not ideal but this is what has been most reliable

2. Create a NordVPN account, go to [dashboard](https://my.nordaccount.com/dashboard/nordvpn/) and follow the instructions for Manual Setup.
   Edit the utils.configs file with the Username and Password under Service Credentials.
   
3. Install openVPN
   ```brew install openvpn```

4. Install Njord (Python Client for NordVPN)
	**pip**:
	
		pip install njord
	
	 **conda**:
	
		conda install -c njord
	
	
#### Requirements  
1. Install chromedriver with brew:
	```brew install --cask chromedriver```

2. Make sure you have downloaded and installed the **Stata App**  to harmonize the data according to the [methods](https://github.com/jack-madison/Google-Trends) we are using. 

3. Install Python packages:

	**pip**:

		pip install pytrends pandas statsmodels matplotlib selenium
   
	 **conda**:
   
		conda install -c conda-forge pytrends pandas statsmodels matplotlib selenium


####  Usage
1. Clone this repository:
```
git clone https://github.com/uclgrp16/GoogleTrendsCelebrityEvents
cd GoogleTrendsCelebrityEvents
```

2.  Run project_data_collection.py:
```
python project_data_collection.py
```

The data will take a long time to download, because it is trying to avoid 429 (TooManyRequests)  error, which is why pytrends is not an option for the data collection.
We also haven't been able to find a headless option to download the data...

#### Output 

The raw and harmonized csv files will be saved in the 'data' directory.  
