from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from urllib.parse import quote

import time
import os
import pandas as pd
from datetime import date, timedelta
from calendar import monthrange
import subprocess

base_url = 'https://trends.google.com/trends/explore?'
webdriver_path = '/opt/homebrew/bin/chromedriver'

def check_dir(dirname: str) -> str: 
    """Checks if directory exists, if not creates directory
        Args:
        dirname (str) : name of directory/path to check"""
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname

def get_last_date_of_month(year: int, month: int) -> date:
    """Function from pytrends.dailydata
    Given a year and a month returns an instance of the date class
    containing the last day of the corresponding month.
    Source: https://stackoverflow.com/questions/42950/get-last-day-of-the-month-in-python
    """
    return date(year, month, monthrange(year, month)[1])

def convert_dates_to_timeframe(start: date, stop: date) -> str:
    """Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data for a specific time frame
    from Google Trends.
    """
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"

    
def parse_dates(time_range: str) -> str:
    """Args:
        time_range (str): time range of data to collect in format YYYY-MM-DD YYYY-MM-DD"""
    return [(pd.to_datetime(x).year,pd.to_datetime(x).month) for x in time_range.split(' ')]

def gen_stata(filename,monthly=False):
    """Calls StataMP via subprocess to harmonize the data
    Args: filename (str): the final dataframe saved as csv"""
    out_name = filename.replace('.csv','_harmonized.csv')
    abs_fname = os.path.abspath(filename)
    abs_ofname = os.path.abspath(out_name)
    if monthly:
        fname = 'utils/stata_weekly.do'
    else:
        fname = 'utils/stata_daily.do'
    with open(fname,'r') as inf:
        contents = inf.read()
    contents = contents.replace('INPUT_FILENAME', f'"{abs_fname}"')
    contents = contents.replace('OUTPUT_FILENAME', f'"{abs_ofname}"')
    do_file = filename.replace('.csv','.do')
    with open(do_file, 'w') as of:
        of.write(contents)
    exe_path = '/Applications/Stata/StataMP.app/Contents/MacOS/StataMP'
    command = f'{exe_path} -e {do_file}'.split(' ')
    subprocess.call(command)
    if os.path.exists(abs_ofname):
        print("Successfully harmonized data using Brodeur et al.'s (2021) method !")
    else:
        print('Error ...')
    return out_name


class DataCollector:
    def __init__(self, kw_list, timeframe,geo='',sleep_time=int(3),PROXY=False,verbose=True):
        self.kw_list = kw_list
        self.kws = ','.join(self.kw_list)
        self.timeframe = timeframe
        self.geo = geo
        self.hl = 'en-GB'
        self.verbose = verbose
        self.sleep_time = sleep_time
        self.gen_download_dirs()
        self.service = Service(executable_path=webdriver_path)
        self.proxy = self.init_proxy(PROXY) # If True, uses NORDVPN to connect to Proxies
        

    def init_proxy(self, PROXY):
        """If we want to use NORDVPN Proxy, then we use the module njord to initiate a Client via OpenVPN.
        In utils.nord_vpn_config, the username and password need to be provided, these can be found/generated in your NordVPN account (online) in Services/NordVPN/ManualSetup.
        If not provided, the credentials will be asked via stdin.
        The Client will ask for the sudo password once to make the connections."""
        if PROXY:
            from njord import Client
            from utils.configs import nord_vpn_config
            if nord_vpn_config['username'] and nord_vpn_config['password']:
                username = nord_vpn_config['username']
                password = nord_vpn_config['password']
            else:
                import getpass
                username = getpass.getpass('Username for NORDVPN API')
                password = getpass.getpass('Password for NORDVPN API')
            self.client = Client(username,password)
            try:
                self.client.connect()
                self.test_ip()
            except Exception as e:
                print(f'Error with connection : {e}')
                return None
        else:
            return None
    
    def test_ip(self):
        """Tests that the IP Address of the Proxy (NordVPN Server) is correctly set"""
        self.init_browser(download=False,headless=True)
        self.browser.get('http://httpbin.org/ip')
        proxy_ip = self.client.status()['ip']
        if proxy_ip in self.browser.page_source:
            print(f'Successful connection to {proxy_ip}')
        else:
            page_source = self.browser.page_source
            start = page_source.find('origin": "')
            end = page_source.find('"\n}\n')
            current_ip = page_source[start:end]
            print(f'Currently connected to {current_ip} ... The proxy IP is {proxy_ip}')
        self.browser.quit()
        
    def gen_download_dirs(self):
        """Creates directory to save the csv files of Google Trends data"""
        dir_geo = 'world' if self.geo == '' else self.geo
        dir_kw = check_dir(os.path.join('data',self.kws.replace(',','_').replace(' ','_')))
        self.download_path = check_dir(os.path.join(dir_kw, dir_geo))

    def restart(self):
        """Restart the Server if there's errors"""
        self.client.disconnect()
        self.browser.quit()
        self.client.connect()
        self.init_browser()
        if self.current_url:
            self.browser.get(self.current_url)
            time.sleep(float(0.5))
            self.browser.get(self.current_url)
            self.get_cookie_button()


    def gen_url(self, time_range: str) -> str:
        """Creates the url to request csv file in Google Trends
            Args: time_range(str) : time range of data to collect in format YYYY-MM-DD YYYY-MM-DD"""
        if self.geo == '':
            url_ = f'{base_url}date={quote(time_range)}&q={quote(self.kws)}&hl={self.hl}'
        else:
            url_ = f'{base_url}date={quote(time_range)}&geo={self.geo}&q={quote(self.kws)}&hl={self.hl}'
        return url_
    
    def init_browser(self,download=True,headless=False):
        """Initialize the Chrome Webdriver browser with the parameters
        Source: https://ggiesa.wordpress.com/2018/05/15/scraping-google-trends-with-selenium-and-python/"""
        chrome_options = Options()
        if download:
            download_prefs = {'download.default_directory' : os.path.abspath(self.temp_download_path),
                  'download.prompt_for_download' : False,
                  'profile.default_content_settings.popups' : False,
                  'download.directory_upgrade': True,
                  'safebrowsing.enabled': True}
            chrome_options.add_experimental_option('prefs', download_prefs)
        if headless:
            chrome_options.add_argument('--headless=new')
        else:
            chrome_options.add_argument('--window-size=480x270')
        self.browser = webdriver.Chrome(service=self.service,options=chrome_options)

    def get_cookie_button(self):
        """Refresh the page until we get the cookies button"""
        attempts,cookie_button = 0, None
        while not cookie_button:
            try:
                cookie_button = self.browser.find_element(By.CSS_SELECTOR, ".cookieBarButton.cookieBarConsentButton")
                cookie_button.click()
            except Exception as e:
                if self.verbose:
                    print("Couldn't find export button for timeline, refreshing the page...")
                self.browser.refresh()
                time.sleep(self.sleep_time) #We sleep for a bit 
                attempts+=1
                if attempts == 10:
                    self.restart()
                    attempts = 0

    def download_trend_data(self):
        """Refresh the page until we get the download csv button"""
        attempts,button = 0, None
        while not button:
            try:
                button = self.browser.find_element(By.CSS_SELECTOR, '.widget-actions-item.export')
                button.click()
            except Exception as e:
                if self.verbose:
                    print("Couldn't find export button for timeline, refreshing the page...")
                self.browser.refresh()
                time.sleep(self.sleep_time) #We sleep for a bit 
                attempts+=1
                if attempts == 10:
                    self.restart()
                    attempts = 0
    
    def get_low_regions(self):
        """If we download worldwide GeoData, we refresh need to check 'Include low search volume regions' checkbox"""
        attempts,checkbox_element = 0, None
        while not checkbox_element:
            try:
                checkbox_element = self.browser.find_element(By.CSS_SELECTOR, 'md-checkbox[aria-label="Include low search volume regions"]')
                checkbox_element.click()
            except Exception as e:
                if self.verbose:
                    print("Low region checkbox not found, refreshing the page...")
                self.browser.refresh()
                time.sleep(self.sleep_time) #We sleep for a bit 
                attempts+=1
                if attempts == 10:
                    self.restart()
                    attempts = 0
    
    def download_geo_data(self):
        """Download Regional data for timerange"""
        attempts,button_geo =0, None
        while not button_geo:
            try:
                if self.geo == '':
                    self.get_low_regions()
                time.sleep(self.sleep_time)
                fe_geo_chart_element = self.browser.find_element(By.CSS_SELECTOR, '.fe-geo-chart-generated')
                button_geo = fe_geo_chart_element.find_element(By.CSS_SELECTOR, '.widget-actions-item.export')
                button_geo.click()
            except Exception as e:
                if self.verbose:
                    print(".fe-geo-chart-generated element not found, refreshing the page...")
                self.browser.refresh()
                time.sleep(self.sleep_time)  #We sleep for a bit 
                attempts+=1
                if attempts == 10:
                    self.restart()
                    attempts = 0

    def _fetch_data(self, url) :
        """Attempts to fecth data and retries in case of a ResponseError.
        Source adapted from: https://github.com/GeneralMills/pytrends/blob/master/pytrends/dailydata.py """
        self.current_url = url
        self.browser.get(url)
        #self.browser.minimize_window()
        time.sleep(float(0.5))
        self.browser.get(url)
        self.get_cookie_button()
        self.download_trend_data()
        time.sleep(self.sleep_time)
        self.download_geo_data()
        time.sleep(self.sleep_time)
    
    def _fetch_data_no_geo(self, url) :
        """Attempts to fecth data and retries in case of a ResponseError. Does NOT return geo data
        Source adapted from: https://github.com/GeneralMills/pytrends/blob/master/pytrends/dailydata.py """
        self.current_url = url
        self.browser.get(url)
        #self.browser.minimize_window()
        time.sleep(float(0.5))
        self.browser.get(url)
        self.get_cookie_button()
        self.download_trend_data()
        time.sleep(self.sleep_time)

    def get_data(self,time_range):
        """Download the csv file from Google Trends and rename the file to the corresponding time range.
        Args: 
            time_range (str) : the time_range to download the data in format YYYY-MM-DD YYYY-MM-DD
        Source: https://ggiesa.wordpress.com/2018/05/15/scraping-google-trends-with-selenium-and-python/"""
        url = self.gen_url(time_range)
        self.init_browser()
        self._fetch_data(url)
        self.browser.quit()
        data_dir = check_dir(os.path.join(self.temp_download_path,'data'))
        geo_dir = check_dir(os.path.join(self.temp_download_path,'geo'))
        new_name = os.path.join(data_dir,f"{time_range.replace(' ','_')}_timeline.csv")
        os.rename(os.path.join(self.temp_download_path,'multiTimeline.csv'),new_name)
        new_name_geo = os.path.join(geo_dir,f"{time_range.replace(' ','_')}_geo_timeline.csv")
        os.rename(os.path.join(self.temp_download_path,'geoMap.csv'),new_name_geo)
        print(f'Success for {time_range}')

    def get_data_no_geo(self,time_range):
        """Download the csv file from Google Trends and rename the file to the corresponding time range.
        Args: 
            time_range (str) : the time_range to download the data in format YYYY-MM-DD YYYY-MM-DD
        Source: https://ggiesa.wordpress.com/2018/05/15/scraping-google-trends-with-selenium-and-python/"""
        url = self.gen_url(time_range)
        self.init_browser()
        self._fetch_data_no_geo(url)
        self.browser.quit()
        data_dir = check_dir(os.path.join(self.temp_download_path,'data'))
        new_name = os.path.join(data_dir,f"{time_range.replace(' ','_')}_timeline.csv")
        os.rename(os.path.join(self.temp_download_path,'multiTimeline.csv'),new_name)
        print(f'Success for {time_range}')
    

    def get_weekly_monthly(self, start_date, stop_date):
        ##If the time range is bigger than 1890 days, we need to perform additional harmonization
        if ((stop_date-start_date).days) > 1889:
            self.temp_download_path = check_dir(os.path.join(self.download_path,'monthly'))
            self.get_data_no_geo(convert_dates_to_timeframe(start_date, stop_date)) #get full df
        else:
            self.temp_download_path = check_dir(os.path.join(self.download_path,'weekly'))
            self.get_data_no_geo(convert_dates_to_timeframe(start_date, stop_date)) #get full df
        self.temp_download_path = check_dir(os.path.join(self.download_path,'weekly'))
        current = start_date
        if ((stop_date-start_date).days) > 1889:
            rounds = round(((stop_date-start_date).days)/1889)
            for _ in range(rounds):
                new_stop = current+timedelta(int(1889))#using int() because some of us are using SageMath and SageMath's Integer is not recognized
                new_stop = new_stop - timedelta(int(new_stop.day+1))
                if new_stop > stop_date:
                    new_stop = stop_date
                self.get_data_no_geo(convert_dates_to_timeframe(current,new_stop))
                current = new_stop+timedelta(int(1))#using int() because some of us are using SageMath and SageMath's Integer is not recognized
        else:
            pass
    

    def get_geo_daily(self,event_date):
        event_date = pd.to_datetime(event_date)
        start,end = event_date-timedelta(int(15)),event_date+timedelta(int(15))
        self.temp_download_path = check_dir(os.path.join(self.download_path,'daily'))
        geo_dir = check_dir(os.path.join(self.temp_download_path,'geo_event'))
        while start != end:
            timeframe = convert_dates_to_timeframe(start,start+timedelta(int(1)))
            url = self.gen_url(timeframe)
            self.current_url = url
            self.init_browser()
            self.browser.get(url)
            time.sleep(float(0.5))
            self.browser.get(url)
            self.get_cookie_button()
            self.download_geo_data()
            time.sleep(self.sleep_time)
            self.browser.quit()
            start = start+timedelta(int(1))
            new_name_geo = os.path.join(geo_dir,f"{timeframe.replace(' ','_')}_geo_timeline.csv")
            os.rename(os.path.join(self.temp_download_path,'geoMap.csv'),new_name_geo)
            print(f'Success for {timeframe}')
    
    def iter_timerange(self):
        """Iterate over the entire timerange given and get the data for:
        1. Weekly values for the entire timeframe
        2. Daily values by splitting the timeframe into months
        We save the files to the corresponding directory (categorized by Keyword and Region)
        Adapted from: https://github.com/GeneralMills/pytrends/blob/master/pytrends/dailydata.py"""
        # Set up start and stop dates
        start,end = parse_dates(self.timeframe)
        start_date = date(*start, 1) 
        stop_date = get_last_date_of_month(*end)
        self.get_weekly_monthly(start_date,stop_date)
        current = start_date
        self.temp_download_path = check_dir(os.path.join(self.download_path,'daily'))  #we save the csv files into a 'daily' director
        while current < stop_date:
            last_date_of_month = get_last_date_of_month(current.year, current.month)
            timeframe = convert_dates_to_timeframe(current, last_date_of_month)
            self.get_data_no_geo(timeframe)
            current = last_date_of_month + timedelta(days=int(1)) #using int(1) because some of us are using SageMath and SageMath's Integer is not recognized
            time.sleep(int(1))  # don't go too fast or Google will send 429s




class DataCleaner:
    def __init__(self,kw_list, timeframe,regions):
        """Cleans the downloaded data
        Args: 
            kw_list (list) - list of keywords
            time_range (str) - time range to collect data for format : 'YYYY-MM-DD YYYY-MM-DD'
            regions (list) - List of geo regions used to collect data"""
        self.kw_list = kw_list
        self.kws = ','.join(self.kw_list)
        self.base_dir = os.path.join('data',self.kws.replace(',','_').replace(' ','_'))
        self.timeframe = timeframe
        self.regions = regions
        self.start, self.end = parse_dates(timeframe)
    
    def gen_download_dirs(self,geo):
        """Creates directory to save the csv files of Google Trends data"""
        self.download_path = check_dir(os.path.join(self.base_dir, geo))
        self.daily_path = os.path.join(self.download_path,'daily','data')
        self.weekly_path = os.path.join(self.download_path,'weekly','data')
        self.clean_dir = check_dir(os.path.join(os.path.dirname(self.download_path), 'clean'))

    def multikeyword_clean(self,df):
        """Cleans the dataframe and renames the columns, 
        this function accepts dataframes with multiple keywords and dataframes that have Day/Week/Month data"""
        if df.columns.values[0] == 'Month':
            tail = '_monthly'
        elif df.columns.values[0] == 'Week':
            tail = '_weekly'
        elif df.columns.values[0] == 'Day':
            tail = '_daily'
        for colname in df.columns.values[1:]:
            if len(self.kw_list)  == 1:
                new_name = 'raw'+tail
            else:
                new_name = colname.split(':')[0].replace(' ','_')+tail
            df.rename(columns={colname:new_name},inplace=True)
        df = df.rename(columns={'Day':'date','Week':'date','Month':'date'})
        return df
        
    def load_df(self, filepath):
        """Loads the csv file and runs multikeyword_clean automatically"""
        df = pd.read_csv(filepath,skiprows=int(2))
        df = self.multikeyword_clean(df)
        df['date']= pd.to_datetime(df['date'].apply(lambda x: x.replace('.','-')))
        return df
    
    def load_directory(self,input_path):
        dfs = []
        for file in os.listdir(input_path):
            if file[-3:] == 'csv':
                dfs.append(self.load_df(os.path.join(input_path,file)))
        df = pd.concat(dfs,ignore_index=True)
        return df

    def merge_dfs(self,df1,df2):
        df1.set_index('date',inplace=True)
        df2.set_index('date',inplace=True)
        complete = df1.join(df2, how='outer')
        complete.ffill(inplace=True)
        complete = complete.dropna()
        complete = complete.astype(str).replace('<1','1').astype(float)
        return complete

    def get_complete(self):
        df_weekly = self.load_directory(self.weekly_path)
        df_daily = self.load_directory(self.daily_path)
        complete = self.merge_dfs(df_daily,df_weekly)
        return complete

    def df_to_stata(self, complete, kw, geo,is_monthly=False):
        if is_monthly:
            kw = 'monthly_'+kw
        output_name = os.path.join(self.clean_dir,f'{kw}_{geo}_complete.csv')
        complete.to_csv(output_name)
        stata_file = gen_stata(output_name,monthly=is_monthly)
        keyword_df = pd.read_csv(stata_file)
        keyword_df['date']= pd.to_datetime(keyword_df['date'].apply(lambda x: x.replace('.','-')))
        keyword_df['region'] = geo
        keyword_df['keyword'] = kw
        return keyword_df

    def multi_keyword(self,geo):
        multi_complete = self.get_complete()
        keyword_dfs = []
        for kw in self.kw_list:
            kw = kw.replace(' ','_')
            col_names = [x for x in multi_complete.columns if kw in x]
            partial_split = multi_complete[col_names]
            renames = {x:x.replace(kw,'raw') for x in col_names}
            partial_split = partial_split.rename(columns=renames)
            is_monthly =  'raw_monthly' in partial_split.columns
            keyword_df = self.df_to_stata(partial_split, kw, geo,is_monthly=is_monthly)
            keyword_dfs.append(keyword_df)
        full_df = pd.concat(keyword_dfs,ignore_index=True)
        return full_df

    def single_keyword(self,geo):
        complete = self.get_complete()
        output_df = self.df_to_stata(complete, self.kws, geo )
        return output_df

    def do_monthly(self,geo):
        monthly_path = os.path.join(self.download_path,'monthly','data')
        df_weekly = self.load_directory(self.weekly_path)
        df_monthly = self.load_directory(monthly_path)
        complete_monthly = self.merge_dfs(df_weekly,df_monthly)
        harmo_week = self.df_to_stata(complete_monthly, self.kws.replace(' ','_'),geo,True)
        df_weekly_week = harmo_week[['date','harmonized_weekly']].rename(columns={'harmonized_weekly':'raw_weekly'})
        df_daily = self.load_directory(self.daily_path)
        complete = self.merge_dfs(df_daily,df_weekly_week)
        output_df = self.df_to_stata(complete, self.kws.replace(' ','_'),geo,False)
        return output_df


    def iter_regions(self,monthly):
        output_dfs = []
        for geo in self.regions:
            self.gen_download_dirs(geo)
            if monthly:
                output_df = self.do_monthly(geo)
                output_dfs.append(output_df)
            else:
                if len(self.kw_list) == 1:
                    output_df = self.single_keyword(geo)
                    output_dfs.append(output_df)
                else:
                    output_df = self.multi_keyword(geo)
                    output_dfs.append(output_df)
        return output_dfs
