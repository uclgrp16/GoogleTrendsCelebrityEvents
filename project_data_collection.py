"""Code written by uclqevo@ucl.ac.uk for Academic project at UCL 
based on pytrends, https://ggiesa.wordpress.com/2018/05/15/scraping-google-trends-with-selenium-and-python/, 
Bordeur et al (2021) and https://github.com/jack-madison/Google-Trends"""

from data_collector import DataCollector, DataCleaner

### Download & Clean Data simultaneously

def gather_euros(keyword):
    geo_values = ['US','GB','']
    for region in geo_values:
        col =  DataCollector(keyword, '2020-01-01 2022-12-31',geo=region,PROXY=True)
        col.iter_timerange()
    cleaner = DataCleaner(keyword,'2020-01-01 2022-12-31',geo_values)
    cleaner.iter_regions(monthly=False)

def gather_coca_cola():
    geo_values = ['US','GB','']
    for region in geo_values:
        col =  DataCollector(['Coca Cola'], '2015-01-01 2022-12-31',geo=region,PROXY=True)
        col.iter_timerange()
    cleaner = DataCleaner(['Coca Cola'],'2015-01-01 2022-12-31',geo_values)
    cleaner.iter_regions(monthly=True)

#### Older data was collected with timeframe 2021-01-01 2021-12-31 for all keywords, and additionally 2016-01-01 2016-12-31 for Coca Cola

if __name__ == "__main__":
    keyword_lists = [['Pepsi'],['Coca Cola','Pepsi'],['Heineken'],['Heineken','Budweiser'],['Budweiser']]
    for keyword in keyword_lists:
        gather_euros(keyword)
    gather_coca_cola()
