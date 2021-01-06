'''VIX ETF50'''
#Reference：http://doc.xueqiu.com/14b76e809d826d3fdce4f79d.pdf
#Code Intorduction：
#Part 1：Data Preprocessing (Data format's transformation and redundant data's elimination)
#Part 2：Component Options Selecting（future price → strike price → selection)
#Part 3：Calculations of daily volatility and ETF50's VIX

#Some default settings:
#1，Contracts of nearby month are contracts settled in current month（benchmark_months=1）;
#2，When current trading day is less than seven days from current month's settlement day, all contracts are extended for one month（distance_threshold=7）;
#2，Contracts of deferred month are contracts settled in next month（deferred_months=1）;
#3，The year data start is 2020（baseline_year = 2020）;
#4，Rf risk: nearby month contract - SHIBOR_1M，deferred month contract - SHIBOR_3M;
#5，365 days in one year（d=365）;

import numpy as np
import pandas as pd

data_origin = pd.read_csv('data_option.csv')
data_shibor_origin = pd.read_csv('data_SHIBOR_2020.csv')
data = data_origin.copy() #For convenience only
data_shibor = data_shibor_origin.copy()
data_shibor = data_shibor.rename(columns={'date':'trading_day'})

#--------------------------------------------------------------------------------------------------------------------PART 1
#Keep the columns we want 
data = data[['trading_day','exe_enddate','instrument','exe_mode','exe_price','close']].copy()

#dtype transformation（int2datatime）
def data_int2dt(data,columns_name): 
    for i in columns_name:
        temp = str(data[i].tolist())
        temp = temp[1:-1] 
        temp = temp.split(', ')   
        data[i] = pd.to_datetime(temp,format="%Y/%m/%d")
    return data
data = data_int2dt(data,['trading_day','exe_enddate'])
data_shibor = data_int2dt(data_shibor,['trading_day'])
data = data.merge(data_shibor,on=['trading_day'])
#Extract year and month
data['trading_year'] = data['trading_day'].map(lambda x:x.year)
data['trading_month'] = data['trading_day'].map(lambda x:x.month)
data['exe_end_year'] = data['exe_enddate'].map(lambda x:x.year)
data['exe_end_month'] = data['exe_enddate'].map(lambda x:x.month)
#Extract execution date (month and day)
exe_list = list(set(data['exe_enddate'].tolist()))
exe_df = pd.DataFrame({'exe_date':exe_list})
exe_df['trading_year'] = exe_df['exe_date'].map(lambda x:x.year)
exe_df['trading_month'] = exe_df['exe_date'].map(lambda x:x.month)
#merge df
data = pd.merge(data,exe_df,on=['trading_year','trading_month'])
del exe_df

#judge whether is a nearby month contract or a deferred one or neither 
#settings are described at the begining
baseline_year = 2020
distance_threshold = 7
benchmark_months = 1
deferred_months = 1
data['maturity'] = (data['exe_enddate'] - data['trading_day'])/pd.Timedelta(days=1)
data['nearest_distance'] = (data['trading_day'] - data['exe_date'])/pd.Timedelta(days=1)
data['is_threshold'] = data['nearest_distance']>-distance_threshold
#modify data of month according to the starting year 
data['trading_month'] = data['trading_month']+12*(data['trading_year']-baseline_year)
data['exe_end_month'] = data['exe_end_month']+12*(data['exe_end_year']-baseline_year)
#threshold caculation
data['bmmonth_threshold'] = data['trading_month']+benchmark_months-1
data['bmmonth_threshold'][data['is_threshold']==True]=data['bmmonth_threshold']+1
data['dmonth_threshold']=data['bmmonth_threshold']+deferred_months
#judgement
data['is_benchmark'] = (data['exe_end_month']<=data['bmmonth_threshold']) * (data['maturity']>(distance_threshold-1)) 
data['is_deferred'] = (data['is_benchmark']==False) * (data['exe_end_month']<=\
                       data['dmonth_threshold'])* (data['maturity']>(distance_threshold-1))

data = data[['trading_day','exe_enddate','maturity','instrument','exe_mode','exe_price',\
             'close','is_benchmark','is_deferred','1M','3M']].copy()

#--------------------------------------------------------------------------------------------------------------------PART 2
#Component Options Selecting
d = 365
data_call = data[data['exe_mode']=='认购'].copy()
data_put = data[data['exe_mode']=='认沽'].copy()
data_merge = pd.merge(data_call,data_put,on=['trading_day','exe_enddate','exe_price','is_benchmark',\
                                             'is_deferred','maturity','1M','3M'])
data_merge['spread'] = np.abs(data_merge['close_x']-data_merge['close_y'])
#Select the strike price at which the pair of call option and put option has minimum price spread 
#Nearby month contract
data_benchmark = data_merge[data_merge['is_benchmark']].copy()
data_benchmark['min_spread'] = data_benchmark.groupby('trading_day')['spread'].transform('min')
data_spread_bm = data_benchmark[data_benchmark['spread']==data_benchmark['min_spread']]
#calculate future's price with the strike price and the price spread
data_spread_bm['discount_factor'] = np.exp(data_spread_bm['1M']/d*data_spread_bm['maturity'])
data_spread_bm['future_0'] = data_spread_bm['exe_price'] - \
                           data_spread_bm['discount_factor'] * data_spread_bm['spread']
data_spread_bm = data_spread_bm[['trading_day','future_0']].copy()
data = pd.merge(data,data_spread_bm,on=['trading_day'])

#deferred contract
data_deferred = data_merge[data_merge['is_deferred']].copy()
data_deferred['min_spread'] = data_deferred.groupby('trading_day')['spread'].transform('min')
data_spread_d = data_deferred[data_deferred['spread']==data_deferred['min_spread']]
data_spread_d['discount_factor'] = np.exp(data_spread_d['3M']/d*data_spread_d['maturity'])
data_spread_d['future_1'] = data_spread_d['exe_price'] - \
                           data_spread_d['discount_factor'] * data_spread_d['spread']
data_spread_d = data_spread_d[['trading_day','future_1']].copy()
data = pd.merge(data,data_spread_d,on=['trading_day'])
del data_call,data_put,data_merge,data_benchmark,data_deferred,data_spread_bm,data_spread_d

#Calculate the maximum strike price(K0) that below the future price
data['future'] = data['future_0'] * data['is_benchmark'] + \
                 data['future_1'] * data['is_deferred']
data['temp'] = (data['exe_price'] * ((data['future'] - data['exe_price'])>=0)) + \
               ( -100 * ((data['future'] - data['exe_price'])<0))
data['exe0'] = data.groupby('future')['temp'].transform('max')
#examine if there is any specials
future_list = list(set(data['future'].tolist()))
for i in future_list:
    if data[data['future']==i]['exe0'].max()==-100:
        data['exe0'] = data['exe0']+(data['future']==i)*(100+data['future'])
del future_list
#select component options
data['is_con'] = (data['exe_mode']=='认购') * (data['exe_price']>=data['exe0']) * (data['exe0']>0) + \
                 (data['exe_mode']=='认沽') * (data['exe_price']<=data['exe0'])
data = data[data['is_con']==True].copy()
data = data[['trading_day','maturity','instrument','exe_mode','exe_price','close',\
             'is_benchmark','is_deferred','future','exe0','1M','3M']].copy()

#--------------------------------------------------------------------------------------------------------------------PART 3
trading_day_list = list(set(data['trading_day'].tolist()))
trading_day_list.sort()

def sigma_daily(df,d):
    df.sort_values(by='exe_price', inplace=True, ascending=True)
    df = df.reset_index(drop = True)
    strike_index = df[df['exe_price']==df['exe0']].index.tolist() 
    #if there are more than one contracts (normally two) at strike price K0, the close price is the mean of all contracts' close prices.
    if len(strike_index)>1:
        index_a,index_b = strike_index[0],strike_index[1]
        df['close'][index_a] = (df['close'][index_a]+df['close'][index_b])/2
        df = df.drop(index = [index_b])
    l=len(df)
    df['i-1'] = df['exe_price'].shift(-1) 
    df['i+1'] = df['exe_price'].shift(1)
    df['i-1'][0], df['i+1'][0] = df['i-1'][0]*2, df['exe_price'][0]*2
    df['i-1'][l-1], df['i+1'][l-1] = df['exe_price'][l-1]*2, df['i+1'][l-1]*2
    df['delta_K'] = (df['i-1']-df['i+1'])/2
    df['temp_K'] = df['delta_K']/(df['exe_price'] ** 2) * df['close']
    df['temp_K'] = df['temp_K'].astype(np.double)
    T1,F1,K0,r= df['maturity'][0],df['future'][0],df['exe0'][0],df['r'][0]
    sigma = 2*(df['temp_K'].sum())*np.exp(r*T1/d)/T1-(F1/K0-1)**2/T1
    return sigma,T1
VIX_list = list()
for i in trading_day_list:
    temp = data[data['trading_day']==i].copy()
    #nearby month
    temp_bm = temp[temp['is_benchmark']].copy()
    temp_bm = temp_bm.rename(columns={'1M':'r'})
    sigma_bm,T1 = sigma_daily(temp_bm,d)  
    #deferred
    temp_d = temp[temp['is_deferred']].copy()
    temp_d = temp_d.rename(columns={'3M':'r'})
    sigma_d,T2 = sigma_daily(temp_d,d)   
    #VIX
    VIX = 100*((T1*sigma_bm*((T2-30)/(T2-T1))+T2*sigma_d*((30-T1)/(T2-T1)))*365/30)**(0.5)
    VIX_list.append(VIX)
VIX_df = pd.DataFrame({'trading_day':trading_day_list,'VIX':VIX_list})
VIX_df = VIX_df.set_index(['trading_day'])
