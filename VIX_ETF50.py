'''VIX ETF50'''
#参考文档：ETF 期权研究系列之七：VIX 编制原理与方法（华宝证券）
#编写思路：
#第一部分：数据处理，主要是格式转化，以及用不到的数据的剔除
#第二部分：成分期权的选择（算出远期价格后找到对应的执行价，以执行价对应筛选近月和远月期权）
#第三部分：日度波动率和VIX指数的计算

#需要注意或可以调整的地方包括：
#1，近月合约的选取默认为当月（benchmark_months=1）
#2，近月合约距离到期日小于七日则顺延一月（distance_threshold=7）
#2，远月的范围默认为次月（deferred_months=1）
#3，数据起始年份默认为2020（baseline_year = 2020）
#4，无风险利率设定默认为近月合约SHIBOR_1M，远月合约SHIBOR_3M
#5，一年天数的设定默认为365（d=365）

import numpy as np
import pandas as pd

data_origin = pd.read_csv('data_option.csv')
data_shibor_origin = pd.read_csv('data_SHIBOR_2020.csv')
data = data_origin.copy() #将原数据作为备份
data_shibor = data_shibor_origin.copy()
data_shibor = data_shibor.rename(columns={'date':'trading_day'})

''' 预处理'''
#保留需要的列
data = data[['trading_day','exe_enddate','instrument','exe_mode','exe_price','close']].copy()

'''日期预处理'''
#格式转化（int转datatime）
def data_int2dt(data,columns_name): #输入数据包括(df，需要转格式的列名/list格式输入)
    for i in columns_name:
        temp = str(data[i].tolist())
        temp = temp[1:-1] #删掉多余的符号
        temp = temp.split(', ')   
        data[i] = pd.to_datetime(temp,format="%Y/%m/%d")
    return data
data = data_int2dt(data,['trading_day','exe_enddate'])
data_shibor = data_int2dt(data_shibor,['trading_day'])
data = data.merge(data_shibor,on=['trading_day'])
#提取年份、月份
data['trading_year'] = data['trading_day'].map(lambda x:x.year)
data['trading_month'] = data['trading_day'].map(lambda x:x.month)
data['exe_end_year'] = data['exe_enddate'].map(lambda x:x.year)
data['exe_end_month'] = data['exe_enddate'].map(lambda x:x.month)
#结算日数据df，命名考虑主要是为了方便后续合并
exe_list = list(set(data['exe_enddate'].tolist()))
exe_df = pd.DataFrame({'exe_date':exe_list})
exe_df['trading_year'] = exe_df['exe_date'].map(lambda x:x.year)
exe_df['trading_month'] = exe_df['exe_date'].map(lambda x:x.month)
#合并df
data = pd.merge(data,exe_df,on=['trading_year','trading_month'])
del exe_df

'''判断近月/远月'''
#数据起始年份
baseline_year = 2020
#距离执行日少于七天的合约不再考虑在内
distance_threshold = 7
#设立近月和远月的月份数
benchmark_months = 1
deferred_months = 1
data['maturity'] = (data['exe_enddate'] - data['trading_day'])/pd.Timedelta(days=1)
#与当月执行日的距离
data['nearest_distance'] = (data['trading_day'] - data['exe_date'])/pd.Timedelta(days=1)
#判断是否小于阈值, True表示当月结算合约已不应被算入近月合约
data['is_threshold'] = data['nearest_distance']>-distance_threshold
#根据起始年份做月份调整
data['trading_month'] = data['trading_month']+12*(data['trading_year']-baseline_year)
data['exe_end_month'] = data['exe_end_month']+12*(data['exe_end_year']-baseline_year)
#合约月份阈值
data['bmmonth_threshold'] = data['trading_month']+benchmark_months-1
data['bmmonth_threshold'][data['is_threshold']==True]=data['bmmonth_threshold']+1
data['dmonth_threshold']=data['bmmonth_threshold']+deferred_months
#判断是否近月/远月，是则为True，否则为False;
data['is_benchmark'] = (data['exe_end_month']<=data['bmmonth_threshold']) * (data['maturity']>(distance_threshold-1)) 
data['is_deferred'] = (data['is_benchmark']==False) * (data['exe_end_month']<=\
                       data['dmonth_threshold'])* (data['maturity']>(distance_threshold-1))
#只保留需求的列
data = data[['trading_day','exe_enddate','maturity','instrument','exe_mode','exe_price',\
             'close','is_benchmark','is_deferred','1M','3M']].copy()

'''近月合约成分计算'''
#设定无风险利率和一年天数
d = 365
#重新整理df，计算同一执行价格下的价差
data_call = data[data['exe_mode']=='认购'].copy()
data_put = data[data['exe_mode']=='认沽'].copy()
data_merge = pd.merge(data_call,data_put,on=['trading_day','exe_enddate','exe_price','is_benchmark',\
                                             'is_deferred','maturity','1M','3M'])
data_merge['spread'] = np.abs(data_merge['close_x']-data_merge['close_y'])

#选取近月合约，保留价差最小的期权
data_benchmark = data_merge[data_merge['is_benchmark']].copy()
data_benchmark['min_spread'] = data_benchmark.groupby('trading_day')['spread'].transform('min')
#保留价差最小期权的行
data_spread_bm = data_benchmark[data_benchmark['spread']==data_benchmark['min_spread']]
#计算远期价格（日度复利）
data_spread_bm['discount_factor'] = np.exp(data_spread_bm['1M']/d*data_spread_bm['maturity'])
data_spread_bm['future_0'] = data_spread_bm['exe_price'] - \
                           data_spread_bm['discount_factor'] * data_spread_bm['spread']
#保留需要的部分，并入data
data_spread_bm = data_spread_bm[['trading_day','future_0']].copy()
data = pd.merge(data,data_spread_bm,on=['trading_day'])

'''远月合约成分计算'''
#与近月合约同理
data_deferred = data_merge[data_merge['is_deferred']].copy()
data_deferred['min_spread'] = data_deferred.groupby('trading_day')['spread'].transform('min')
data_spread_d = data_deferred[data_deferred['spread']==data_deferred['min_spread']]
data_spread_d['discount_factor'] = np.exp(data_spread_d['3M']/d*data_spread_d['maturity'])
data_spread_d['future_1'] = data_spread_d['exe_price'] - \
                           data_spread_d['discount_factor'] * data_spread_d['spread']
data_spread_d = data_spread_d[['trading_day','future_1']].copy()
data = pd.merge(data,data_spread_d,on=['trading_day'])
del data_call,data_put,data_merge,data_benchmark,data_deferred,data_spread_bm,data_spread_d

'''删除不满足条件的成分合约'''
#合并future，执行价为低于future的最高执行价
#剔除执行价大于future的认购和小于future的认沽，删除不需要的列
data['future'] = data['future_0'] * data['is_benchmark'] + \
                 data['future_1'] * data['is_deferred']
#计算执行价
#大于future置为负数，选出小于future的最大执行价
data['temp'] = (data['exe_price'] * ((data['future'] - data['exe_price'])>=0)) + \
               ( -100 * ((data['future'] - data['exe_price'])<0))
data['exe0'] = data.groupby('future')['temp'].transform('max')
#循环用于检查是否有特殊情况
future_list = list(set(data['future'].tolist()))
for i in future_list:
    if data[data['future']==i]['exe0'].max()==-100:
        data['exe0'] = data['exe0']+(data['future']==i)*(100+data['future'])
del future_list
#计算是否为成分股并仅保留成分股
data['is_con'] = (data['exe_mode']=='认购') * (data['exe_price']>=data['exe0']) * (data['exe0']>0) + \
                 (data['exe_mode']=='认沽') * (data['exe_price']<=data['exe0'])
data = data[data['is_con']==True].copy()
#保留需要的列
data = data[['trading_day','maturity','instrument','exe_mode','exe_price','close',\
             'is_benchmark','is_deferred','future','exe0','1M','3M']].copy()

'''计算波动率和VIX'''
#获取日期列表并排序
trading_day_list = list(set(data['trading_day'].tolist()))
trading_day_list.sort()
#日度波动率计算函数定义，输入期权序列，无风险利率和年度天数，输出sigma和T1
def sigma_daily(df,d):
    df.sort_values(by='exe_price', inplace=True, ascending=True)
    df = df.reset_index(drop = True)
    strike_index = df[df['exe_price']==df['exe0']].index.tolist() 
    #判断执行价上是否只有一个合约，有多个合约时取均值，将均值代入第一行收盘价，并只保留第一行
    if len(strike_index)>1:
        index_a,index_b = strike_index[0],strike_index[1]
        df['close'][index_a] = (df['close'][index_a]+df['close'][index_b])/2
        df = df.drop(index = [index_b])
    #计算执行价差序列
    l=len(df)
    df['i-1'] = df['exe_price'].shift(-1) 
    df['i+1'] = df['exe_price'].shift(1)
    df['i-1'][0], df['i+1'][0] = df['i-1'][0]*2, df['exe_price'][0]*2
    df['i-1'][l-1], df['i+1'][l-1] = df['exe_price'][l-1]*2, df['i+1'][l-1]*2
    df['delta_K'] = (df['i-1']-df['i+1'])/2
    df['temp_K'] = df['delta_K']/(df['exe_price'] ** 2) * df['close']
    df['temp_K'] = df['temp_K'].astype(np.double)
    #提取相关参数
    T1,F1,K0,r= df['maturity'][0],df['future'][0],df['exe0'][0],df['r'][0]
    #计算sigma
    sigma = 2*(df['temp_K'].sum())*np.exp(r*T1/d)/T1-(F1/K0-1)**2/T1
    return sigma,T1
#日度循环计算
#设列表装结果
VIX_list = list()
for i in trading_day_list:
    temp = data[data['trading_day']==i].copy()
    #计算近月
    temp_bm = temp[temp['is_benchmark']].copy()
    temp_bm = temp_bm.rename(columns={'1M':'r'})
    sigma_bm,T1 = sigma_daily(temp_bm,d)  
    #计算远月
    temp_d = temp[temp['is_deferred']].copy()
    temp_d = temp_d.rename(columns={'3M':'r'})
    sigma_d,T2 = sigma_daily(temp_d,d)   
    #计算VIX
    VIX = 100*((T1*sigma_bm*((T2-30)/(T2-T1))+T2*sigma_d*((30-T1)/(T2-T1)))*365/30)**(0.5)
    VIX_list.append(VIX)
VIX_df = pd.DataFrame({'trading_day':trading_day_list,'VIX':VIX_list})
VIX_df = VIX_df.set_index(['trading_day'])
