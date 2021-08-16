#!/usr/bin/env python
# coding: utf-8

# # Notes

# **TO DO**
# - 

# This script replicates the Fama French 5 (2x2x2x2) risk factors SMB, HML, RMW, and CMA, in addition to the excess market risk factor, and augments the factor set with a candidate factor based on measures pass by the user in a file or a a list of files. The data come from CRSP for pricing related items and Compustat for fundamental data. The data are accessed through WRDS.
# 
# This script has been adapted from the Fama French 3 factors script posted on WRDS, written by Qingyi (Freda) Song Drechsler in April 2018 and updated in June 2020.<br>(https://wrds-www.wharton.upenn.edu/pages/support/applications/python-replications/fama-french-factors-python/<br>https://www.fredasongdrechsler.com/full-python-code/fama-french)"

# Research notes:
# 
# - only ordinary common stocks (CRSP sharecode 10 and 11) in NYSE, AMEX and NASDAQ (exchange code 1,2,3) and at least 2 years on Compustat are included in the sample (Fama and French (1993, 2015); https://wrds-www.wharton.upenn.edu/pages/support/applications/risk-factors-and-industry-benchmarks/fama-french-factors/).
# - all the breakpoints are computed only on NYSE stocks (from the sample).
# - market cap is calculated at issue-level (permno in CRSP), and book value of equity is calculated at company level (permco in Compustat), it is needed to aggregate market cap at company level (permco in CRSP) for later book-to-market value calculation. And market cap of companies at December of year t-1 is used for portfolio formation at June of year t. Details on how to link CRSP and Compustat:<br>
#     https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/crspcompustat-merged-ccm/wrds-overview-crspcompustat-merged-ccm/<br>https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/
# - there were cases when the same firm (CRSP permco) had two or more securities (CRSP permno) on the same date. For the purpose of ME for the firm, I aggregated all ME for a given CRSP permco, date. This aggregated ME was assigned to the CRSP permno according to the following criteria largest market equity (ME), higher number of years on Compustat (count) (as recommended by WRDS (https://wrds-www.wharton.upenn.edu/pages/support/applications/risk-factors-and-industry-benchmarks/fama-french-factors/) and finally random, in this order. If the ME and years on Compustat are the same there is no other unbiased criteria but random (one would select the one with either largest or smallest return). However these cases are less than 100. The ME to assign to the permco is the sum of the ME of all the permno of that permco.
# - the relevant share code for Fama French factors constructions are 10 and 11 (ordinary common stocks). The permno for the same permco may have different share code (shrcd), filtering them before applying the logic o the previous point would end up in loosing market capitalization. The solution is to delete later, when each permco has only one permno, all the permno with shrcd different from 10 or 11.
# - I merged CRSP and Compustat using the CRSP CCM product (as of April 2010) as recommended by WRDS (https://wrds-www.wharton.upenn.edu/pages/support/applications/risk-factors-and-industry-benchmarks/fama-french-factors/) matching Compustat's gvkey (from calendar year t-1) to CRSP's permno as of June year t. Data was cleaned for unnecessary duplicates. First there were cases when different gvkeys exist for same permno-date. I solved these duplicates by only keeping those cases that are flagged as 'primary' matches by CRSP's CCM (linkprim='P'). There were other unnecessary duplicates that were removed (I kept the oldest gvkey for each permno, finally I randomly picked one gvkey for each of of the about 30 pairs od dupliated permno which were practically identical if not for fractions of decimals differences on certain measures). Some companies on Compustat may have two annual accounting records in the same calendar year. This is produced by change in the fiscal year end during the same calendar year. In these cases, we selected the last annual record for a given calendar year.

# Variable definitions (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html)
# 
# - ME: Market Equity. Market equity (size) is price times shares outstanding. Price is from CRSP, shares outstanding are from Compustat (if available) or CRSP.
# 
# - BE: Equity. Book equity is constructed from Compustat data or collected from the Moody’s Industrial, Financial, and Utilities manuals. BE is the book value of stockholders’ equity, plus balance sheet deferred taxes and investment tax credit (if available), minus the book value of preferred stock. Depending on availability, we use the redemption, liquidation, or par value (in that order) to estimate the book value of preferred stock. Stockholders’ equity is the value reported by Moody’s or Compustat, if it is available. If not, we measure stockholders’ equity as the book value of common equity plus the par value of preferred stock, or the book value of assets minus total liabilities (in that order). See Davis, Fama, and French, 2000, “Characteristics, Covariances, and Average Returns: 1929-1997,” Journal of Finance, for more details.
# 
# - BE/ME: Book-to-Market. The book-to-market ratio used to form portfolios in June of year t is book equity for the fiscal year ending in calendar year t-1, divided by market equity at the end of December of t-1.
#  
# - OP: Operating Profitability. The operating profitability ratio used to form portfolios in June of year t is annual revenues minus cost of goods sold, interest expense, and selling, general, and administrative expense divided by the sum of book equity and minority interest for the last fiscal year ending in t-1.
#  
# - INV: Investment. The investment ratio used to form portfolios in June of year t is the change in total assets from the fiscal year ending in year t-2 to the fiscal year ending in t-1, divided by t-2 total assets.

# Techincal notes:
# 
# - In order to tun the script one has to connect ot the WRDS databases and have a valid WRDS account. Here are the details on how to set up a connection or run the scrip on the WRDS cloud.<br>https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-wrds-cloud/<br>https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/
# - WRDS Python library documentation
# https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/querying-wrds-data-python/

# User guide:
# 
# - Basic
#     - the user has to place one or more measures file (.csv) in the path assigned to the variable nf_measures_path
#     - each measure file can contain multiple measures on the columns (with headers)
#     - each measure file contains on the rows pairs of "jdate" (dates of last day of June in format YYYY-MM-DD) and "permno" (containing Compustat PERMNO). The first two columns in the csv file must be called "permno and "date"
#     - for each measure file passed, the corresponding factors (and additional file for firms count) are saved in a csv file with a similar name, the rows are oranized for date, measure for which robust breakpoints have been choosed, the breakpoints percentailes, measure name
# 
# - Advanced
#     - in the Fama French 5 factors procedure the factors are constructed using as breakpoints 30th and 70th percentiles for B/M, OP, and INV
#     - here the user can specify in the list variable measures_robust_check_bp all the factors for which he or she wants to use alternative breakpoint percentiles (for instance measures_robust_check_bp=['bm', 'inv', 'op', 'nf'] if alternative breakpoints want ot be used for all the factors)
#     - the alernative breakpoints are fixed to be 10th and 90th, 20th and 80th, 30th and 70th, 40th and 60th.

# Descriptions of Fama French 5 factors (2x3) can be found on Kenneth French's website.<br>http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html <br>https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html"

# ![image.png](attachment:image.png)

# References:
# - Fama, Eugene F. and Kenneth R. French, 1993, Common Risk Factors in Stocks and Bonds, Journal of Financial Economics, 33, 3-56.
# - Fama, E.F. and French, K.R., 2015. A five-factor asset pricing model. Journal of financial economics, 116(1), pp.1-22.

# # Script

# In[12]:


# list of all measures for which a factor will be computed
all_measures=['bm','inv','op','nf']
# measures for which all the breakpoints (10_90, 20_80, 30_70, 40_60) subsets are used instead of the default 30_70
# measures_robust_check_bp=['bm', 'inv', 'op', 'nf']
measures_robust_check_bp=[]


# In[13]:


import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt
import urllib
import zipfile
from scipy import stats
import glob
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

#set to 1 to delete intermediate datasets
save_memory=0

#set to 1 to print more check outputs
print_advance_output=0


# ## Functions

# In[14]:


# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


# ## Computation

# In[15]:


project_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\"
factors_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\factors\\"
test_assets_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\test_assets\\"
ff5measures_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\FF5 measures\\"
nf_measures_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\measures\\"


# Read the CRSP table with stock returns produced by FF5_measures_computer_and_FF5_factors_replicator\FF5_measures_computer_and_FF5_factors_replicator.ipynb

# In[16]:


crsp3=pd.read_csv(ff5measures_path+'crsp_returns.csv', sep=',')
crsp3['jdate']=pd.to_datetime(crsp3['jdate'])
# crsp3.set_index(['permno','jdate'], inplace=True)
crsp3


# Read the original FF5 data

# In[17]:


ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
urllib.request.urlretrieve(ff_url,'F-F_Research_Data_5_Factors_2x3_CSV.zip ')
zip_file = zipfile.ZipFile('F-F_Research_Data_5_Factors_2x3_CSV.zip', 'r')
zip_file.extractall()
zip_file.close()
ff_factors_original = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', skiprows = 3)
ff_factors_original = ff_factors_original[0:ff_factors_original.isnull().any(axis=1).idxmax()].rename(columns={'Unnamed: 0':'date'}).set_index('date').astype(float).div(100)
ff_factors_original.index = pd.to_datetime(ff_factors_original.index, format='%Y%m')+MonthEnd(0)
ff_factors_original=ff_factors_original[ff_factors_original.index<=crsp3['date'].max()]
ff_factors_original


# Reading FF5 measures

# In[18]:


ff5_measures=pd.read_csv(ff5measures_path+'FF5_measures.csv', sep=',')
ff5_measures['jdate']=pd.to_datetime(ff5_measures['jdate'])
# ff5_measures.set_index(['permno','jdate'], inplace=True)/
#in order to keep using part of the scritp "FF5_measures_computer_and_FF5_factors_replicator" the info in ff5_measures
#can be used instea of the original ccm_jun. A copy is done to avoid changing variable names later
ccm_jun=ff5_measures.copy()
if save_memory:
    del ff5_measures
ccm_jun


# Clone ccm_jun and crsp3 to preserve the original and recall them at every new computation without reloading them

# In[19]:


ccm_jun_original=ccm_jun.drop(columns=ccm_jun.columns[ccm_jun.columns.str.contains('_ok')], axis=0).copy() #dropped the is_.._ok columns
crsp3_original=crsp3.copy()


# Reading the new factor measures files

# In[20]:


nf_measures_files=glob.glob(nf_measures_path+"*.csv")
nf_measures_files


# In[ ]:


print('##################################################################################################################')
print('##################################################################################################################')
print('### START LOOP ON MEASURES FILES')
print('##################################################################################################################')
print('##################################################################################################################')

# for f in tqdm([nf_measures_files[0]], desc='measures_files_loop'): #DEBUG: SLICING LIST OF MEASURES FILES
for f in tqdm(nf_measures_files, desc='measures_files_loop'): 

    ff_factors_final2export=pd.DataFrame()
    ff_nfirms_final2export=pd.DataFrame()
    
    #read the file
    print('#####################################################################################')
    print("reading file {}".format(f.split('\\')[-1]))
    pd.read_csv(f)
    nf_measures=pd.read_csv(f)
    nf_measures.rename(columns={'date':'jdate'}, inplace=True)
    nf_measures['jdate']=pd.to_datetime(nf_measures['jdate'], format="%Y%m")+MonthEnd(0)    
    nf_measures=pd.merge(ccm_jun_original.loc[:,['permno','jdate']], nf_measures, how='inner', on=['permno','jdate'])
    nf_measures.set_index(['jdate','permno'], inplace=True)

    print('#####################################################################################')
    print('min, max and mean number of stocks with good measure, over periods, for each measure')    
    print(nf_measures.notna().groupby('jdate').sum().agg(['min', 'max', 'mean']).T.astype(int))

    #filter the pairs of stocks and june dates not in common betweeen FF5 measures and New Factor measures
    ccm_jun=pd.merge(ccm_jun_original, nf_measures, how='inner', left_on=['jdate','permno'], right_index=True).loc[:,ccm_jun_original.columns.to_list()]
    print('#####################################################################################')
    print("{} pairs of stocks and june dates in common betweeen FF5 and New Factor".format(ccm_jun.shape[0]))

   # at this point shrcd different from 10 and 11, not valid price and shares and company with less than 2 years of
    # history on Compustat have been removed, even though their conditions are explicitly present here

    # size breakdown
    nyse_sz=ccm_jun[(ccm_jun['exchcd']==1)&
            ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))&
            (~ccm_jun['prc'].isna())&
            (~ccm_jun['shrout'].isna())].groupby(['jdate'])['me'].median().to_frame().reset_index().rename(columns={'me':'sizemedn'})
    nyse_breaks=pd.merge(pd.DataFrame([d-relativedelta(months=1) for d in ff_factors_original.index if d.month==7]).set_index(0), nyse_sz, how='left', left_index=True, right_on='jdate')

    # bp breakpoints
    nyse_bm=ccm_jun[(ccm_jun['exchcd']==1)&
            ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))&
            (~ccm_jun['dec_me'].isna())&
            (ccm_jun['be']>0)].groupby(['jdate'])['beme'].median().to_frame().reset_index().rename(columns={'beme':'bm50'})

    # inv breakdown
    nyse_inv=ccm_jun[(ccm_jun['exchcd']==1)&
            ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))&                 
            (~ccm_jun['inv'].isna())].groupby(['jdate'])['inv'].median().to_frame().reset_index().rename(columns={'inv':'inv50'})

    # op breakdown
    nyse_op=ccm_jun[(ccm_jun['exchcd']==1)&
            ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))&  
            (ccm_jun['be']>0)&                               
            (~ccm_jun['revt'].isna())&
            (~ccm_jun.loc[:,['cogs','xsga','xint']].isna().all(axis=1))].groupby(['jdate'])['op'].median().to_frame().reset_index().rename(columns={'op':'op50'})

    #join back size, beme and inv breakdown
    nyse_breaks = pd.merge(nyse_breaks,pd.merge(pd.merge(nyse_bm, nyse_inv, how='outer', on=['jdate']), nyse_op, how='outer', on=['jdate']), how='left', on=['jdate'])

    # allign breakpoints to variables
    ccm1_jun = pd.merge(ccm_jun, nyse_breaks, how='left', on=['jdate'])
    ccm1_jun['ffyear']=ccm1_jun['jdate'].dt.year


    # Adding flag to the stocks with good FF5 measures

    ccm1_jun.loc[:,'is_size_ok']=np.where((ccm1_jun['me']>0)&
                                          ((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))
                                          ,1,0)

    ccm1_jun.loc[:,'is_beme_ok']=np.where((ccm1_jun['be']>0)&
                                          (ccm1_jun['dec_me'].notna())&
                                          ((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))&
                                          (ccm1_jun['me']>0),1,0)

    ccm1_jun.loc[:,'is_inv_ok']=np.where((ccm1_jun['inv'].notna())&
                                         ((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))&
                                         (ccm1_jun['me']>0),1,0)

    ccm1_jun.loc[:,'is_op_ok']=np.where(((ccm1_jun['be'])>0)&(ccm1_jun['revt'].notna())&
                                        (ccm1_jun.loc[:,['xsga','xint','cogs']].notna().any(axis=1))&
                                        ((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))&
                                        (ccm1_jun['me']>0),1,0)
    

    # Assign portfolios for FF5 measures

    print('#####################################################################################')     
    # assign size portfolios
    print('size breakpoints 50_50')
    ccm1_jun['szport']=np.where((ccm1_jun['is_size_ok']==1), 
                                np.where(ccm1_jun['me']<=ccm1_jun['sizemedn'], 'S', 
                                         np.where(ccm1_jun['me']>ccm1_jun['sizemedn'], 'B',
                                                  np.nan))
                                , np.nan)
        
    # assign book-to-market portfolios 
    print('beme breakpoints 50_50')
    ccm1_jun['bmport']=np.where((ccm1_jun['is_beme_ok']==1), 
                                np.where(ccm1_jun['beme']<=ccm1_jun['bm50'], 'L', 
                                         np.where(ccm1_jun['beme']>ccm1_jun['bm50'], 'H',
                                                  np.nan))
                                , np.nan)  
    
    # assign operating profitability portfolios
    print('op breakpoints 50_50')
    ccm1_jun['opport']=np.where((ccm1_jun['is_op_ok']==1), 
                                np.where(ccm1_jun['op']<=ccm1_jun['op50'], 'W', 
                                         np.where(ccm1_jun['op']>ccm1_jun['op50'], 'R',
                                                  np.nan))
                                , np.nan) 

    # assign investment portfolios
    print('inv breakpoints 50_50')
    ccm1_jun['invport']=np.where((ccm1_jun['is_inv_ok']==1), 
                                np.where(ccm1_jun['inv']<=ccm1_jun['inv50'], 'C', 
                                         np.where(ccm1_jun['inv']>ccm1_jun['inv50'], 'A',
                                                  np.nan))
                                , np.nan)    

    ccm1_jun[ccm1_jun=='nan']=np.nan

    print('##################################################################################################################')
    print('### START LOOP ON MEASURES OF FILE: {}'.format(f.split('\\')[-1]))
    print('##################################################################################################################')

#     nf_measures=nf_measures.iloc[:,3].to_frame() #DEBUG: UNCOMMENT
    for current_measure, m in tqdm(nf_measures.iteritems(), total=nf_measures.shape[1], desc='measures_loop'):  
        print('#########################################################################')
        print("current measure: {}".format(current_measure))
        print('#########################################################################')
        nf_measure_i=m.to_frame().reset_index().rename(columns={current_measure:'nf'}) 
        
        #Join the current measure with FF5 measures, and filter the dates previous the first new factor measure available.
        #First delete columns related to the previous measure 
        ccm1_jun=ccm1_jun.drop(columns=ccm1_jun.columns[ccm1_jun.columns.str.contains('nf')].to_list())
        ccm1_jun=pd.merge(ccm1_jun, nf_measure_i, how='inner', on=['permno','jdate'])

        
        #if the observations for the current measures are all 0, the column is set to nan and continue
        if ccm1_jun['nf'].abs().sum()==0:
            ccm1_jun['nf']=np.nan
            print("##################################################\n!!!!!!!! measures not available !!!!!!!!")
            continue #DEBUG: COMMENT
            
        print('#########################################################################')
        print("ccm1_jun dimension: {} {}".format(ccm1_jun.shape[0],ccm1_jun.shape[1]))
        print('missing measures (%)')
        print(ccm1_jun.loc[:,['beme','inv', 'op', 'nf']].isna().sum().div(ccm1_jun.shape[0], axis=0).round(4)*100)
        print('#########################################################################')

        ccm1_jun.loc[:,'is_nf_ok']=np.where((ccm1_jun['nf'].notna())&
                                             ((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))&
                                             (ccm1_jun['me']>0),1,0)
   
        # at this point shrcd different from 10 and 11, not valid price and shares and company with less than 2 years of
        # history on Compustat have been removed, even though their conditions are explicitly present here

        # nf breakdown
        nyse_nf=ccm1_jun[(ccm1_jun['exchcd']==1)&
                ((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))&                                           
                (~ccm1_jun['nf'].isna())].groupby(['jdate'])['nf'].median().to_frame().reset_index().rename(columns={'nf':'nf50'})

        # allign breakpoints to variables
        ccm1_jun = pd.merge(ccm1_jun, nyse_nf, how='left', on=['jdate'])
        
        #Statistics about stock with available measures

        stocks_good_measures_n=(ccm1_jun.loc[:,['jdate']+ccm1_jun.columns[ccm1_jun.columns.str.contains('_ok')].to_list()].groupby('jdate').sum())
        stocks_good_measures_perc=(ccm1_jun.loc[:,['jdate']+ccm1_jun.columns[ccm1_jun.columns.str.contains('_ok')].to_list()].groupby('jdate').sum()).div(ccm1_jun.groupby('jdate').size(), axis=0).round(4)*100
        print("#########################################################################\nstocks available for each measures (number)")
        print(stocks_good_measures_n)
        print("#########################################################################")        

        # assign new factor portfolios
        print('nf breakpoints 50_50')
        ccm1_jun['nfport']=np.where((ccm1_jun['is_nf_ok']==1), 
                                    np.where(ccm1_jun['nf']<=ccm1_jun['nf50'], 'Y', 
                                             np.where(ccm1_jun['nf']>ccm1_jun['nf50'], 'X',
                                                      np.nan))
                                    , np.nan) 

        ccm1_jun[ccm1_jun=='nan']=np.nan      

        
        # store portfolio assignment as of June
        june=ccm1_jun.loc[:,['permno','ffyear']+ccm1_jun.columns[ccm1_jun.columns.str.contains('port')].to_list()]
        if save_memory:
            del ccm1_jun
        n_ptfs=june.groupby(june.columns[june.columns.str.contains('port')].to_list()).ngroups
        print("{} portfolios available".format(n_ptfs))
      
        # merge back with monthly records
        crsp3 = crsp3_original[['date','permno','retadj','wt','ffyear','jdate']]
        ccm3=pd.merge(crsp3,june, how='left', on=['permno','ffyear'])
        if save_memory:
            del june
            del crsp3
        ccm3.dropna(axis=0, how='all', subset=ccm3.columns[ccm3.columns.str.contains('port')], inplace=True)
        ccm3

        # select the columns that contains portfolios
        cols2keep=[c+'port' for c in all_measures]
        ccm4=ccm3[['jdate','szport','retadj','wt']+cols2keep] 

        # delete observations for which any of the 4 portfolios is not available
        ccm4=ccm4.dropna(axis=0, how='any', subset=ccm4.columns[ccm4.columns.str.contains('port')])   

#         # delete observations for which all of the 4 portfolios are not available
#         ccm4=ccm4.dropna(axis=0, how='all', subset=ccm4.columns[ccm4.columns.str.contains('port')])   
        
        ### calculate the portfolio returns
        ccm4['retadj_X_wt']=ccm4['retadj']*ccm4['wt']
        vwret=ccm4.groupby(['jdate']+ccm4.columns[ccm4.columns.str.contains('port')].to_list())['wt','retadj_X_wt'].sum().reset_index()
        vwret['vwret']=vwret['retadj_X_wt']/vwret['wt']
        vwret['ptf_code']=vwret['szport']+vwret['bmport']+vwret['opport']+vwret['invport']+vwret['nfport']
        vwret        

        # transpose and add missing date (right join with ff_factors_original)
        ff_factors=vwret.pivot(index='jdate', columns='ptf_code', values='vwret').reset_index().rename(columns={'jdate':'date'}).sort_values('date').set_index('date').fillna(0)
        ff_factors=pd.merge(ff_factors_original, ff_factors, how='left', left_index=True, right_index=True).drop(columns=ff_factors_original.columns.to_list())
        
        if ff_factors.shape[1]!=2**(len(all_measures)+1):
            print("##################################################\n!!!!!!!! building block portfolio missing !!!!!!!!")
            continue #DEBUG: COMMENT

        #########################
        # factors construction #
        #########################

        # HML factors
        ff_factors['HML'] = (ff_factors.loc[:,ff_factors.columns.str[1]=='H'].sum(axis=1)-
                             ff_factors.loc[:,ff_factors.columns.str[1]=='L'].sum(axis=1))/16

        # CMA factors
        ff_factors['CMA'] = (ff_factors.loc[:,ff_factors.columns.str[3]=='C'].sum(axis=1)-
                             ff_factors.loc[:,ff_factors.columns.str[3]=='A'].sum(axis=1))/16

        # RMW factors
        ff_factors['RMW'] = (ff_factors.loc[:,ff_factors.columns.str[2]=='R'].sum(axis=1)-
                             ff_factors.loc[:,ff_factors.columns.str[2]=='W'].sum(axis=1))/16

        # XMY factors (new factor NF)
        ff_factors['XMY'] = (ff_factors.loc[:,ff_factors.columns.str[4]=='X'].sum(axis=1)-
                             ff_factors.loc[:,ff_factors.columns.str[4]=='Y'].sum(axis=1))/16

        # SMB factor
        ff_factors['SMB'] = (ff_factors.loc[:,ff_factors.columns.str[0]=='S'].sum(axis=1)-
                            ff_factors.loc[:,ff_factors.columns.str[0]=='B'].sum(axis=1))/16
        
        # integrate the factors table with the computed excess market risk factor
        mkt_rf=pd.merge(ccm3.groupby('jdate').apply(wavg, 'retadj','wt').to_frame().rename(columns={0: 'Mkt'}),
             ff_factors_original['RF'].to_frame(), how='right', right_index=True, left_index=True
            ).diff(axis=1, periods=-1)['Mkt'].to_frame().rename(columns={'Mkt':'Mkt-RF'}) #1st col - 2nd col

        ff_factors = pd.merge(ff_factors, mkt_rf, how='inner', right_index=True, left_index=True)
        ff_factors_final=ff_factors.loc[:,['Mkt-RF','SMB','HML','CMA','RMW','XMY']]

        print('#################################################################################')
        print("computing factors: {} dates, {} factors, from {} to {}".format(ff_factors_final.shape[0], ff_factors_final.shape[1], ff_factors_final.index.date.min(), ff_factors_final.index.date.max()))          
                

        #firms count
        ff_nfirms_final=ccm3.loc[:,['jdate']+ccm3.columns[ccm3.columns.str.contains('port')].to_list()].set_index('jdate').dropna(how='any', axis=0).groupby('jdate').count()
        ff_nfirms_final=pd.merge(ff_nfirms_final, ccm3.groupby('jdate')['retadj'].count().to_frame().rename(columns={'retadj':'Mkt-RF'}), how='outer', right_index=True, left_index=True)
        ff_nfirms_final.rename(columns={'szport':'SMB','bmport':'HML','invport':'CMA','nfport':'XMY','opport':'RMW',}, inplace=True)
        ff_nfirms_final=pd.merge(ff_nfirms_final, ff_factors_original.index.to_frame(), how='right', left_index=True, right_index=True).drop(columns='date').reset_index().rename(columns={'date':'jdate'}).set_index('jdate')
        ff_nfirms_final=ff_nfirms_final[['Mkt-RF','SMB','HML','CMA','RMW','XMY']].fillna(0).astype(int)

        ff_perchfirms_final=pd.merge(ff_nfirms_final, ccm3.groupby('jdate').size().to_frame(), how='inner', right_index=True, left_index=True)
        ff_perchfirms_final=ff_perchfirms_final.div(ff_perchfirms_final[0], axis=0).drop(columns=0).round(4)*100

        print('#########################################################################')
        print("average percentage of stocks available for each measure")
        print(ff_perchfirms_final.mean(axis=0))

            

        ff_comp=pd.merge(ff_factors_original, ff_factors_final, how='inner', right_index=True, left_index=True, suffixes=('_orig',''))

        print('#########################################################################')

        print("NEW FACTOR BASED ON: {}".format(current_measure))
        print("corr computed vs original Mkt-RF ", stats.pearsonr(ff_comp['Mkt-RF_orig'],ff_comp['Mkt-RF'])[0])
        print("corr computed (2x2x2x2x2) SMB vs original (2x3) SMB", stats.pearsonr(ff_comp['SMB_orig'], ff_comp['SMB'])[0])
        print("corr computed (2x2x2x2x2) SMB vs original (2x3) HML", stats.pearsonr(ff_comp['HML_orig'], ff_comp['HML'])[0])
        print("corr computed (2x2x2x2x2) SMB vs original (2x3) RMW", stats.pearsonr(ff_comp['RMW_orig'], ff_comp['RMW'])[0])
        print("corr computed (2x2x2x2x2) SMB vs original (2x3) CMA", stats.pearsonr(ff_comp['CMA_orig'], ff_comp['CMA'])[0])

        print("####### END COMPUTATION MEASURE: {}".format(current_measure))

        if print_advance_output:
            plt.figure(figsize=(15,20))
            plt.suptitle("Comparison of Calculated (2x2x2x2x2) FF5 vs Original (2x3) FF5 Factors", fontsize=20)

            ax1 = plt.subplot(611)
            ax1.set_title('SMB', fontsize=13)
            ax1.set_xlim([ff_comp.index.date.min(), ff_comp.index.date.max()])
            ax1.plot(ff_comp['SMB_orig'], 'r--', ff_comp['SMB'], 'b-')
            ax1.legend(('SMB_orig','SMB'), loc='upper right', shadow=True)

            ax2 = plt.subplot(612)
            ax2.set_title('HML', fontsize=13)
            ax2.set_xlim([ff_comp.index.date.min(), ff_comp.index.date.max()])
            ax2.plot(ff_comp['HML_orig'], 'r--', ff_comp['HML'], 'b-')
            ax2.legend(('HML_orig','HML'), loc='upper right', shadow=True)

            ax3 = plt.subplot(613)
            ax3.set_title('RMW', fontsize=13)
            ax3.set_xlim([ff_comp.index.date.min(), ff_comp.index.date.max()])
            ax3.plot(ff_comp['RMW_orig'], 'r--', ff_comp['RMW'], 'b-')
            ax3.legend(('RMW_orig','RMW'), loc='upper right', shadow=True)

            ax4 = plt.subplot(614)
            ax4.set_title('CMA', fontsize=13)
            ax4.set_xlim([ff_comp.index.date.min(), ff_comp.index.date.max()])
            ax4.plot(ff_comp['CMA_orig'], 'r--', ff_comp['CMA'], 'b-')
            ax4.legend(('CMA_orig','CMA'), loc='upper right', shadow=True)

            ax5 = plt.subplot(615)
            ax5.set_title('    Mkt-RF', fontsize=13)
            ax5.set_xlim([ff_comp.index.date.min(), ff_comp.index.date.max()])
            ax5.plot(ff_comp['Mkt-RF_orig'], 'r--', ff_comp['Mkt-RF'], 'b-')
            ax5.legend(('Mkt-RF_orig','Mkt-RF'), loc='upper right', shadow=True)

            ax6 = plt.subplot(616)
            ax6.set_title('    New Factor: '+current_measure, fontsize=13)
            ax6.set_xlim([ff_comp.index.date.min(), ff_comp.index.date.max()])
            ax6.plot(ff_comp['XMY'], 'b-')

            plt.subplots_adjust(top=0.95, hspace=0.2)

            plt.show()    

        #appends for export to a file 
        ff_factors_final['measures_robust_bp']=""
        ff_factors_final['bp']=""
        ff_factors_final['measure']=current_measure
        ff_factors_final2export=pd.concat([ff_factors_final2export, ff_factors_final], axis=0) 

        #append for export to a file 
        ff_nfirms_final['measures_robust_bp']=""
        ff_nfirms_final['bp']=""
        ff_nfirms_final['measure']=current_measure
        ff_nfirms_final2export=pd.concat([ff_nfirms_final2export, ff_nfirms_final], axis=0)            

            
    #write tables to export, to files
    ff_nfirms_final2export.to_csv(factors_path+f.split('\\')[-1].split('.')[0].replace('measures_','2x2x2x2x2_factors_nfirms_')+'.csv', index = True, float_format='%.6f')
    ff_factors_final2export.to_csv(factors_path+f.split('\\')[-1].split('.')[0].replace('measures_','2x2x2x2x2_factors_')+'.csv', index = True)
    


# In[ ]:


get_ipython().system('jupyter nbconvert --to script factors_builders_FF5_2x2x2x2x2_augment_by_candidate_factor_working.ipynb')

