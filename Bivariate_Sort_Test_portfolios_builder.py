#!/usr/bin/env python
# coding: utf-8

# # Notes

# **To DO**
# 

# This script allows the user to replicate the test portfolios construction methodology based on bivariate sorts described in Fama and French (2015) (used to test their 3 and 5 factors models) and to extend it including a new measure (hereinafter referred to as NF, which stands for New Factor) to be used in the sorts (together with Size, Book-to-Market, Operating Profitability and Investments). The measure is an arbitrary input, the user can pass either one or multiple files the procedure, each file contain either one or more measure, organized in columns.
# 
# This script builds the following portfolios following Fama and Frech (2015):
# 
# - 4 5x5 bivariate sorts portfolios of excess returns with independent sort
#     - SIZExINV
#     - SIZExOP
#     - SIZExBM
#     - SIZExNF
#     
# However, some more bivariate sorts portfolios are built, besides the Fama and French (2015), the total set consists of:
# - SIZExBM
# - SIZExINV
# - SIZExOP
# - SIZExNF
# - BMxINV
# - BMxOP
# - BMxNF
# - INVxOP
# - INVxNF
# - OPxNF
# 
# The methodology follows Fama and French (2015):
# - 5x5 bivariate sorts:
#     - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_op_inv.html
#     - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_beme_inv.html
#     - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_beme_op.html
#     - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_me_inv.html
#     - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_me_op.html
#     - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports.html
# 

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

# Descriptions of Fama French 5 factors (2x3) can be found on Kenneth French's website.<br>http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html <br>https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html

# ![image.png](attachment:image.png)

# References:
# - Fama, Eugene F. and Kenneth R. French, 1993, Common Risk Factors in Stocks and Bonds, Journal of Financial Economics, 33, 3-56.
# - Fama, E.F. and French, K.R., 2015. A five-factor asset pricing model. Journal of financial economics, 116(1), pp.1-22.

# # Script

# In[1]:


# list of all measures for which a factor will be computed
all_measures=['bm','op','inv','nf']
# measures for which all the breakpoints (10_90, 20_80, 30_70, 40_60) subsets are used instead of the default 30_70
# measures_robust_check_bp=['bm', 'inv', 'op', 'nf']
measures_robust_check_bp=[]


# In[2]:


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
from itertools import combinations

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

#set to 1 to delete intermediate datasets
save_memory=0

#set to 1 to print more check outputs
print_advance_output=1


# ## Functions

# In[3]:


# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


# In[4]:


def read_ff_test_assets(url):
    zip_name=url.split('/')[-1]
    file_name=zip_name.split('.')[0].replace('_CSV','.csv')
    urllib.request.urlretrieve(url,zip_name)
    zip_file = zipfile.ZipFile(zip_name, 'r')
    zip_file.extractall()
    zip_file.close()
    with open(file_name) as f:
        lines=f.readlines()
        num = [i for i, l in enumerate(lines) if l=='  Average Value Weighted Returns -- Monthly\n']
    rows_to_skip=num[0]+1
    ff_ptfs=pd.read_csv(file_name, skiprows=rows_to_skip, encoding='cp1252').rename(columns={'Unnamed: 0':'date'})
    ff_ptfs=ff_ptfs.iloc[:ff_ptfs.loc[:,'date'].isnull().idxmax()-1]
    ff_ptfs['date']=pd.to_datetime(ff_ptfs['date'], format='%Y%m')
    ff_ptfs.set_index('date', inplace=True)
    # ff_ptfs=ff_ptfs[(ff_ptfs.index >= pd.to_datetime(test_startdate, format='%Y%m').to_period('m')) & (ff_ptfs.index <= pd.to_datetime(test_enddate, format='%Y%m').to_period('m'))]
    ff_ptfs=ff_ptfs.astype(float).div(100)
    ff_ptfs.index = pd.to_datetime(ff_ptfs.index, format='%Y%m')+MonthEnd(0)
    ff_ptfs=ff_ptfs[ff_ptfs.index<=ff_factors_original.index.max()]
    ff_ptfs=ff_ptfs[ff_ptfs.index>=ff_factors_original.index.min()]
    return ff_ptfs


# ## Computation

# In[5]:


bivariate_sorts=[("_".join(map(str, comb))) for comb in combinations(['size']+all_measures, 2)]
  
# Print the obtained permutations 
print('bivariate sorts portfolios built on:\n- {}'.format('\n- '.join(bivariate_sorts))+'\n')


# In[6]:


project_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\"
factors_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\factors\\"
test_assets_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\test_assets\\bivariate_sort\\"
ff5measures_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\FF5 measures\\"
nf_measures_path="G:\\My Drive\\PhD\\Research\\Indipendent project\\estimated_data\\measures\\"


# Read the CRSP table with stock returns produced by FF5_measures_computer_and_FF5_factors_replicator\FF5_measures_computer_and_FF5_factors_replicator.ipynb

# In[7]:


crsp3=pd.read_csv(ff5measures_path+'crsp_returns.csv', sep=',')
crsp3['jdate']=pd.to_datetime(crsp3['jdate'])
crsp3


# Read the original Fama and French factors

# In[8]:


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


# Reading the original Fama and French Bivariate sort portfolios

# In[9]:


FamaFrench_test_assets_repository={
'size_bm':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_5x5_CSV.zip',
'size_inv':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_ME_INV_5x5_CSV.zip',
'size_op':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_ME_OP_5x5_CSV.zip',
'bm_inv':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_BEME_INV_5x5_CSV.zip',
'bm_op':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_BEME_OP_5x5_CSV.zip',
'op_inv':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_OP_INV_5x5_CSV.zip'
                                    }


# In[10]:


if "assets_downloaded" not in locals():
    ff_test_assets=pd.DataFrame()
    for key in FamaFrench_test_assets_repository:
        print(key, '->', FamaFrench_test_assets_repository[key])
        ptfs_i=read_ff_test_assets(FamaFrench_test_assets_repository[key])
        print('downloaded {}: {} dates {} portfolios'.format(key, ptfs_i.shape[0],ptfs_i.shape[1]))
        ptfs_i=ptfs_i.reset_index().melt(id_vars='date', var_name='ptf_code', value_name='ret')
        ptfs_i['sort_on']=key
        ff_test_assets=pd.concat([ff_test_assets,ptfs_i], axis=0)
        assets_downloaded=[]
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('SMALL','ME1')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('BIG','ME5')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('LoBM','BM1')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('HiBM','BM5')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('LoINV','INV1')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('HiINV','INV5')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('LoOP','OP1')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('HiOP','OP5')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace(' ','_')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('ME','size')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('BM','bm')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('INV','inv')
    ff_test_assets['ptf_code']=ff_test_assets['ptf_code'].str.replace('OP','op')    
    ff_test_assets.set_index(['date','ptf_code'], inplace=True)
ff_test_assets    


# Reading FF5 measures

# In[11]:


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

# In[12]:


ccm_jun_original=ccm_jun.drop(columns=ccm_jun.columns[ccm_jun.columns.str.contains('_ok')], axis=0).copy() #dropped the is_.._ok columns
crsp3_original=crsp3.copy()


# Reading the new factor measures files

# In[13]:


nf_measures_files=glob.glob(nf_measures_path+"*.csv")
nf_measures_files


# In[ ]:


print('##################################################################################################################')
print('##################################################################################################################')
print('### START LOOP ON MEASURES FILES')
print('##################################################################################################################')
print('##################################################################################################################')

# for f in tqdm([nf_measures_files[15]], desc='measures_files_loop'): #DEBUG: SLICING LIST OF MEASURES FILES
for f in tqdm(nf_measures_files, desc='measures_files_loop'): 

    ptfs2export=pd.DataFrame()
    ptfs_nfirms2export=pd.DataFrame()
    
    #read the file
    print('#####################################################################################')
    print("reading file {}".format(f.split('\\')[-1]))
    pd.read_csv(f)
    nf_measures=pd.read_csv(f)
    nf_measures.rename(columns={'date':'jdate'}, inplace=True)
    nf_measures['jdate']=pd.to_datetime(nf_measures['jdate'], format="%Y%m")+MonthEnd(0)    
    nf_measures=pd.merge(ccm_jun_original.loc[:,['permno','jdate']], nf_measures, how='inner', on=['permno','jdate'])
    nf_measures.set_index(['jdate','permno'], inplace=True)
    
    measures_legend_f=nf_measures.columns.to_frame().reset_index().drop(columns=0).rename(columns={'index':'measure'})
    measures_legend_f['index']=measures_legend_f.index+1

    print('#####################################################################################')
    print('min, max and mean number of stocks with good measure, over periods, for each measure')    
    print(nf_measures.notna().groupby('jdate').sum().agg(['min', 'max', 'mean']).T.astype(int))

    #filter the pairs of stocks and june dates not in common betweeen FF5 measures and New Factor measures
    ccm_jun=pd.merge(ccm_jun_original, nf_measures, how='inner', left_on=['jdate','permno'], right_index=True).loc[:,ccm_jun_original.columns.to_list()]
    print('#####################################################################################')
    print("{} pairs of stocks and june dates in common betweeen FF5 and New Factor".format(ccm_jun.shape[0]))

    # at this point shrcd different from 10 and 11, not valid price and shares and company with less than 2 years of
    # history on Compustat have been removed, even though their conditions are explicitly present here

    # Adding flag to the stocks with good FF5 measures
    ccm_jun.loc[:,'is_size_ok']=np.where((ccm_jun['me']>0)&
                                          ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))
                                          ,1,0)

    ccm_jun.loc[:,'is_beme_ok']=np.where((ccm_jun['be']>0)&
                                          (ccm_jun['dec_me'].notna())&
                                          ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))&
                                          (ccm_jun['me']>0),1,0)

    ccm_jun.loc[:,'is_inv_ok']=np.where((ccm_jun['inv'].notna())&
                                         ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))&
                                         (ccm_jun['me']>0),1,0)

    ccm_jun.loc[:,'is_op_ok']=np.where(((ccm_jun['be'])>0)&(ccm_jun['revt'].notna())&
                                        (ccm_jun.loc[:,['xsga','xint','cogs']].notna().any(axis=1))&
                                        ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))&
                                        (ccm_jun['me']>0),1,0)

    #assign portfolios for FF5 measures based on independent univariate sorting

    # assign size portfolios 
    size_ptf=(ccm_jun.set_index(['permno']).groupby('jdate')
                                          .apply(lambda x:
                                                          pd.cut(x['me'], [-float("inf")]+
                                                                          x[(x['exchcd']==1)&
                                                                          ((x['shrcd']==10) | (x['shrcd']==11))&
                                                                          (~x['prc'].isna())&
                                                                          (~x['shrout'].isna())]['me']
                                                                          .quantile([0.2, 0.4, 0.6, 0.8])
                                                                          .reset_index()['me'].to_list()
                                                                           +[float("inf")],
                                                                 labels=['size1','size2','size3','size4','size5'],
                                                                 right=True, include_lowest=True
                                                                )
                                                ).to_frame()).rename(columns={'me':'sizeport'})
    
    # assign book-to-market portfolios 
    beme_ptf=(ccm_jun.set_index(['permno']).groupby('jdate')
                                          .apply(lambda x:
                                                          pd.cut(x['beme'], [-float("inf")]+
                                                                            x[(x['exchcd']==1)&
                                                                            ((x['shrcd']==10) | (x['shrcd']==11))&
                                                                            (~x['dec_me'].isna())&
                                                                            (x['be']>0)]['beme']
                                                                            .quantile([0.2, 0.4, 0.6, 0.8])
                                                                            .reset_index()['beme'].to_list()
                                                                            +[float("inf")],
                                                                 labels=['bm1','bm2','bm3','bm4','bm5'],
                                                                 right=True, include_lowest=True
                                                                )
                                                ).to_frame()).rename(columns={'beme':'bmport'})
    
    # assign operating profitability portfolios
    op_ptf=(ccm_jun.set_index(['permno']).groupby('jdate')
                                          .apply(lambda x:
                                                          pd.cut(x['op'], [-float("inf")]+
                                                                          x[(x['exchcd']==1)&
                                                                          ((x['shrcd']==10) | (x['shrcd']==11))&  
                                                                          (x['be']>0)&                               
                                                                          (~x['revt'].isna())&
                                                                          (~x.loc[:,['cogs','xsga','xint']].isna()
                                                                          .all(axis=1))]['op']
                                                                          .quantile([0.2, 0.4, 0.6, 0.8])
                                                                          .reset_index()['op'].to_list()
                                                                          +[float("inf")],
                                                                 labels=['op1','op2','op3','op4','op5'],
                                                                 right=True, include_lowest=True
                                                                )
                                                ).to_frame()).rename(columns={'op':'opport'})
 
    
    # assign investments portfolios 
    inv_ptf=(ccm_jun.set_index(['permno']).groupby('jdate')
                                          .apply(lambda x:
                                                          pd.cut(x['inv'],[-float("inf")]+
                                                                          x[(x['exchcd']==1)&
                                                                          ((x['shrcd']==10) | (x['shrcd']==11))&                 
                                                                          (~x['inv'].isna())]['inv']
                                                                          .quantile([0.2, 0.4, 0.6, 0.8])
                                                                          .reset_index()['inv'].to_list()
                                                                          +[float("inf")],
                                                                 labels=['inv1','inv2','inv3','inv4','inv5'],
                                                                 right=True, include_lowest=True
                                                                )
                                                ).to_frame()).rename(columns={'inv':'invport'})

    #join back size, beme and inv breakdown
    portfolios = pd.merge(inv_ptf,pd.merge(pd.merge(size_ptf, beme_ptf, how='outer', right_index=True, left_index=True), op_ptf, how='outer', right_index=True, left_index=True), how='outer', right_index=True, left_index=True)

    # allign breakpoints to variables
    ccm1_jun = pd.merge(ccm_jun, portfolios, how='left', left_on=['jdate','permno'], right_index=True)
    ccm1_jun['ffyear']=ccm1_jun['jdate'].dt.year
    ccm1_jun

    print('##################################################################################################################')
    print('### START LOOP ON MEASURES OF FILE: {}'.format(f.split('\\')[-1]))
    print('##################################################################################################################')

#     nf_measures=nf_measures.iloc[:,13:]#.to_frame() #DEBUG: UNCOMMENT
    for current_measure, m in tqdm(nf_measures.iteritems(), total=nf_measures.shape[1], desc='measures_loop'):  
        print('#########################################################################')
        print("current measure: {}".format(current_measure))
        print('#########################################################################')
        nf_measure_i=m.to_frame().reset_index().rename(columns={current_measure:'nf'}) 
        
#         nf_measure_not_good=0
        
        #initialize the dataframe where all the ptfs for the current measure will be saved
        ptfs_m=pd.DataFrame()
        ptfs_nfirms_m=pd.DataFrame()
        
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

        #Statistics about stock with available measures

        stocks_good_measures_n=(ccm1_jun.loc[:,['jdate']+ccm1_jun.columns[ccm1_jun.columns.str.contains('_ok')].to_list()].groupby('jdate').sum())
        stocks_good_measures_perc=(ccm1_jun.loc[:,['jdate']+ccm1_jun.columns[ccm1_jun.columns.str.contains('_ok')].to_list()].groupby('jdate').sum()).div(ccm1_jun.groupby('jdate').size(), axis=0).round(4)*100
        print("#########################################################################\nstocks available for each measures (number)")
        print(stocks_good_measures_n)
        print("#########################################################################")        
        

        #check if for each date there is at least one stock with nf measure available
#         if ccm1_jun.groupby('jdate')['is_nf_ok'].sum().min()==0:
#             print('!!!!!!!!!!!!there are dates with not even one stock with nf measure available!!!!!!!!!!!!')
#             nf_measure_not_good=1 #when set to 1, construction of portfolios based on sort on nf measures, are skipped
#         else:

        # assign new factor portfolios
        epsilon=0.00000000001 #for many measure the bins are not monotonically increasing, therefore a decreasing value (proportional to epsilon) is subracted to each bin edge to make them monotonically increasing
        dates_with_at_least_2_nf=ccm1_jun[(ccm1_jun['exchcd']==1)&((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))&(~ccm1_jun['nf'].isna())].groupby('jdate')['nf'].nunique().to_frame()
        dates_with_at_least_2_nf=dates_with_at_least_2_nf[dates_with_at_least_2_nf['nf']>=2].reset_index()['jdate']
        ccm1_jun_with_at_least_2_valid_nf_for_each_per_period=pd.merge(ccm1_jun, dates_with_at_least_2_nf.to_frame(), how='inner', on='jdate').reset_index().dropna(subset=['nf'])
        nf_ptf=(ccm1_jun_with_at_least_2_valid_nf_for_each_per_period.dropna(subset=['nf']).set_index(['permno']).groupby('jdate')
                                              .apply(lambda x: pd.cut(x['nf'], ([-float("inf")]+
                                                                                [l-epsilon*(4-ind-1) for ind, l in 
                                                                                enumerate(
                                                                                x[(x['exchcd']==1)&
                                                                                ((x['shrcd']==10) | (x['shrcd']==11))&                                           
                                                                                (~x['nf'].isna())]['nf']
                                                                                .quantile([0.2, 0.4, 0.6, 0.8])
                                                                                .reset_index()['nf'].to_list() 
                                                                                )]
                                                                                +[float("inf")]),
                                                                     labels=['nf1','nf2','nf3','nf4','nf5'],
                                                                     right=True, include_lowest=True
                                                                    )
                                                    ).to_frame()).reset_index().rename(columns={'nf':'nfport'}) 
        ccm1_jun = pd.merge(ccm1_jun, nf_ptf, how='left', on=['jdate','permno'])        
        

            #check/debug querys
    #         nf_bp=(ccm1_jun[(ccm1_jun['exchcd']==1)&
    #         ((ccm1_jun['shrcd']==10) | (ccm1_jun['shrcd']==11))&                                           
    #         (~ccm1_jun['nf'].isna())].groupby('jdate')['nf']
    #         .quantile([0, 0.2, 0.4, 0.6, 0.8, 1])).to_frame().reset_index().rename(columns={'level_1':'nf_bp'})
    #         print(pd.pivot(nf_bp, index='jdate', columns='nf_bp', values='nf').add_prefix('nf_'))

        #check whether accidentally some not ok permno received a portfolio
        if (ccm1_jun[ccm1_jun['is_size_ok']==0]['sizeport'].nunique()+
         ccm1_jun[ccm1_jun['is_beme_ok']==0]['bmport'].nunique()+
         ccm1_jun[ccm1_jun['is_op_ok']==0]['opport'].nunique()+
         ccm1_jun[ccm1_jun['is_inv_ok']==0]['invport'].nunique()) !=0:
            print('!!!!!!!!!!!!!!!!!!accidentally some not ok permno received a portfolio!!!!!!!!!!!!!!!!!!')

        # store portfolio assignment as of June
        june=ccm1_jun.loc[:,['permno','ffyear']+ccm1_jun.columns[ccm1_jun.columns.str.contains('port')].to_list()]
        if save_memory:
            del ccm1_jun

        # merge back with monthly records
        crsp3 = crsp3_original[['date','permno','retadj','wt','ffyear','jdate']]
        ccm3=pd.merge(crsp3,june, how='left', on=['permno','ffyear'])
        if save_memory:
            del june
            del crsp3
        ccm3.dropna(axis=0, how='all', subset=ccm3.columns[ccm3.columns.str.contains('port')], inplace=True)

        ################################
        ## Bivariate sorts portfolios ##
        ################################
        print('################################\n## Bivariate sorts portfolios ##\n################################')
    
        #loop on the pair of measures for the bivariate sort
        for sort_on in bivariate_sorts:
            
#             if ('nf' in sort_on)&(nf_measure_not_good==1):
#                     print('#################################################################################\n!!!!!!!!!!!Skip construction double sort portfolios based on {} !!!!!!!!!!!'.format(', '.join(sort_on.split('_'))))
#                     continue
                    
            print('#################################################################################\nConstructing double sort portfolios based on {}'.format(', '.join(sort_on.split('_'))))
            sort_on_cols=[m+'port' for m in sort_on.split('_')]

            # select the columns that contains portfolios
            ccm4=ccm3[['jdate','retadj','wt']+sort_on_cols]

            ### calculate the portfolio returns
            ccm4['retadj_X_wt']=ccm4['retadj']*ccm4['wt']
            vwret=ccm4.groupby(['jdate']+ccm4.columns[ccm4.columns.str.contains('port')].to_list())['wt','retadj_X_wt'].sum().reset_index()
            vwret['vwret']=vwret['retadj_X_wt']/vwret['wt']
            vwret[sort_on_cols]=vwret[sort_on_cols].astype(str)
            vwret['ptf_code']=vwret[sort_on_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

            print("unique values of number of portfolios for each date: {}".format(vwret.groupby('jdate')['ptf_code'].nunique().unique()))
            

            # transpose and add missing date (right join with ff_factors_original)
            ptfs=vwret.pivot(index='jdate', columns='ptf_code', values='vwret').reset_index().rename(columns={'jdate':'date'}).sort_values('date').set_index('date').fillna(0)
            ptfs=pd.merge(ff_factors_original, ptfs, how='left', left_index=True, right_index=True).drop(columns=ff_factors_original.columns.to_list())

            if ptfs.shape[1]!=25:
                print("!!!!!!!!!!!!!!!!!!!!building block portfolio missing!!!!!!!!!!!!!!!!!!!!")
                continue #DEBUG: COMMENT

            # integrate the factors table with the risk-free
            ptfs=pd.merge(ff_factors_original['RF'].to_frame(), ptfs, how='left', right_index=True, left_index=True)
            ptfs=ptfs.sub(ff_factors_original['RF'], axis=0).drop(columns='RF')
           
            print("computing portfoliost: {} dates, {} portfolios, from {} to {}".format(ptfs.shape[0], ptfs.shape[1], ptfs.index.date.min(), ptfs.index.date.max()))          

            #firms count
            ptfs_nfirms=ccm4.groupby(['jdate']+ccm4.columns[ccm4.columns.str.contains('port')].to_list())['retadj_X_wt'].count().reset_index()
            ptfs_nfirms[sort_on_cols]=ptfs_nfirms[sort_on_cols].astype(str)
            ptfs_nfirms['ptf_code']=ptfs_nfirms[sort_on_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            ptfs_nfirms=ptfs_nfirms.pivot(index='jdate', columns='ptf_code', values='retadj_X_wt').reset_index().rename(columns={'jdate':'date'}).sort_values('date').set_index('date').fillna(0)
            ptfs_nfirms=pd.merge(ff_factors_original.index.to_frame(),ptfs_nfirms , how='right', left_index=True, right_index=True).drop(columns='date')

            ptfs_nfirms_perc=pd.merge(ptfs_nfirms, ccm3.groupby('jdate').size().to_frame(), how='inner', right_index=True, left_index=True)
            ptfs_nfirms_perc=np.sum(ptfs_nfirms_perc.div(ptfs_nfirms_perc[0], axis=0).drop(columns=0).round(4)*100, axis=1)

            print("average percentage of stocks available for each measure: {}".format(ptfs_nfirms_perc.mean(axis=0)))

            #appends to ptfs for the current measure
            ptfs=ptfs.reset_index().melt(id_vars='date', var_name='ptf_code', value_name='ret').set_index(['date','ptf_code'])
            ptfs['sort_on']=sort_on
            ptfs['measure']=measures_legend_f.loc[measures_legend_f['measure']==current_measure,'index'].values[0]
            ptfs_m=pd.concat([ptfs_m,ptfs], axis=0)

            #append to ptfs nfirms for the current measure
            ptfs_nfirms=ptfs_nfirms.reset_index().melt(id_vars='date', var_name='ptf_code', value_name='nfirms').set_index(['date','ptf_code'])
            ptfs_nfirms['sort_on']=sort_on
            ptfs_nfirms['measure']=measures_legend_f.loc[measures_legend_f['measure']==current_measure,'index'].values[0]
            ptfs_nfirms_m=pd.concat([ptfs_nfirms_m,ptfs_nfirms], axis=0)

        #statistcs for the ptfs of the current measure
        ptfs_comp=pd.merge(ptfs_m.drop(columns=['measure']), ff_test_assets, how='right', on=['date','ptf_code'])
        print('####################################################################\n####################################################################\ncorrelation between original Fama French test assets and replicated:')
        print(ptfs_comp.groupby('sort_on_x')['ret_x','ret_y'].apply(lambda x: x.corr().iloc[0,1]))

        #append the current measure ptfs to the current file ptfs
        ptfs2export=pd.concat([ptfs2export, ptfs_m], axis=0) 
        ptfs_nfirms2export=pd.concat([ptfs_nfirms2export, ptfs_nfirms], axis=0) 

    #write tables to export, to files
    ptfs_nfirms2export.to_csv(test_assets_path+f.split('\\')[-1].split('.')[0].replace('measures_','bivariate_sorts_test_portfolios_nfirms_')+'.csv', index = True, float_format='%.6f')
    ptfs2export.to_csv(test_assets_path+f.split('\\')[-1].split('.')[0].replace('measures_','bivariate_sorts_test_portfolios_')+'.csv', index = True, float_format='%.6f')
    measures_legend_f.to_csv(test_assets_path+f.split('\\')[-1].split('.')[0].replace('measures_','measures_legend_bivariate_sorts_test_portfolios_')+'.csv', index = False, float_format='%.6f')
    


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Bivariate_Sort_Test_portfolios_builder.ipynb')

