This script allows the user to replicate the test portfolios construction methodology based on bivariate sorts described in Fama and French (2015) (used to test their 3 and 5 factors models) and to extend it including a new measure (hereinafter referred to as NF, which stands for New Factor) to be used in the sorts (together with Size, Book-to-Market, Operating Profitability and Investments). The measure is an arbitrary input, the user can pass either one or multiple files the procedure, each file contain either one or more measure, organized in columns.

This script builds the following portfolios following Fama and Frech (2015):

- 4 5x5 bivariate sorts portfolios of excess returns with independent sort
    - SIZExINV
    - SIZExOP
    - SIZExBM
    - SIZExNF
    
- 6 2x4X4 three-way sorts portfolios of excess returns with size dependent sort 
    - SIZExBMxINV
    - SIZExBMxOP
    - SIZExBMxNF
    - SIZExINVxOP
    - SIZExINVxNF
    - SIZExOPxNF    
    
However, some more bivariate sorts portfolios are built, besides the Fama and French (2015), the total set consists of:
- SIZExBM
- SIZExINV
- SIZExOP
- SIZExNF
- BMxINV
- BMxOP
- BMxNF
- INVxOP
- INVxNF
- OPxNF
- SIZExBMxINV
- SIZExBMxOP
- SIZExBMxNF
- SIZExINVxOP
- SIZExINVxNF
- SIZExOPxNF


The methodology follows Fama and French (2015):
- 5x5 bivariate sorts:
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_op_inv.html
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_beme_inv.html
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_beme_op.html
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_me_inv.html
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_me_op.html
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports.html

- 2x4x4 three-way sorts:
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/32_ports_me_op_inv.html
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/32_ports_me_beme_inv.html
    - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/32_ports_me_beme_op.html


Research notes:

- only ordinary common stocks (CRSP sharecode 10 and 11) in NYSE, AMEX and NASDAQ (exchange code 1,2,3) and at least 2 years on Compustat are included in the sample (Fama and French (1993, 2015); https://wrds-www.wharton.upenn.edu/pages/support/applications/risk-factors-and-industry-benchmarks/fama-french-factors/).
- all the breakpoints are computed only on NYSE stocks (from the sample).
- market cap is calculated at issue-level (permno in CRSP), and book value of equity is calculated at company level (permco in Compustat), it is needed to aggregate market cap at company level (permco in CRSP) for later book-to-market value calculation. And market cap of companies at December of year t-1 is used for portfolio formation at June of year t. Details on how to link CRSP and Compustat:<br>
    https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/crspcompustat-merged-ccm/wrds-overview-crspcompustat-merged-ccm/<br>https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/
- there were cases when the same firm (CRSP permco) had two or more securities (CRSP permno) on the same date. For the purpose of ME for the firm, I aggregated all ME for a given CRSP permco, date. This aggregated ME was assigned to the CRSP permno according to the following criteria largest market equity (ME), higher number of years on Compustat (count) (as recommended by WRDS (https://wrds-www.wharton.upenn.edu/pages/support/applications/risk-factors-and-industry-benchmarks/fama-french-factors/) and finally random, in this order. If the ME and years on Compustat are the same there is no other unbiased criteria but random (one would select the one with either largest or smallest return). However these cases are less than 100. The ME to assign to the permco is the sum of the ME of all the permno of that permco.
- the relevant share code for Fama French factors constructions are 10 and 11 (ordinary common stocks). The permno for the same permco may have different share code (shrcd), filtering them before applying the logic o the previous point would end up in loosing market capitalization. The solution is to delete later, when each permco has only one permno, all the permno with shrcd different from 10 or 11.
- I merged CRSP and Compustat using the CRSP CCM product (as of April 2010) as recommended by WRDS (https://wrds-www.wharton.upenn.edu/pages/support/applications/risk-factors-and-industry-benchmarks/fama-french-factors/) matching Compustat's gvkey (from calendar year t-1) to CRSP's permno as of June year t. Data was cleaned for unnecessary duplicates. First there were cases when different gvkeys exist for same permno-date. I solved these duplicates by only keeping those cases that are flagged as 'primary' matches by CRSP's CCM (linkprim='P'). There were other unnecessary duplicates that were removed (I kept the oldest gvkey for each permno, finally I randomly picked one gvkey for each of of the about 30 pairs od dupliated permno which were practically identical if not for fractions of decimals differences on certain measures). Some companies on Compustat may have two annual accounting records in the same calendar year. This is produced by change in the fiscal year end during the same calendar year. In these cases, we selected the last annual record for a given calendar year.


Variable definitions (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html)

- ME: Market Equity. Market equity (size) is price times shares outstanding. Price is from CRSP, shares outstanding are from Compustat (if available) or CRSP.

- BE: Equity. Book equity is constructed from Compustat data or collected from the Moody’s Industrial, Financial, and Utilities manuals. BE is the book value of stockholders’ equity, plus balance sheet deferred taxes and investment tax credit (if available), minus the book value of preferred stock. Depending on availability, we use the redemption, liquidation, or par value (in that order) to estimate the book value of preferred stock. Stockholders’ equity is the value reported by Moody’s or Compustat, if it is available. If not, we measure stockholders’ equity as the book value of common equity plus the par value of preferred stock, or the book value of assets minus total liabilities (in that order). See Davis, Fama, and French, 2000, “Characteristics, Covariances, and Average Returns: 1929-1997,” Journal of Finance, for more details.

- BE/ME: Book-to-Market. The book-to-market ratio used to form portfolios in June of year t is book equity for the fiscal year ending in calendar year t-1, divided by market equity at the end of December of t-1.
 
- OP: Operating Profitability. The operating profitability ratio used to form portfolios in June of year t is annual revenues minus cost of goods sold, interest expense, and selling, general, and administrative expense divided by the sum of book equity and minority interest for the last fiscal year ending in t-1.
 
- INV: Investment. The investment ratio used to form portfolios in June of year t is the change in total assets from the fiscal year ending in year t-2 to the fiscal year ending in t-1, divided by t-2 total assets.


Techincal notes:

- In order to tun the script one has to connect ot the WRDS databases and have a valid WRDS account. Here are the details on how to set up a connection or run the scrip on the WRDS cloud.<br>https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-wrds-cloud/<br>https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/
- WRDS Python library documentation
https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/querying-wrds-data-python/


User guide:

- Basic
    - the user has to place one or more measures file (.csv) in the path assigned to the variable nf_measures_path
    - each measure file can contain multiple measures on the columns (with headers)
    - each measure file contains on the rows pairs of "jdate" (dates of last day of June in format YYYY-MM-DD) and "permno" (containing Compustat PERMNO). The first two columns in the csv file must be called "permno and "date"
    - for each measure file passed, the corresponding factors (and additional file for firms count) are saved in a csv file with a similar name, the rows are oranized for date, measure for which robust breakpoints have been choosed, the breakpoints percentailes, measure name

- Advanced
    - in the Fama French 5 factors procedure the factors are constructed using as breakpoints 30th and 70th percentiles for B/M, OP, and INV
    - here the user can specify in the list variable measures_robust_check_bp all the factors for which he or she wants to use alternative breakpoint percentiles (for instance measures_robust_check_bp=['bm', 'inv', 'op', 'nf'] if alternative breakpoints want ot be used for all the factors)
    - the alernative breakpoints are fixed to be 10th and 90th, 20th and 80th, 30th and 70th, 40th and 60th.
    
Descriptions of Fama French 5 factors (2x3) can be found on Kenneth French's website.<br>http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html <br>https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html    


References:
- Fama, Eugene F. and Kenneth R. French, 1993, Common Risk Factors in Stocks and Bonds, Journal of Financial Economics, 33, 3-56.
- Fama, E.F. and French, K.R., 2015. A five-factor asset pricing model. Journal of financial economics, 116(1), pp.1-22.
