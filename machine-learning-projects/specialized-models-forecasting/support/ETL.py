#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
import os

#thrid part
from currency_converter import CurrencyConverter
import numpy as np
import pandas as pd
import re

def ETL(filepath
        , rates_filepath = 'data/rates.csv', re_generate_rates=False
        , to_replace_company_names = None, value_company_names = None):
    '''
    Runs ETL
    '''
    # list files
    files_csv = [f for f in os.listdir(filepath) if '.DS_Store' not in f]
    
    # load files
    files_list = []
    df = None
    try:
        for f in files_csv:
            print(f)
            if df is None:
                df = pd.read_csv(filepath + f)
                df['Store'] = f[14:16]
            else:
                df_temp = pd.read_csv(filepath + f)
                df_temp['Store'] = f[14:16]
                df = pd.concat([df,df_temp],axis=0)
    except Exception as e:
        print(e)
    
    df.reset_index(drop=True,inplace=True)
    
    # Name can be the same across stores
    df['OrderId'] = df['Name'] + '-' + df['Store']
    
    #drop cancelled and refunded orders
    filt_c = df.loc[df['Cancelled at'].notna(),'OrderId']
    filt_r = df.loc[df['Financial Status'] == 'refunded','OrderId']
    df.drop(df.loc[
        (df['OrderId'].isin(filt_c)) | (df['OrderId'].isin(filt_r))
        ].index,axis=0,inplace=True)

    df.reset_index(drop=True,inplace=True)
    
    # Remove useless columns
    df.drop([
        'Email','Accepts Marketing','Lineitem compare at price'
        ,'Lineitem requires shipping','Lineitem taxable','Billing Phone'
        ,'Shipping Name','Shipping Street','Shipping Address1','Shipping Address2'
        ,'Shipping Company','Shipping City','Shipping Zip','Shipping Province'
        ,'Shipping Country','Shipping Phone','Notes','Note Attributes'
        ,'Payment Reference','Vendor','Outstanding Balance','Employee','Location'
        ,'Device ID','Id','Risk Level','Source','Tax 1 Name','Tax 1 Value'
        ,'Tax 2 Name','Tax 2 Value','Tax 3 Name','Tax 3 Value','Tax 4 Name','Tax 4 Value'
        ,'Tax 5 Name','Tax 5 Value','Phone','Receipt Number','Cancelled at'
        ]
        ,axis=1,inplace=True)
    
    # Create lineitem model, lineitem amount, unit price
    df['Lineitem model'] = df['Lineitem name'].str.split('-',n=1).str[0]
    df['Lineitem amount'] = (df['Lineitem quantity'] * df['Lineitem price']) - df['Lineitem discount']
    df['Lineitem unit price'] = df['Lineitem amount'] / df['Lineitem quantity']
    
    # Add region, not for the order lines
    conditions = [
        (df['Billing Country'] == 'US') | (df['Billing Country'] == 'CA')
        , df['Billing Country'].notna()
        ]
    results = ['NAM','INT']
    df['Region']=np.select(conditions,results,default=None)
    
    # Add distributor flag, not for the order lines
    df['Distributor'] = np.where(df['Tags'].str.contains('Distributor',case=False) == True,'Y',None)
    
    # Add source, not for the order lines

    conditions = [
        (df['Billing Company'].notna()) | (df['Distributor'] == 'Y'), #some distributors have Billing company null
        (df['Billing Company'].isna()) & (df['Currency'].notna()) #currency is condition to filter out order lines
        ] 
    results = ['B2B','DIR']
    df['Source'] = np.select(conditions,results,default=None)
    
    # Populate the Date of the order which whichever date is avalable from fulfilled backward to created, not for the order lines
    conditions = [
        df['Fulfilled at'].notna(),
        df['Fulfilled at'].isna() & df['Currency'].notna() #currency is condition to filter out order lines
        ]
    results = [df['Fulfilled at'], df['Created at']]
    df['Date'] = np.select(conditions,results,default=None) # default applies to all the lineitems
    # make Date into the YYYY-MM_DD format
    df['Date'] = df['Date'].str.split(' ',expand=True)[0]
    
    # Populate a lineitem Date which is the same as Created at because this is the only date on order lines
    df['Lineitem date'] = df['Created at'].str.split(' ',expand=True)[0]
    
    # Update the currency rates
    df_rates = make_currency_rates(df, rates_filepath, re_generate_rates)
    
    # join the rates dataframe with the df dataframe 
    df = df.merge(df_rates,on=['Date','Currency'],how='left')
    
    # Add total in usd and the other currencies, not for the order lines
    df['Total_usd'] = np.where(df['Currency'] == 'USD',df['Total'],df['Total'] * df['Rate']) # no updates on order lines where Rate is null
    
    # For distributors in which the company is on the billing name, not for the order lines
    df['Billing Company'] = np.where((df['Billing Company'].isna()) & (df['Distributor'] == 'Y')
                                   ,df['Billing Name'],df['Billing Company'])
    
    # Standardize distributors' names
    df['Billing Company'].replace(to_replace_company_names,value_company_names,inplace=True,regex=True)  #Note: regex=True
    
    # Parse tags and add Agent for distributors' sales, not for the order lines
    df['Tags'] = np.where(df['Distributor'] == 'Y',df['Tags'],'')

    df[['temp_notag','temp_AVEMEA','temp_Ludoma','temp_Power','temp_Distributor']] =    df.loc[
        df['Tags'].notna()
        ,'Tags'
        ]\
        .str.split(',')\
        .apply(lambda x: [s.replace(' ','') for s in x])\
        .apply(lambda x: [s.replace('distributor','Distributor') for s in x])\
        .str.get_dummies()

    conditions = [
        df['temp_AVEMEA'] == 1
        ,df['temp_Ludoma'] == 1
        ,df['temp_Power'] == 1
        ,df['temp_Distributor'] == 1
        ]
    results = ['CH1','CH2','CH3','CH4']
    df['Agent'] = np.select(conditions,results,default=None)

    df.drop(['temp_notag','temp_AVEMEA','temp_Ludoma','temp_Power','temp_Distributor'],axis=1,inplace=True)
    
    # Create total quantity for each order
    df['Total_quantity'] = df.join(df.groupby('OrderId')['Lineitem quantity'].sum()
                                            , on='OrderId'
                                            , how='left'
                                            , rsuffix='_1').loc[:, 'Lineitem quantity_1']

    # Update total quantity to nan for order lines
    df.loc[df.Total_usd.isna(), 'Total_quantity'] = np.nan
    
    # Save data
    df.to_csv('data/Backup_orders_after_preparation.csv',index=False)
    
    return df
    
def make_currency_rates(df, rates_filepath = 'data/rates.csv', re_generate_rates=False):
    '''
    Update the currency rates
    '''

    # for updates take from the csv file and dates after the last update
    if re_generate_rates == False: 
        try:
            df_rates = pd.read_csv(rates_filepath,usecols=['Date','Rate','Currency'])
        except Exception as e:
            print(e)
        date_start = df_rates['Date'].max()
    else:
        date_start = '2016-01-01' #to re-generate take from beginning

    date_currency_index = df.loc[
            (df['Currency'].notna())
            & (df['Currency'] != 'USD')
            & (df['Date'] > date_start)
            ,['Date','Currency']
            ]

    # make Date into datetime format for use with the currency API
    date_currency_index['Date'] = pd.to_datetime(date_currency_index['Date'].str.split(' ',n=1).str[0])

    # drop duplicates (one day and currency touple for each order)
    date_currency_index = date_currency_index.drop_duplicates()                            .sort_values('Date', ascending=False)                            .reset_index(drop=True)

    # run the currency API
    CC = CurrencyConverter()
    rate_index = [CC.convert(1, c, 'USD', date=d) for d,c in date_currency_index.values] #get_rate(c, "USD",d)
    # put together the dataframe
    df_rates_update = pd.DataFrame({
        'Date':date_currency_index['Date']
        ,'Rate':rate_index
        ,'Currency':date_currency_index['Currency']
        }
    )
    # format the Date as YYYY-MM-DD
    df_rates_update['Date'] = df_rates_update['Date'].dt.strftime('%Y-%m-%d')
    # merge the update from date_start
    if re_generate_rates == False:
        df_rates = pd.concat([df_rates,df_rates_update],axis=0, ignore_index=True)
    else:
        df_rates = df_rates_update #I am re-generating the currency rates
    # sort and reset index
    df_rates.sort_values('Date',ascending=False,inplace=True)
    df_rates.reset_index(drop=True)
    # save the rates to csv for future updates
    df_rates.to_csv(rates_filepath,columns=['Date','Rate','Currency'])
    
    return df_rates

