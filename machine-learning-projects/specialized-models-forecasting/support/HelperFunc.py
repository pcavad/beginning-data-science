#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Python
from datetime import datetime
import ipywidgets as widgets
import os

# Thrid part
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
import numpy as np
import pandas as pd
import re

def load_orders(path):
    '''
    Load orders
    '''
    
    try:
        df = pd.read_csv(path)
        df['Date']=pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df['Lineitem date']=pd.to_datetime(df['Lineitem date'], format='%Y-%m-%d')
    except Exception as e:
        print(e)
        pass
    
    return df

def make_pivot_orders(df,date_start=(2020,1,1),date_end=(2020,12,31),period='Q',df_filter=True, drill_down=['Billing Company']):
    '''
    Main report in pivot table
    '''
    
    header_mask=[
        'OrderId','Billing Company','Billing Name','Billing Country'\
         ,'Financial Status','Fulfillment Status','Date','Created at'\
         ,'Paid at','Fulfilled at','Currency','Subtotal','Shipping'\
         ,'Total', 'Total_usd','Discount Amount','Source','Region'\
         ,'Agent','Distributor','Tags','Name'
    ]

    df_filtered=df.loc[
        (df['Date']>=datetime(date_start[0], date_start[1], date_start[2]))
        & (df['Date']<=datetime(date_end[0], date_end[1], date_end[2]))
        & (df['Distributor']=='Y')
        & (df_filter) #additional custom made filter
        ,header_mask
        ]

    df_filtered['period']=df_filtered['Date'].dt.to_period(period)
    
    df_pivot=df_filtered.pivot_table(values='Total_usd', index=drill_down, columns=['period']                                         , aggfunc=np.sum, margins=True, margins_name='Total', fill_value=0)

    return df_pivot


def make_pivot_orderlines(df,date_start=(2020,1,1),date_end=(2020,12,31),period='Q',df_filter=True, drill_down=['Lineitem sku']):
    '''
    Order lines report
    '''
    
    columns_mask=[
        'OrderId','Created at','Lineitem sku','Lineitem name'\
          ,'Lineitem model','Lineitem quantity','Lineitem unit price'\
          ,'Lineitem amount'
        ]

    filt_Id=df.loc[df['Distributor']=='Y','OrderId']
    filt_sd=df.loc[df['Date']>=datetime(date_start[0], date_start[1], date_start[2]),'OrderId']
    filt_ed=df.loc[df['Date']<=datetime(date_end[0], date_end[1], date_end[2]),'OrderId']

    df_filtered=df.loc[
        (df['OrderId'].isin(filt_Id))&(df['OrderId'].isin(filt_sd))
        &(df['OrderId'].isin(filt_ed)
        &(df_filter)) #additional custom made filter
        ,columns_mask]

    #I need to define the period on a date which is present in the order lines
    df_filtered['period']=df['Lineitem date'].dt.to_period(period)

    #Use df_joined_i to add the Billing Company 
    df_filtered=df_filtered.join(df.loc[df['Distributor']=='Y',['OrderId','Billing Company']]                                     .set_index('OrderId'),on='OrderId', how='inner')

    df_pivot = df_filtered.pivot_table(values='Lineitem quantity', index=drill_down, columns=['period']                                      ,  aggfunc=np.sum, margins=True, margins_name='Total', fill_value=0)
    
    return df_pivot

def plot_pivot_orders(headers, lines):
    '''
    Plotting orders in 2 separate tabs
    '''
    out1 = widgets.Output()
    out2 = widgets.Output()

    tab = widgets.Tab(children = [out1, out2])
    tab.set_title(0, 'Orders')
    tab.set_title(1, 'Order items')
    display(tab)

    with out1:
        display(headers.style.format('{:,.0f}'))

    with out2:
        display(lines.style.format('{:,.0f}'))

def make_pivot_orders_channel(df, date_start=(2020,1,1), date_end=(2020,12,31), period='M', roll=12):
    '''
    Orders aggregated and transformed data
    '''
    year_start=date_start[0]
    month_start=date_start[1]
    day_start=date_start[2]
    year_end=date_end[0]
    month_end=date_end[1]
    day_end=date_end[2]
    set_period=period
    
    mask = ['Date', 'Agent', 'Total_usd']

    df_filtered=df.loc[
        (df['Date']>=datetime(year_start,month_start,day_start))
        & (df['Date']<=datetime(year_end,month_end,day_end))
        & (df['Distributor']=='Y')
        , mask
        ]
    
    new_index = pd.date_range(df_filtered.loc[(df.Distributor == 'Y'), 'Date'].min()
                              , df_filtered.loc[(df.Distributor == 'Y'), 'Date'].max())

    orders_totals = df_filtered.pivot_table(values='Total_usd', index='Date', columns='Agent'
                            , aggfunc=np.sum, margins=True, margins_name='Total', fill_value=0)\
                    .reindex(new_index, fill_value=0)\
                    .resample(period).sum()
    
    orders_totals.columns.name = None

    orders_totals_concat = pd.concat(
                            [
                            orders_totals
                            , orders_totals.diff().add_suffix('_diff')
                            , orders_totals.rolling(roll).mean().fillna(0).add_suffix('_roll')
                            , orders_totals.cumsum().add_suffix('_cum')
                            ]
                            , axis = 1
                            )
      
    return orders_totals_concat

def add_value_labels(ax, spacing=5,symbol='',min_label=0):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{}{:,.0f}".format(symbol,y_value)
        
        #If value is low don't show the bar
        if y_value < min_label:
            label = ''

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


def data_to_plot(df,date_start=(2020,1,1),date_end=(2020,12,31),period='Q',df_filter=True, kind='total',norm=False):
    '''
    Return data for plotting
    '''

    year_start=date_start[0]
    month_start=date_start[1]
    day_start=date_start[2]
    year_end=date_end[0]
    month_end=date_end[1]
    day_end=date_end[2]
    set_period=period

    df_plot=df.loc[
                (df['Date']>=datetime(year_start,month_start,day_start))
                &(df['Date']<=datetime(year_end,month_end,day_end))
                &(df_filter)
                ,:].copy()

    df_plot['period']=df_plot['Date'].dt.to_period(set_period)

    FILT_DISTR_H = df_plot['Distributor']=='Y'
    FILT_DISTR_L = df_plot['OrderId'].isin(df_plot.loc[df['Distributor']=='Y','OrderId'])
    # FILT_AGENT_L = df_plot['OrderId'].isin(df_plot.loc[df['Agent']=='LUD','OrderId'])

    if kind=='total':
        df_plot_returned=df_plot.groupby(['period'],as_index=True)['Total_usd'].sum().fillna(0)
    if kind=='agent':
        df_plot_returned=df_plot                         .loc[FILT_DISTR_H,:]                        .groupby(['period','Agent'],as_index=False)['Total_usd']                        .sum()                        .sort_values('period')                        .pivot(index='period',columns='Agent')                        .fillna(0)
    if kind=='distributor':
        df_plot_returned=df_plot                         .loc[FILT_DISTR_H,:]                        .groupby(['Billing Company'],as_index=False)['Total_usd']                        .sum()                        .sort_values('Total_usd',ascending=False)                        .fillna(0)
    if kind=='item':
        df_plot_returned=df_plot                         .loc[FILT_DISTR_L,:]                        .groupby(['Lineitem sku'],as_index=False)['Lineitem quantity']                        .sum()                        .sort_values('Lineitem quantity',ascending=False)                        .fillna(0)
    if kind=='model':
        df_plot_returned=df_plot                         .loc[FILT_DISTR_L,:]                        .groupby(['Lineitem model'],as_index=False)['Lineitem quantity']                        .sum()                        .sort_values('Lineitem quantity',ascending=False)                        .fillna(0)
    if kind=='currency':
        df_plot_returned=df_plot.set_index('period').resample(set_period)['Rate'].mean().to_frame()
        
    if norm==False:
        return df_plot_returned
    else:
        df_plot_returned=(df_plot_returned-df_plot_returned.min())/(df_plot_returned.max()-df_plot_returned.min())
        return df_plot_returned


def plot_dashboard (df,date_start=(2020,1,1),date_end=(2020,12,31),period='Q', figsize=(20,15), h_pad=15):
    '''
    Plot dashboard
    '''
    
    year_start=date_start[0]
    month_start=date_start[1]
    day_start=date_start[2]
    year_end=date_end[0]
    month_end=date_end[1]
    day_end=date_end[2]
    set_period=period
    
    date_start_formatted = datetime(date_start[0], date_start[1], date_start[2]).strftime('%d %b %Y')
    date_end_formatted =  datetime(date_end[0], date_end[1], date_end[2]).strftime('%d %b %Y')
    
    fig, axs = plt.subplots(2,2, figsize=figsize)
    fig.tight_layout(h_pad=h_pad)

    data_to_plot(
                    df
                    , period=period
                    , date_start = date_start
                    , date_end = date_end
                    ,kind='agent'
                    )\
                    .iloc[:,[0,1,2]]\
                    .plot(kind='bar',ax=axs[0,0])

    axs[0,0].set(title='Sale channels', xlabel='', ylabel='usd')
#     axs[0,0].get_legend().set_title('')
#     axs[0,0].get_legend().get_texts()[0].set_text('AV EMEA')
#     axs[0,0].get_legend().get_texts()[1].set_text('Ludoma')
#     axs[0,0].get_legend().get_texts()[2].set_text('Paolo')
    #call the function to add measures to the bars
    add_value_labels(axs[0,0],symbol='$',min_label=50)

    data_to_plot(
                    df
                    , date_start = date_start
                    , date_end = date_end
                    ,kind='distributor')\
                    .set_index('Billing Company')\
                    .nlargest(10,'Total_usd')\
                    .plot(kind='bar',ax=axs[0,1]
                    )

    axs[0,1].set_title('Top 10 distributors, {} - {}.'.format(date_start_formatted, date_end_formatted))
    axs[0,1].set_xlabel('')
    axs[0,1].legend('')
    #call the function to add measures to the bars
    add_value_labels(axs[0,1])

    data_to_plot(
                    df
                    , date_start = date_start
                    , date_end = date_end
                    ,kind='item')\
                    .set_index('Lineitem sku')\
                    .nlargest(10,'Lineitem quantity')\
                    .plot(kind='bar',ax=axs[1,0]
                    )

    axs[1,0].set_title('Top 10 skus for distributors, {} - {}.'.format(date_start_formatted, date_end_formatted))
    axs[1,0].set_xlabel('')
    axs[1,0].legend('')
    #call the function to add measures to the bars
    add_value_labels(axs[1,0])

    data_to_plot(
                    df
                    , date_start = date_start
                    , date_end = date_end
                    ,kind='model')\
                    .set_index('Lineitem model')\
                    .nlargest(10,'Lineitem quantity')\
                    .plot(kind='bar',ax=axs[1,1]
                    )

    axs[1,1].set_title('Best models for distributors, {} - {}.'.format(date_start_formatted, date_end_formatted))
    axs[1,1].set_xlabel('')
    axs[1,1].legend('')
    #call the function to add measures to the bars
    add_value_labels(axs[1,1])




