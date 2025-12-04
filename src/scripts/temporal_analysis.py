import matplotlib.pyplot as plt
import pandas as pd 

def plot_temporal(df):
    """
    Plot the temporal distribution of negative share in different timestamps 

    :param df: pandas DataFrame of the Reddit data
    """

    #Create a binary column indicating negative sentiment
    df['is_negative'] = (df['LINK_SENTIMENT'] == -1).astype(int)

    #Plot negativity time series
    def plot_negativity_time_series(df):
        #Combine all timestamps to weekly frequency and compute mean negativity
        series = df.set_index('TIMESTAMP').resample('W')['is_negative'].mean()
        plt.figure()
        plt.plot(series.index, series.values)
        plt.xlabel('Time')
        plt.ylabel('Share negative')
        plt.title("Share of negative links over time")
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.show()

    plot_negativity_time_series(df)


    #Plot negativity by period of day

    def negativity_by_period(df):
        neg_count = df[df['is_negative'] == 1].groupby('day_period')['is_negative'].count().reindex(['Night','Morning','Afternoon','Evening'])
        pos_count = df[df['is_negative'] == 0].groupby('day_period')['is_negative'].count().reindex(['Night','Morning','Afternoon','Evening'])
        counts = pd.DataFrame({'Negative': neg_count, 'Non-negative': pos_count})
        counts.plot(kind='bar', stacked=True)
        plt.title('Count of negative and non-negative links by period of day')
        plt.xlabel('Period')
        plt.ylabel('Count')
        plt.grid(axis='y')
        plt.show()

    negativity_by_period(df)

    def negativity_by_day_of_week(df):
        df['day_of_week'] = df['TIMESTAMP'].dt.day_name()
        agg = df.groupby('day_of_week')['is_negative'].mean()
    
        agg = agg.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        plt.figure()
        plt.bar(agg.index, agg.values)
        plt.title('Share of negative links by day of week')
        plt.xlabel('Day')
        plt.ylabel('Share negative')
        plt.grid(axis='y')
        plt.show()
    
    negativity_by_day_of_week(df)


    def negativity_by_month(df):
        df['month'] = df['TIMESTAMP'].dt.month_name()
        agg = df.groupby('month')['is_negative'].mean()
    
        months_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        agg = agg.reindex(months_order)
        plt.figure()
        plt.plot(agg.index, agg.values, marker='o')
        plt.title('Share of negative links by month')
        plt.xlabel('Month')
        plt.ylabel('Share negative')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    negativity_by_month(df)
