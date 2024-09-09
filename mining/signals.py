
import pandas as pd 

LONG_EXIT = 'rsma3 crosses above upper1.5'
LONG_ENTRY = 'rlower1 crosses above ma'	
INDICATORS = ['rsma3','upper1.5','rlower1','ma']
BENCHMARKS = ['rsma3','upper1.5','rlower1','ma']

def process_data_for_predictions(input): 
    input['ma'] = input['close'].rolling(12).mean()
    input['std'] = input['close'].rolling(12).std()

    input.ds= pd.to_datetime(input.ds, unit='ms')

    return pd.melt( input, id_vars=['ds', 'close', 'low', 'high','volume'], value_vars=['ma', 'std'],
                        var_name='unique_id', value_name='y').dropna()

def process_data_for_signals(predictions, hist,modelname): 
      
    hist = hist[['ds','close']]
    hist = hist.drop_duplicates()
    predictions  = predictions.drop(columns='ds')
    predictions.rename(columns={'pred_date': 'ds'}, inplace=True)
    preds_df  = predictions .pivot(columns='unique_id', values=modelname,index='ds').reset_index()
    
    hist['ds']= hist['ds'].astype('str')
    preds_df['ds']= preds_df['ds'].astype('str')
    
    # Calculate rolling means
    for period in [3, 6, 12, 24]:
        hist[f'rsma{period}'] = hist['close'].rolling(period).mean()

    # Calculate rolling standard deviation
    hist['rstd12'] = hist['close'].rolling(12).std()

    # Define multipliers for upper and lower bands
    multipliers = [1, 1.5, 2]
    
    
    bktst= pd.merge(preds_df,hist,on='ds',how='left')


    # Calculate upper and lower bands for 'ma' and 'rsma12'
    for multiplier in multipliers:
        bktst[f'lower{multiplier}'] = bktst['ma'] - (bktst['std'] * multiplier)
        bktst[f'upper{multiplier}'] = bktst['ma'] + (bktst['std'] * multiplier)
        bktst[f'rlower{multiplier}'] = bktst['rsma12'] - (bktst['rstd12'] * multiplier)
        bktst[f'rupper{multiplier}'] = bktst['rsma12'] + (bktst['rstd12'] * multiplier)
        
        
    return bktst

def calculate_signals(df, indicators, benchmarks):
    signals = {}

    for indicator in indicators:
        for benchmark in benchmarks:
            if indicator != benchmark:
                # Greater than conditions
                signals[f"{indicator} > {benchmark}"] = df[indicator] > df[benchmark]

                # Less than conditions
                signals[f"{indicator} < {benchmark}"] = df[indicator] < df[benchmark]

                # Cross over conditions
                signals[f"{indicator} crosses above {benchmark}"] = (df[indicator].shift(1) < df[benchmark].shift(1)) & (df[indicator] >= df[benchmark])

                # Cross under conditions
                signals[f"{indicator} crosses below {benchmark}"] = (df[indicator].shift(1) > df[benchmark].shift(1)) & (df[indicator] <= df[benchmark])

    signals_df = pd.DataFrame(signals)
    return signals_df.tail(1)




def signals_long_entry(df,LONG_ENTRY): 
    return df[LONG_ENTRY].tail(1).values[0]

def signals_long_exit(df): 
    return df[LONG_EXIT].tail(1).values[0]

def signals_short_entry(df): 
    return None

def signals_short_exit(df):
    return None 
