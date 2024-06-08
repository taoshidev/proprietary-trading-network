from neuralforecast import NeuralForecast 
from mining.config import model_path 
from signals import (
    signals_long_entry,signals_long_exit,signals_short_entry,signals_short_exit,
    process_data_for_signals, calculate_signals, 
    INDICATORS,BENCHMARKS,LONG_ENTRY,LONG_EXIT)
import pandas as pd 


# Apply the function to each group

def load_model(model_path=model_path ): 
    model = NeuralForecast.load(model_path)
    return model  

# Function to drop the last row of each group
def drop_last_row(group):
    return group.iloc[:-1]

def multi_predict(model, input, n):
    df = input.copy()
    res = []
    for i in range(n):
        pred_ds = df['ds'].tail(1).values[0]
        p = model.predict(df)
        p['pred_date'] = pred_ds
        res.append(p.groupby('unique_id').tail(1).reset_index())
        df = df.groupby('unique_id').apply(drop_last_row).reset_index(drop=True)
    
    # Concatenate the list and sort by date
    result_df = pd.concat(res).sort_values(by='ds')
    return result_df




def gen_signals_from_predictions(predictions, hist,modelname): 
    
    input =  process_data_for_signals(predictions=predictions,hist=hist,modelname=modelname)
    
    return calculate_signals(input,INDICATORS,BENCHMARKS)

 
def assess_signals(signals):
    
    print(f"signals inputted: {signals.columns}")
    
    results = {
        'long_entry': signals_long_entry(df=signals,LONG_ENTRY=LONG_ENTRY),
        'long_exit': signals_long_exit(df=signals,LONG_EXIT=LONG_EXIT),
        'short_entry': signals_short_entry(signals),
        'short_exit': signals_short_exit(signals) 
    }
    return results
              
        
from enum import Enum



class Signal(Enum):
    LONG= 'long_entry'
    FLAT = ('long_exit','short_exit')
    SHORT= 'short_entry'
    PASS = None

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if isinstance(member.value, tuple):
                if value in member.value:
                    return member
            else:
                if value == member.value:
                    return member
        return None
     
@staticmethod
def map_signals(signals):
    results = assess_signals(signals)

    true_signals = sum([value for key, value in signals.items()])
    true_keys = [key for key, value in signals.items() if value]


    if true_signals > 1:
        return 'FLAT'
    if  true_signals==0 : 
        return  'PASS'
    elif true_signals == 1:
        return Signal.from_value(true_keys[0]).name
    else:
        return False
    
 
          
