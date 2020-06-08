import numpy as np
import pandas as pd
from operator import add

class Book:
    ''' Collection of positions'''
    def __init__(self, name, pos=pd.Series()):
        self._pos = pd.Series()
        self.add_pos(pos)
        
    def add_pos(self, pos):
        if isinstance(pos, dict):
            pos = pd.Series(pos)
        elif not isinstance(pos, pd.Series):
            raise ValueError()    
        self._pos = self._pos.combine(pos, add, fill_value=0.0)
        
    @property
    def pos(self):
        return self._pos
    def __add__(self, other):
        return Book('', self._pos.combine(other._pos, add, fill_value=0.0))
        
        
def get_example_data():
    state = pd.read_csv('state.csv', sep='\t', index_col=0).set_index('TICKER', drop=True)
    lvl_df = pd.read_csv('lvl.csv', sep='\t', index_col=0)
    lvl_df.rename(columns=dict(zip(state['PRICE_PISIN'], state.index)), inplace=True)    
    mask = (state['VAL_BASECCY'] != 0.0) & (state['PRODUCTID'] == 'STOCK')
    book = Book('P')
    book.add_pos(state[mask]['VAL_BASECCY'])
    return lvl_df, book