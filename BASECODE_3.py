# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:19:07 2025

@author: testr
"""

import numpy as np
import pandas as pd
import os
import math
from numba import float64, vectorize, boolean, types,int64
from flask import Flask, request, jsonify

import time
import threading


#%%
app = Flask(__name__)
timing_data = {}

#%%


class calculator:
    def __init__(self):
        self.rfr = .0711
        self.ivv = .1449
        self.n = 1
        
        #check for date range
        
    ######### vectorized functions #############
    @staticmethod
    @vectorize([types.boolean(types.int64, types.int64, types.int64)], nopython=True)
    def is_date_in_range(date_to_check, start_date, end_date):
        return start_date <= date_to_check <= end_date
    
    #round-up  saving gunction
    @staticmethod
    @vectorize([float64(float64)])
    def roundup_100(x):
        return math.ceil(x/100)*100
    
    #sum function for transactions
    @staticmethod
    @vectorize([float64(float64, float64, float64)])
    def sum_func(remanent, fixed, extra):
        if math.isnan(extra):
            extra = 0
        if math.isnan(fixed):
            return remanent + extra
        elif remanent >= fixed:
            return fixed + extra
        elif remanent < fixed:
            return extra
        else:
            return remanent
        
    #date evaluation
    @staticmethod    
    @vectorize([int64(types.int64, types.int64, types.int64, types.int64)]) 
    def range_date_eval(date_to_check, start_date, end_date, index):
        if start_date <= date_to_check <= end_date:
            return index
        else:
            return -1  # if not within the range
    
    
    ###################
    
    #sum transaction
    def transaction_date_range_sum(self, df,k, date_column='date', amount_column='amount'):
        df['date_int'] = df['date'].astype(np.int64)
        new_df = df.copy()
        #new_df['date'] = new_df['date']
        k['amount'] = 0.0
        #ew_df['amount'] = 0.0
        k['start_int'] = k['start'].astype(np.int64)
        k['end_int'] = k['end'].astype(np.int64)
        for index, row in new_df.iterrows():
            date_to_check = row['date_int']

            #date_to_check = date

            #Iterate the range and use Numba to evaluate if the date fits in the range and get the value if so.
            for indexRange, range_row in k.iterrows():
                start_date_int = range_row['start_int']
                end_date_int = range_row['end_int']

                is_valid = self.is_date_in_range(date_to_check, start_date_int, end_date_int)

                #If the date is valid, sum the amount value
                if (is_valid):
                    k.at[indexRange,'amount'] += row['remanent']

        return k[['start','end','amount']]
    
    #p-q-k input proceessing state
    def constraints_processor(self,data,switch=0):
        data = pd.DataFrame(data)
        try:
            data['start'] = pd.to_datetime(data['start'],format='%Y-%m-%d %H:%M')
            data['end'] = pd.to_datetime(data['end'],format='%Y-%m-%d %H:%M')
        except:
            data['start'],data['end'] = pd.to_datetime(data['start']),pd.to_datetime(data['end'])
        if switch == 0:
            data.iloc[:, [0]] = data.iloc[:, [0]].astype(np.float64)
        data['check'] = data['end']-data['start']
        valid_vals = data['check'].astype(np.int64)>= 0
        invalid_vals = data['check'].astype(np.int64)>= 0
        data = data[valid_vals]
        invalid_df = data[invalid_vals]
        return data,invalid_df
    
    #transaction inputs processing
    def transaction_input_processor(self,transaction,verify = False):
        data = pd.DataFrame(transaction)
        data['amount'] = data['amount'].astype(np.float64)
        try:
            data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d %H:%M')
        except:
            data['date'] = pd.to_datetime(data['date'])
        data['ceiling'] = self.roundup_100(data['amount'].values)
        #data['remanent'] = data['ceiling']-data['amount']
        if verify == True:
            valid_vals = data['amount']>= 0
            invalid_vals = data['amount']< 0
            data = data[valid_vals]
            invalid_df = data[invalid_vals]
            return data,invalid_df
        else:
            return data
    #date dependencies for each datae reange
    def get_val_for_date(self,date_to_check, range_df):
        indexes = np.arange(len(range_df))
        # Call vectorized function 
        range_indexes = self.range_date_eval(date_to_check, 
                                             range_df['start_int'].values, 
                                             range_df['end_int'].values, 
                                                         indexes)
        range_indexes = range_indexes[range_indexes != -1]
    
        if len(range_indexes) > 0:
            range_index = range_indexes[0]
            return range_df.iloc[:, [0]].iloc[range_index].sum()
        return None #If there is not a valid range
    
    

    #post processfor sum function evaluation
    def post_process_constraints(self,range_df,col):
        #set intersections
        #range_df['start'] = pd.to_datetime(range_df['start'])
        #range_df['end'] = pd.to_datetime(range_df['end'])

        range_df = range_df.sort_values(by='start').reset_index(drop=True)

        new_rows = []
        i = 0
        while i < len(range_df):
            current_start = range_df['start'][i]
            current_end = range_df['end'][i]
            current_value = range_df[col][i]

            j = i + 1
            while j < len(range_df) and range_df['start'][j] <= current_end:
                next_start = range_df['start'][j]
                next_end = range_df['end'][j]
                next_value = range_df[col][j]

                if next_start > current_start:
                    new_rows.append({'start': current_start, 'end': next_start - pd.Timedelta(minutes=1), col: current_value})
                    current_value = current_value
                    current_start = next_start

                if next_end < current_end:
                    new_rows.append({'start': next_start, 'end': next_end, col: current_value+next_value})
                    current_start = next_end+ pd.Timedelta(minutes=1)
                    current_value=current_value

                else:
                    new_rows.append({'start': next_start, 'end': current_end, col: current_value+next_value})
                    current_start = current_end+ pd.Timedelta(minutes=1)
                    current_value=current_value

                j+=1
                i=j # advance the pointer i to the next non processed value

            new_rows.append({'start': current_start, 'end': current_end, col: current_value}) #append not processed dates

            i+=1

        range_df = pd.DataFrame(new_rows)
        range_df = range_df.sort_values(by='start').reset_index(drop=True)
        
        return range_df[[col,'start','end']]
    #post processs for date range evaluation
    def post_process_daterange(self,k):
    
        k['start'] = pd.to_datetime(k['start'])
        k['end'] = pd.to_datetime(k['end'])
 
        k = k.sort_values(by='start').reset_index(drop=True)
 
        new_rows = []
        i = 0
        while i < len(k):
            current_start = k['start'][i]
            current_end = k['end'][i]
 
            j = i + 1
            while j < len(k) and k['end'][j] <= current_end:
                next_start = k['start'][j]
                next_end = k['end'][j]
 
                if next_start > current_start:
                    new_rows.append({'start': current_start, 'end': next_start - pd.Timedelta(minutes=1)})
                    current_start = next_start
 
                if next_end < current_end:
                    new_rows.append({'start': next_start, 'end': next_end})
                    current_start = next_end + pd.Timedelta(minutes=1)
 
                else:
                    new_rows.append({'start': next_start, 'end': current_end})
                    current_start = current_end + pd.Timedelta(minutes=1)
 
                j += 1
            new_rows.append({'start': current_start, 'end': current_end})
            i += 1
 
        k = pd.DataFrame(new_rows)
        k = k[k['start'] <= k['end']]
        k = k.sort_values(by='start').reset_index(drop=True)
        k = k.drop_duplicates(subset=['start', 'end'], keep='last')
        return k


    #main container of date  and constraint evaluation

    def t_date_eval(self,data,p,col):
        p = self.post_process_constraints(p, col)
        p['start_int'] = p['start'].astype(np.int64)
        p['end_int'] = p['end'].astype(np.int64)
        data[f'{col}_value'] = data['date_int'].apply(self.get_val_for_date, args=(p,))
        return data
    
    
    
    
    #def policy_transaction(self,data):
        
    #q fixed amount
    #p extra amount
    #k date range
    #main  container of evaluations_verifications
    def transaction_verification(self,data,verify,eoy,q,p,k=[]):
        if verify == False:
            data['remanent'] = data.ceiling-data.amount
        else:
            data['remanent'] = data.ceiling-data.amount
            data['date_int'] = data['date'].astype(np.int64)
            data['remanent']  = data['remanent'].astype(np.float64)

            if len(p)>0:
                data =  self.t_date_eval(data,p,'extra')
            else:
                data['extra_value'] = np.nan
            if len(q)>0:
                data =  self.t_date_eval(data,q,'fixed')
            else:
                data['fixed_value'] = np.nan
            data['fixed_value']  = data['fixed_value'].astype(np.float64)
            data['extra_value']  = data['extra_value'].astype(np.float64)
            data['remanent_1'] = self.sum_func(data['remanent'].values,
                                               data['fixed_value'].values,
                                               data['extra_value'].values)
            intermediate  = data[['date','amount','ceiling','remanent_1']]
            intermediate  = intermediate.rename(columns = {'remanent_1':'remanent'})
            data = intermediate
        
        return data
    #main transaction function
    def transaction_main_function(self,transactions,verify = False,wage=0,age=0,eoy =False,q=[],p = [],k=[]):
        
        #self.wage = wage
        #self.age =age
        #print(q)
        if verify == False:
            transactions = self.transaction_input_processor(transactions,verify)
        else:
            transactions,t_errors = self.transaction_input_processor(transactions,verify)
        if verify==True:
            if len(q)>0:
                q,q_error= self.constraints_processor(q)
            if len(p)>0:
                p,p_error = self.constraints_processor(p)
            if len(k)>0:
                self.k_out,k_error = self.constraints_processor(k,1)
        intermediate = self.transaction_verification(transactions,verify,eoy,q,p,k)
        
        if verify == False:
            return intermediate
        else:
            return intermediate,t_errors
    ####### finance functions
    def ppr_calc(self):
        return self.ytor*(min(self.cap_wage,self.amount_inv)) + self.amount_inv*(1+(self.rfr/self.n))**(self.n*self.ytor)
    
    def ret_calc(self):
        return self.amount_inv*(1+(self.rfr/self.n))**(self.n*self.ytor)
    
    def infl_adj(self,value):
        return value/(1+(self.infl/100))**self.ytor
    
    #########
    
    def return_main(self,type_ret,transactions,age,wage,inflation):
        k = self.k_out
        k_custom = self.post_process_daterange(k)
        self.infl = inflation
        self.ytor = 65-age +1  if 65>age else 5 
        self.cap_wage = min(wage*12*.1,206367.6) ##### added cap for max amount based in current fiscal law
        #k['amount'] = 0.0
        k_range = self.transaction_date_range_sum(transactions,k)
        investment_amount = self.transaction_date_range_sum(transactions,k_custom)
        self.amount_inv = investment_amount['amount'].sum()
        if type_ret =='ppr':
            ppr_ret = self.ppr_calc()
            adj_ret = self.infl_adj(ppr_ret)
        else:
            ivv_ret = self.ret_calc()
            adj_ret = self.infl_adj(ivv_ret)
        return k_range,investment_amount,adj_ret
#%%
    
data_processor = calculator()
#%%
@app.route('/blackrock/challenge/v1/transactions:parse', methods=['POST'])
def api_transactions():
    features = request.json
    #transactions = features.get('')
    start_time = time.time()
    response = data_processor.transaction_main_function(features,False)
    end_time = time.time()
    latency = end_time - start_time
    response = response.to_dict(orient='records')
    timing_data['last_execution_time'] = latency
    return jsonify(response)

@app.route('/blackrock/challenge/v1/transactions:validator', methods=['POST'])
def api_transactions_validator():
    features = request.json
    wage = features.get('wage')
    transactions = features.get('transactions')
    start_time = time.time()

    valid,invalid = data_processor.transaction_main_function(transactions,True,wage)
    end_time = time.time()
    latency = end_time - start_time
    
    valid = valid.to_dict(orient='records')
    invalid = invalid.to_dict(orient='records')
    timing_data['last_execution_time'] = latency
    return jsonify({
                    'valid':valid,
                    'invalid':invalid})

@app.route('/blackrock/challenge/v1/transactions:filter', methods=['POST'])
def api_transactions_filter():
    features = request.json
    wage = features.get('wage')
    transactions = features.get('transactions')
    p = features.get('p')
    q = features.get('q')
    k = features.get('k')
    start_time = time.time()
    valid,invalid = data_processor.transaction_main_function(transactions,True,wage,0,False,q,p,k)
    end_time = time.time()
    latency = end_time - start_time
    
    valid = valid.to_dict(orient='records')
    invalid['message'] = 'Negative amounts not allowed'
    invalid = invalid.to_dict(orient='records')
    timing_data['last_execution_time'] = latency
    return jsonify({
                    'valid':valid,
                    'invalid':invalid})

@app.route('/blackrock/challenge/v1/returns:ppr', methods=['POST'])
def api_feature4():
    features = request.json
    wage = features.get('wage')
    age = features.get('age')
    try:
        inflation = features.get('inflation')
    except:
        inflation = 4.83 #####default  inflation
    transactions = features.get('transactions')
    p = features.get('p')
    q = features.get('q')
    k = features.get('k')
    if len(k)<1:
        return jsonify('empty time series provided')
    start_time = time.time()
    valid,invalid_T = data_processor.transaction_main_function(transactions,True,wage,0,False,q,p,k)
    k_ranges,investment_amount,profit = data_processor.return_main("ppr",valid,age,wage,inflation)
    invested_total = investment_amount['amount'].sum()
    ceiling_total = valid['remanent'].sum()
    transactions_total = valid['amount'].sum()
    end_time = time.time()
    latency = end_time - start_time
    k_ranges_dict = k_ranges.to_dict(orient='records')
    timing_data['last_execution_time'] = latency
    return jsonify({
                    'transactionsTotalAmount':transactions_total,
                    'transactionsTotalceiling':ceiling_total,
                    'investedAmount':invested_total,
                    'profits':profit,
                    'savingsByDates':k_ranges_dict})

@app.route('/blackrock/challenge/v1/returns:ishares', methods=['POST'])
def api_feature5():
    features = request.json
    wage = features.get('wage')
    age = features.get('age')
    try:
        inflation = features.get('inflation')
    except:
        inflation = 4.83 #####default inflation
    transactions = features.get('transactions')
    p = features.get('p')
    q = features.get('q')
    k = features.get('k')
    if len(k)<1:
        return jsonify('empty time series provided')
    start_time = time.time()
    valid,investment_amount,invalid = data_processor.transaction_main_function(transactions,True,wage,0,False,q,p,k)
    k_ranges,investment_amount,profit = data_processor.return_main("ivv",valid,age,wage,inflation)
    invested_total = investment_amount['amount'].sum()
    ceiling_total = valid['remanent'].sum()
    transactions_total = valid['amount'].sum()
    end_time = time.time()
    latency = end_time - start_time
    k_ranges_dict = k_ranges.to_dict(orient='records')
    timing_data['last_execution_time'] = latency
    return jsonify({
                    'transactionsTotalAmount':transactions_total,
                    'transactionsTotalceiling':ceiling_total,
                    'investedAmount':invested_total,
                    'profits':profit,
                    'savingsByDates':k_ranges_dict})

@app.route('/blackrock/challenge/v1/performance', methods=['GET'])
def get_metrics():
        # Get memory usage and thread count at the time of metric retrieval
     num_threads = threading.active_count()
    
     # Check if timing data is available
     if 'last_execution_time' in timing_data:
         latency = timing_data['last_execution_time']
    
         # Clear timing data after retrieval
         timing_data.clear() #Avoid old data issues - only use it once
     else:
         latency = None  # Or some default value
         result = "No data"
    
     return jsonify({
         'latency in seconds': latency,
         'active_threads': num_threads,
     })

    
#%%
CONTAINER_PORT = int(os.environ.get('CONTAINER_PORT', 80))

HOST_PORT = int(os.environ.get('HOST_PORT', 5477))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=CONTAINER_PORT) #