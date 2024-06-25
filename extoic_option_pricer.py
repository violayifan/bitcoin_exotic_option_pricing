# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:39:28 2024

@author: viola
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.misc import derivative
import os
import seaborn as sns



class snowball_option(object):
    
    def __init__(self,S0,K,vol,KI_barrier,KO_barrier,KO_obs_period,annualized_coupon,T,obs_reminder=1):
        self.S0=S0
        self.K=K
        self.vol=vol
        self.KI_barrier=KI_barrier
        self.KO_barrier=KO_barrier
        self.KO_obs_period=KO_obs_period
        self.coupon=annualized_coupon
        self.T=T
        self.obs_reminder=obs_reminder
        self.expiry_before_maturity=False
        self.obs_point=None
    
    
    def payoff_call(self,Spath):
        
        #get maturity
        T=self.T
        KO_price=self.S0*self.KO_barrier
        KI_price=self.S0*self.KI_barrier
        
        #check if knock out on knock-out observation day        
        daily_index=list(filter(lambda t:t%24==0,range(len(Spath))))[1:]
        obs_index=np.array([self.obs_reminder+i*self.KO_obs_period for i in range(int(np.ceil((len(Spath)-1)/24/7)))])
        obs_index=obs_index[obs_index<=self.T]*24
        
        S_obs=Spath[[obs_index]]
        S_daily=Spath[[daily_index]]
        
        if np.max(S_obs)>= KO_price:
            knock_out_index=obs_index[np.where(S_obs>=KO_price)[0][0]]
            expiry_time= knock_out_index/24
            payoff_amount= 1+self.coupon * expiry_time/365
            
            if expiry_time<T:
                self.expiry_before_maturity=True
        else:
            if np.min(S_daily)>= KI_price:
                payoff_amount=1+self.coupon*T/365
                expiry_time=T
            else:
                payoff_amount= min(1,Spath[-1]/self.K)
                expiry_time=T
        return payoff_amount,expiry_time
    
    def payoff_put(self,Spath):
        
        #get maturity
        T=self.T
        KO_price=self.S0*self.KO_barrier
        KI_price=self.S0*self.KI_barrier
        
        #check if knock out on knock-out observation day        
        daily_index=list(filter(lambda t:t%24==0,range(len(Spath))))[1:]
        obs_index=np.array([self.obs_reminder+i*self.KO_obs_period for i in range(int(np.ceil((len(Spath)-1)/24/7)))])
        obs_index=obs_index[obs_index<=self.T]*24
        
        S_obs=Spath[[obs_index]]
        S_daily=Spath[[daily_index]]
        
        if np.min(S_obs)<= KO_price:
            knock_out_index=obs_index[np.where(S_obs<=KO_price)[0][0]]
            expiry_time= knock_out_index/24
            payoff_amount= 1+self.coupon * expiry_time/365
            
            if expiry_time<T:
                self.expiry_before_maturity=True
        else:
            if np.max(S_daily)<= KI_price:
                payoff_amount=1+self.coupon*T/365
                expiry_time=T
            else:
                payoff_amount= min(1,self.K/Spath[-1])
                expiry_time=T
        return payoff_amount,expiry_time



    
class shark_fin(object):
    
   def __init__(self,S0,K1,K2,T,APR_low,APR_mid,APR_high,vol):
       
        self.S0=S0
        self.K1=K1
        self.K2=K2
        self.T=T
        self.vol=vol
        
        self.APR_low=APR_low
        self.APR_mid=APR_mid
        self.APR_high=APR_high
   
    
   def payoff_up_and_out_call(self,Spath):
       
       ST=Spath[-1]
       expiry_time=self.T
       
       if ST<self.K1:
           payoff_amount=1+self.APR_low*expiry_time/365
           
       elif ST>self.K2:
           payoff_amount=1+self.APR_low*expiry_time/365

       else:
           payoff_amount=1+(self.APR_mid+ \
                            (self.APR_high- self.APR_mid)/(self.K2-self.K1) *(ST-self.K1))*expiry_time/365
           

       return payoff_amount,expiry_time
   
   def payoff_down_and_out_put(self,Spath):
        
        ST=Spath[-1]
        expiry_time=self.T
        
        if ST<self.K1:
           payoff_amount=1+self.APR_low*expiry_time/365
           
        elif ST>self.K2:
           payoff_amount=1+self.APR_low*expiry_time/365
          
        else:
            payoff_amount=1+(self.APR_high-\
                             (self.APR_high- self.APR_mid)/(self.K2-self.K1)*(ST-self.K1))*expiry_time/365
        
        return payoff_amount,expiry_time


         
def mc_pricer(option,payoff_func,rf=0.1,div=0,N=24*18,Npaths=10000):       
    #generate stock path
    S0=option.S0
    vol=option.vol
    T=option.T/365
    dt=T/N
    np.random.seed(0)
    dw=np.random.standard_normal((Npaths,N))
    #start mc simulation
    Spath1=np.cumprod(np.exp((rf-div-0.5*vol**2)*dt+vol*np.sqrt(dt)*dw),axis=1)*S0
    Spath2=np.cumprod(np.exp((rf-div-0.5*vol**2)*dt+vol*np.sqrt(dt)*-dw),axis=1)*S0
    Spath1=np.concatenate((np.ones((Npaths,1))*S0,Spath1),axis=1)
    Spath2=np.concatenate((np.ones((Npaths,1))*S0,Spath2),axis=1)
    pv_sum=0
    for path_index in range (Npaths):
        path1=Spath1[path_index]
        path2=Spath2[path_index]
        payoff1,expiry_time1=eval("option.{}(path1)".format(payoff_func))
        payoff2,expiry_time2=eval("option.{}(path2)".format(payoff_func))
        pv1=payoff1*np.exp(-rf*expiry_time1/365)
        pv2=payoff2*np.exp(-rf*expiry_time2/365)
        pv=0.5*(pv1+pv2)
        pv_sum += pv
    pv=pv_sum/Npaths
    return pv
                

def calculate_fair_coupon_snowball(pricer_func,option_class,payoff_func,S0,K,vol,\
                         KI_barrier,KO_barrier,KO_obs_period,T):
    
    #get coupon when option price=captial=1
    coupon=brentq(lambda x: pricer_func(option_class(S0,K,vol,\
                         KI_barrier,KO_barrier,KO_obs_period,x,T),payoff_func)-1,-10,10)
    return coupon


def calculate_implied_vol_snowball(pricer_func,option_class,payoff_func,S0,K,coupon,\
                         KI_barrier,KO_barrier,KO_obs_period,T):
    
    #get implied vol when option price=captial=1
    vol=brentq(lambda x: pricer_func(option_class(S0,K,x,\
                         KI_barrier,KO_barrier,KO_obs_period,coupon,T),payoff_func)-1,0,5)
    return vol

if __name__=='__main__':
    
    
    work_path=os.getcwd()

    # bitcoin estimated annualized vol
    df_bitcoin_px=pd.read_csv(os.path.join(work_path,"BTC-USD.csv"),index_col=0,parse_dates=True)
    vol=np.mean(df_bitcoin_px["Close"].pct_change().resample("M").std()*np.sqrt(365))
    
    #snowball param(KO_obs_period:weekly)
    S0,K,KI_barrier,KO_barrier,KO_obs_period,annualized_coupon,T=63168,63168,0.92,1.03,7,0.66,18
    snowball=snowball_option(S0,K,vol,KI_barrier,KO_barrier,KO_obs_period,annualized_coupon,T)
    price1=mc_pricer(snowball,"payoff_call")
    
    coupon=calculate_fair_coupon_snowball(mc_pricer,snowball_option,"payoff_call",S0,K,vol,KI_barrier,KO_barrier,KO_obs_period,T)

    
    shark_fin_opt=shark_fin(62331.6,62900,74800,7,0.04,0.045,0.13,vol)
    price2=mc_pricer(shark_fin_opt,"payoff_up_and_out_call",N=24*7)

    



    
    













