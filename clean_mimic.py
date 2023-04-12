from distutils.command.clean import clean
from xmlrpc.client import boolean
import os
from ast import arg
import numpy as np
import pandas as pd
import argparse
import importlib
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def merge_all(adm,ed,tri,pat):
    df = ed.merge(tri.drop('subject_id',axis = 1), on = 'stay_id')
    df = df.merge(adm.drop('hadm_id',axis = 1), on = 'subject_id')
    df = df.merge(pat.drop(['dod'],axis = 1), on = 'subject_id')
    df = df.drop_duplicates(subset='stay_id')
    # sort by intime
    df = df.sort_values(by = 'intime')
    df = df.set_index('stay_id')
    return df

COLUMNS_TO_KEEP = [
    'subject_id',
    'hadm_id',
    'intime',
    'admission_type',
    'admission_location',
    'temperature',
    'heartrate',
    'resprate',
    'o2sat',
    'sbp',
    'dbp',
    'pain', 
    'acuity',
    'insurance',
    'language',
    'marital_status',
    'ethnicity',
    'chiefcomplaint', 
    'gender',
    'anchor_age'
]


def adm_count(x):
    """Counts previous admissions for a subject's visits"""
    tmp = pd.Series(index=x.index)
    tmp.loc[x.index[0]]=0
    tmp.iloc[1:] = (~x.iloc[:-1]['hadm_id'].isna()).cumsum()
    tmp.name='prev_adm'

    return tmp

def remove_outliers(data,columns = ['temperature', 'heartrate', 
'resprate', 'o2sat', 'sbp', 'dbp', 'pain','acuity']):

    min_temp = 95
    max_temp = 105
    min_hr = 30
    max_hr = 300
    min_rs = 2
    max_rs = 200
    min_o2 = 50
    max_o2 = 100

    min_sbp = 30
    max_sbp = 400

    min_dbp = 30
    max_dbp = 300

    pain_min = 0
    pain_max = 20

    acu_min = 1
    acu_max = 5

    min_l = [min_temp,min_hr,min_rs,min_o2,min_sbp,min_dbp,pain_min,acu_min]
    max_l = [max_temp,max_hr,max_rs,max_o2,max_sbp,max_dbp,pain_max,acu_max]
    l = len(columns)
    x = data.copy()
    for i in range(l):
        c = columns[i]
        low = min_l[i]
        high = max_l[i]
        x.loc[(x[c]<low) | (x[c] > high),c] = float('nan')
    return x


def process_data(adm,ed,tri,pat,results_path = 'final.csv'):
    print('loading and processing mimic files...')
    adm = pd.read_csv(adm)
    ed = pd.read_csv(ed)
    tri = pd.read_csv(tri)
    pat = pd.read_csv(pat)

    print('merging...')
    df = merge_all(adm,ed,tri,pat)
    df = df[COLUMNS_TO_KEEP]

    print('adding columns..')
    ##########    
    print('previous visits..')
    df.loc[:,'prev_visit'] = df.groupby('subject_id').cumcount()
    print('previous admissions..')
    tmp = df.groupby('subject_id').apply(adm_count)  
    df = pd.merge(df,tmp, on='stay_id')

    df['y'] = ~df.hadm_id.isna()
    # filter observation admissions
    df = df.loc[~((df.y==1) 
                      & (df.admission_type.str.contains('OBSERVATION'))),:]
    ##########    

    print('removing outliers...')
    df = remove_outliers(df)

    df = df.drop(columns=['hadm_id','subject_id', 'intime', 'admission_location','admission_type'])

    print('finished processing dataset.')
    ccr = df["y"].sum()/((~df["y"]).sum())
    print(f'size: {df.shape}, cases: {df.y.sum()/len(df)}')
    print('dataset columns:',df.columns)
    print(df.head())

    print('saving...')
    df.to_csv(results_path,index= False)
    print('done.')


# add more options here later
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Input the file location for the four files from MIMIC IV.", add_help=False)
    parser.add_argument('-mimic_path', action='store', type=str,
                        default='/media/cavalab/public/mimic-iv/mimic-iv-1.0/',
                        help='Path for admission file')
    parser.add_argument('-Admission_File', action='store', type=str,
                        default='core/admissions.csv.gz',
                        help='Path for admission file')
    parser.add_argument('-Edstay_File', action='store', type=str,
                        default='ed/edstays.csv.gz',
                        help='Path for edstay file') 
    parser.add_argument('-Triage_File', action='store', type=str,
                        default='ed/triage.csv.gz',
                        help='Path for Triage File')
    parser.add_argument('-Patient_File', action='store', type=str,
                        default='core/patients.csv.gz',
                        help='Path for Patient file')      
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-p', action='store',
                        dest='PATH',default='mimic4_admissions.csv',type=str,
            help='Path of Saved final fire')

    args = parser.parse_args()

    process_data(os.path.join(args.mimic_path, args.Admission_File), 
                 os.path.join(args.mimic_path, args.Edstay_File), 
                 os.path.join(args.mimic_path, args.Triage_File), 
                 os.path.join(args.mimic_path, args.Patient_File),
                 results_path = args.PATH) 
 
