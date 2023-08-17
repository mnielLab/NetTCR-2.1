#!bin/bash

import os, sys, time
start_time = time.time()

#Silence TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#NETTCR = "/home/projects/vaccine/people/almon/nettcr/"
##NETTCR = $NMHOME
NETTCR = os.environ['NETTCR']

#Set working directory
sys.path.append(NETTCR)

from argparse import ArgumentParser

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("--jobid")
parser.add_argument("-i", "--infile", help="Specify input file with TCR sequences")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
parser.add_argument("-c", "--cdr", default='ab', help="options: cdr3, cdr123. Default is cdr123.")
parser.add_argument("-p", "--peptides", help="Specify peptides (comma separated).")
args = parser.parse_args()

import pandas as pd
import numpy as np
from torch import load
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils import enc_list_bl_max_len, blosum50_20aa
from nettcr_archs import NetTCR_CDR123_pep, NetTCR_CDR3_pep, ConvBlock, CatConv




pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# get input peptides:
if args.peptides != None:
  peplist = args.peptides.upper().split(",")
else:
  print("Please specify at least one peptide!\n", file=sys.stdout)
  sys.exit(2)

#print("# Specified chain:", args.chain)

# get input data (TCR list):
if args.infile != None:
#    print("# Input: " + args.infile )
    inputfile = args.infile
else:
    print("Please specify input!\n", file=sys.stdout)
    sys.exit(2)



# get output file:
if args.outfile != None:
    outfile = args.outfile
else:
    outfile = 'NetTCR_' + str(args.cdr) + '_pred.csv'


## Declarations
avg_pred = 0
encoding = blosum50_20aa
a1_max = 10
a2_max = 8
a3_max = 20
b1_max = 10
b2_max = 20
b3_max = 20
pep_max = 13
n_partition = 5

#print("# Loading data..")

# Define function to make the dataset
def make_tensor_CDR3(df, encoding):
    encoded_a3 = torch.tensor(enc_list_bl_max_len(df.CDR3a, encoding, a3_max)/5, dtype=float)
    encoded_b3 = torch.tensor(enc_list_bl_max_len(df.CDR3b, encoding, b3_max)/5, dtype=float)

    tensor_ds = TensorDataset(encoded_a3.float(), 
                              encoded_b3.float())

    return tensor_ds

# Define function to make the dataset
def make_tensor_CDR123(df, encoding):
    encoded_a1 = torch.tensor(enc_list_bl_max_len(df.A1, encoding, a1_max)/5, dtype=float)
    encoded_a2 = torch.tensor(enc_list_bl_max_len(df.A2, encoding, a2_max)/5, dtype=float)
    encoded_a3 = torch.tensor(enc_list_bl_max_len(df.CDR3a, encoding, a3_max)/5, dtype=float)
    encoded_b1 = torch.tensor(enc_list_bl_max_len(df.B1, encoding, b1_max)/5, dtype=float)
    encoded_b2 = torch.tensor(enc_list_bl_max_len(df.B2, encoding, b2_max)/5, dtype=float)
    encoded_b3 = torch.tensor(enc_list_bl_max_len(df.CDR3b, encoding, b3_max)/5, dtype=float)

    tensor_ds = TensorDataset(encoded_a1.float(), encoded_a2.float(), encoded_a3.float(), 
                              encoded_b1.float(), encoded_b2.float(), encoded_b3.float())

    return tensor_ds

in_data = pd.read_csv(inputfile, header=None, delimiter=',', engine='python')

# Checking that the data is not genomic data
# i.e. checking that the input has other chars than only A, T, C, G

if set(in_data.iloc[:,0].sum()).issubset({"C","A","T","G"}):
    print("It seems the input data is genomic data. Only A, T, C, G found.\n Please input some protein sequences", file=sys.stdout)
    sys.exit(2)

if args.cdr=="cdr3":
    if (len(in_data.columns)!=2):
        print("Error: {} columns found. Only 2 expected (CDR3a and CDR3b)!\n".format(len(in_data.columns)),file=sys.stdout)
        sys.exit(2)
    data = pd.read_csv(inputfile, names=["CDR3a","CDR3b"], delimiter=',', engine='python')
    data['peptide'] = [peplist]*len(data)
    data = data.explode('peptide').reset_index(drop=True)
    
    covid_pred_lists = pd.read_csv(NETTCR+'out/covid_background_nettcr_pep_cdr3.csv')
    
elif args.cdr=="cdr123":
    if (len(in_data.columns)!=6):
        print("Error: {} columns found. Six expected!\n".format(len(in_data.columns)),file=sys.stdout)
        sys.exit(2)
    data = pd.read_csv(inputfile, names=["A1","A2","CDR3a","B1","B2","CDR3b"], delimiter=',', engine='python')
    data['peptide'] = [peplist]*len(data)
    data = data.explode('peptide').reset_index(drop=True)
    data_tensor = make_tensor_CDR123(data, encoding=encoding)
    covid_pred_lists = pd.read_csv(NETTCR+'out/covid_background_nettcr_pep_cdr123.csv')
    
#data_load_time = time.time()-start_time

#print("Data loading time", data_load_time)

# Make predictions with  the 20 models
#print("# Predicting..")
pred_df = pd.DataFrame()
cat_df = pd.DataFrame()
print('Starting prediction', file=sys.stderr)
from time import perf_counter
start = perf_counter()
for pep in peplist:
    pep_data = data[data.peptide==pep]
    cat_df = pep_data.copy()
    if args.cdr=="cdr3":
        data_tensor = make_tensor_CDR3(pep_data, encoding=encoding)
    elif args.cdr=="cdr123":
        data_tensor = make_tensor_CDR123(pep_data, encoding=encoding)

    for t in range(1,6):
        for v in range(1,6):
            if t!=v:
                eval_net = torch.load(NETTCR+'out/'+str(args.cdr)+'_pep/'+pep+'/checkpoint_t.'+str(t)+'.v.'+str(v)+".pt", 
                    map_location=torch.device('cpu'))
                eval_net.eval()

                if args.cdr=="cdr3":
                    a3 = data_tensor[:][0]
                    b3 = data_tensor[:][1]
                    avg_pred += eval_net(a3, b3)
                if args.cdr=="cdr123":
                    a1 = data_tensor[:][0]
                    a2 = data_tensor[:][1]
                    a3 = data_tensor[:][2]
                    b1 = data_tensor[:][3]
                    b2 = data_tensor[:][4]
                    b3 = data_tensor[:][5]
                    avg_pred += eval_net(a1, a2, a3, b1, b2, b3)
    cat_df['prediction'] = np.squeeze(avg_pred.cpu().detach().numpy())/20
    cat_df['peptide'] = pep



    pred_df = pred_df.append(cat_df)

end = perf_counter()
print(f'Finished predictions in {end - start}', file=sys.stderr)

# pred_df = pd.concat([data, pd.Series(np.ravel(avg_pred)/20,name='prediction')],axis=1)
#pred_time = time.time()-start_time
#print("Prediction time",pred_time)

# Calculate percentile rank
percent_rank = []
for i in range(pred_df.shape[0]):
    pep = pred_df.iloc[i].peptide
    pred = pred_df.iloc[i].prediction
    percent_rank.append((covid_pred_lists[pep]>pred).mean())
    
pred_df["percentile_rank"] = pd.Series(percent_rank)
rank_time = time.time()-start_time
#print("Percent ranl time", rank_time)

## Here I am saving the outputs to NetTCR-2.0 tmp dir because I couldn't on NetTCR-2.1 tmp
tempdir = os.path.join('/net/sund-nas.win.dtu.dk/storage/services/www/html/services/NetTCR-2.0/tmp', args.jobid)

os.mkdir(tempdir)
pred_df.to_csv(tempdir+'/nettcr_predictions.csv', index=False, sep=',')
print('Click '+'<a href="https://services.healthtech.dtu.dk/services/NetTCR-2.0/tmp/'+args.jobid+
    '/nettcr_predictions.csv" target="_blank">here</a>'+' to download the results in .csv format.')

#print(pred_df, file=sys.stdout)
time_elapsed = time.time()-start_time
print('Total time elapsed: {:.1f} s '.format(time_elapsed))
#print('Total time elapsed (without import): {:.1f} s '.format(time.time()-s_time))

print("\n \nBelow is a table represention of binding predictions between T-Cell receptor and peptides. \n \n")

print(pred_df)