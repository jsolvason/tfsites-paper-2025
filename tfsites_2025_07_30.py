# import packages 
import pandas as pd
import numpy as np
import csv
import re
import os
import subprocess
import math
import itertools
import pickle
# import random
from Bio import SeqIO, motifs
from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, RegularPolygon
from matplotlib.font_manager import FontProperties
import matplotlib.table 
from matplotlib.colors import is_color_like
import matplotlib.colors as mcolors
# import pymol2

###### additional compare seqs packages ######
# from line_profiler import LineProfiler
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from jinja2 import Template
import markdown
import textwrap
import warnings
import psutil
import base64
from pandas.api.types import is_integer_dtype, is_string_dtype, is_numeric_dtype


##############################################################################################################
# general helper functions
##############################################################################################################

# iupac definitions
iupac2allnt= {
        'A':['A'],
        'C':['C'],
        'G':['G'],
        'T':['T'],
        'R':['A','G'],
        'Y':['C','T'],
        'S':['G','C'],
        'W':['A','T'],
        'K':['G','T'],
        'M':['A','C'],
        'B':['C','G','T'],
        'D':['A','G','T'],
        'H':['A','C','T'],
        'V':['A','C','G'],
        'N':['A','C','G','T'],
}

# convert iupac to regex for matching TFBS 
def iupac_to_regex_pattern(seq): 
    p=''
    for iupac in seq:
        if iupac in ['A','T','G','C']:
            p+=iupac
        else:
            p+='('
            for nt in iupac2allnt[iupac]:
                p+=nt
                p+='|'
            p=p[:-1]
            p+=')'
    return p 

# create possible kmers matching an iupac definition
def iupac_to_all_possible_sequences(dna):
	Round2Seqs={}
	Round2Seqs[0]=[]
	for nt in iupac2allnt[dna[0]]:
		Round2Seqs[0].append(nt)
		
#     print(Round2Seqs)
	for i,iupac in enumerate(dna):
		if i==0: continue
			
		Round2Seqs[i]=[]
		
		for seq in Round2Seqs[i-1]:
			for nt in iupac2allnt[iupac]:
#                 print(nt)
#                 print(seq)
				thisSeq=seq+nt
				Round2Seqs[i].append(thisSeq)
				
	lastRound=max(Round2Seqs.keys())
	return Round2Seqs[lastRound]
    
# reverse complement of sequence, including iupac 
def rev_comp(seq): 
    comp = ''
    base_pairs = {'A':'T', 'G':'C', 'T':'A', 'C':'G', 'N':'N', 'K':'M', 'M':'K', 'S':'S', 'W':'W',
                 'R':'Y', 'Y':'R', 'B':'V', 'V':'B', 'D':'H', 'H':'D'}
    for i in seq:
        comp += base_pairs[i]
    return comp[::-1]

# confirm that genomic coordinates are correct 
def check_pos_in_genome(chrom,pos,ref,chr2seq):
    if chr2seq[chrom][pos]==ref: 
        return True
    else: 
        return False #f'Reference nt {ref} not detected in genome at {chrom}:{pos}'
    
# check whether input is true or false (command-line friendly)
def check_bool(input):
    if str(input).upper() == 'TRUE':
        return True
    elif str(input).upper() == 'FALSE':
        return False
    else:
        raise ValueError('Check True/False input values')

# make sure only ATCG are in seq 
def check_nt_ACGT(seq):
    seq = seq.upper()
    for i in seq: 
        if i not in ['A', 'C', 'G', 'T']:
            raise ValueError('Warning: Sequence must only contain A, C, G, or T')

# make sure only ATCG are in seq 
def check_nt_iupac(seq):
    seq = seq.upper()
    for i in seq: 
        if i not in iupac2allnt:
            raise ValueError('Warning: Sequence must only contain valid IUPAC letters')
    return seq

# make sure colors are valid in matplotlib
def check_color(color):
    if not is_color_like(color):
        raise ValueError('Warning: Color must be a valid named color in matplotlib')

# load pickled genome file 
def pickleToDict(file):
    with open(file,'rb') as f: 
        chr2seq=pickle.load(f)
    return chr2seq

# convert tsv with affinity data to dictionary 
def loadNormalizedFile(file): 
    with open(file) as file:
        
        # ignore header 
        file.readline()
        
        seq2aff = {}
        for line in file:
            line = line.strip().split('\t')
            if len(line) == 1:
                seq2aff[line[0]] = None
            else:
                seq2aff[line[0]] = float(line[1].strip('\n'))

    return seq2aff

# scale 
def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

# kmer length
def get_kmer_length(tf_iupac, tf_ref_data_type, tf_ref_data):

    if (tf_ref_data_type == 'Score'):
        return len(tf_ref_data[0]) # for pwm data
    else:
        return len(tf_iupac)


##############################################################################################################
# PWM scoring helper functions used for all functions (visualize, in silico, compare seqs, etc)
##############################################################################################################

seqDict = {'A':0, 'C':1, 'G':2, 'T': 3}
background_Freq= [0.25, 0.25, 0.25, 0.25]

def all_funcs_get_max_score_of_pwm(input_pwm):
    '''
    all_funcs_get_max_score_of_pwm takes an input pwm that is the output of get_PWM_from_PFM
    
    Returns: returns the max score of the input pwm
    '''
    
    length = len(input_pwm[0])
    maxScore = 0
    for col in range(input_pwm.shape[1]):
        maxScore += np.max(input_pwm[:, col])
        
    return maxScore

def all_funcs_get_min_score_of_pwm(input_pwm):
    '''
    Returns the min score of the input pwm that excludes negative infinity
    '''
    minScore = 0
    if np.isneginf(input_pwm).any() == True:
        for col in range(input_pwm.shape[1]):
            col_values = input_pwm[:, col]
            col_wout_neginf = col_values[col_values != -np.inf]
            minScore += np.min(col_wout_neginf)
    else:
        for col in range(input_pwm.shape[1]):
            minScore += np.min(input_pwm[:, col])
    return minScore

def all_funcs_get_score_of_kmer(input_pwm, kmer, maxScore, minScore):
    '''
    get_pfm_python_object takes an input pfm and outputs a pythonic object to score kmers
    
    Parameters:
    - pfm_object - pfm obtained from get_pfm_python_object
    - kmer - user input 
    - maxScore - max score obtained from all_funcs_get_max_score_of_pwm

    Returns: return relative score of the inputted kmer
    '''
    
    kmer_score = 0
    newMax = maxScore+abs(minScore)

    for i, nuc in enumerate(kmer):
        nucPos = seqDict[nuc]
        newscore = input_pwm[nucPos][i]
        if np.isneginf(newscore):
            return 0
        kmer_score+=newscore
    kmer_score+=abs(minScore)
    relative_score = float(kmer_score/newMax)
    return round(relative_score, 2)

def all_funcs_get_pwm_length(input_single_pwm):
    return len(input_single_pwm[0])


##############################################################################################################
# helper functions used for visualize and in silico
##############################################################################################################

def pwm_to_pwmobj_batch(pwm_input, pwm_file_format='jaspar'):
    #with open(pwm_input) as handle:
    
    # for record in motif_s:
    #     pwmObj[(record.matrix_id,record.name)] = np.array([record.counts[nuc] for nuc in 'ACGT'])
    pwmObj = {}
    pwmObj = motif_parser(pwm_input, pwm_file_format=pwm_file_format)

    return pwmObj

def pwm_to_pwmobj(pwm_input, pwm_file_format='jaspar'):
    # with open(pwm_input) as handle:
    #     motif = motifs.parse(handle, 'jaspar')
    #     pwm = np.array([motif[0].counts[nuc] for nuc in 'ACGT'])
    motif = motif_parser(pwm_input, pwm_file_format=pwm_file_format)
    pwm = np.array(list(motif.values())[0])
    
    return pwm


##############################################################################################################
# 01 - Normalize TF-DNA Affinity Data
##############################################################################################################

def normalizeTfDnaAffinityData(raw_aff_data, 
                               column_of_kmers, 
                               column_of_values, 
                               header_present, 
                               binding_site_definition, 
                               define_highest_relative_affinity_sequence=None, 
                               report_sites_only=False, 
                               # enforce_minimum_relative_affinity=False, #no longer being used 
                               plot_resolution=200,
                               output_svg=False, 
                               output_name=None,
                               out_directory='./'):
                            #    reference_relative_affinity_table=None, 
                            #    histograms_of_relative_affinities=None): 

    '''
    Generate a relative affinity dataset to score binding sites. \n
    Parameters: \n
    - raw_aff_data (.txt): File containing the raw affinity dataset. \n
        - columns (strict order, flexible names) -> sequence, value \n
    - column_of_kmers (integer): Number of the column containing the DNA sequences in the input file (1-indexed, 1 is the first column). \n
    - column_of_values (integer): Number of the column containing the raw affinity values in the input file (1-indexed, 1 is the first column). \n
    - header_present (boolean): If True, a header exists in the input file. If False, no header exists. \n
    - binding_site_definition (string): IUPAC definition of the core transcription factor binding site. \n
    - define_highest_relative_affinity_sequence (string, default = None): The k-mer sequence whose value will be used to normalize the values of all other k-mers. The relative affinity for this k-mer will be 1.0. \n
    - report_sites_only (boolean, default = False): If True, only report k-mers abiding by the binding site definition. If False, report all k-mers. \n
    - out_directory (string, default = ./): Directory to contain all output files. \n
    '''

    # check that inputs are valid
    check_nt_iupac(binding_site_definition)
    if type(column_of_kmers) != int: raise ValueError('Column number for kmers must be a valid integer.')
    if type(column_of_values) != int: raise ValueError('Column number for values must be a valid integer.')
    header = check_bool(header_present)
    report_sites_only = check_bool(report_sites_only)
    output_svg = check_bool(output_svg)
    if define_highest_relative_affinity_sequence is not None: 
        check_nt_ACGT(define_highest_relative_affinity_sequence)
        if len(binding_site_definition) != len(define_highest_relative_affinity_sequence):
            raise ValueError('binding_site_definition and define_highest_relative_affinity_sequence must be the same length.')

    # set internal variable names
    iupac = binding_site_definition
    fwd_col_1idx = column_of_kmers
    mfi_col_1idx = column_of_values
    max_kmer = define_highest_relative_affinity_sequence
    # out_file = reference_relative_affinity_table
    # out_image = histograms_of_relative_affinities

    # check that the given max_kmer has correct length (matches iupac)
    if max_kmer is not None:
        if len(max_kmer) != len(iupac):
            raise ValueError('Warning: ' + max_kmer + ' does not match length of given binding site definition')
    
    # load in data 
    df = pd.read_csv(raw_aff_data, sep='\t', header=None)
    
    # check for header 
    if header:
        df = df.iloc[1:]
    
    # change to zero-based indexing 
    fwd_col = int(fwd_col_1idx) - 1
    mfi_col = int(mfi_col_1idx) - 1

    # calculate reverse comp sequences 
    df[fwd_col]=df[fwd_col].str.upper()
    df[fwd_col].apply(check_nt_ACGT)
    df['rev_seq'] = df[fwd_col].apply(rev_comp)
    
    # create final dataframe for seq + signal 
    final_df = pd.DataFrame()
    final_df['Kmer'] = pd.concat([df[fwd_col], df['rev_seq']])
    final_df['Relative Affinity'] = pd.concat([df[mfi_col], df[mfi_col]])
    final_df.drop_duplicates(keep='first', inplace=True) # account for palindromes

    # check if binding site definition matches kmer length in file
    first_kmer = final_df['Kmer'].iloc[0]
    if len(first_kmer) != len(binding_site_definition):
        raise ValueError(f'Warning: length of binding site definition does not match sequences in affinity file')
    
    # convert aff values to float
    final_df['Relative Affinity'] = final_df['Relative Affinity'].apply(lambda x: float(x))
    
    # calculate max_kmer and its signal 
    final_df['iupac'] = final_df['Kmer'].apply(lambda seq: 1 if (re.search(iupac_to_regex_pattern(iupac), seq) or re.search(iupac_to_regex_pattern(rev_comp(iupac)), seq)) else 0)
    calc_signal = final_df.loc[final_df['iupac'] == 1, 'Relative Affinity'].max()
    calc_max_kmer = (final_df['Kmer'].loc[final_df['Relative Affinity'] == calc_signal]).iloc[0]
    
    # if max_kmer is given, use it but give warning if there are iupac kmers with higher affinity 
    if max_kmer is not None:
        max_kmer = max_kmer.upper()
        user_max_kmer_signal = (final_df.loc[final_df['Kmer'] == max_kmer, 'Relative Affinity']).iloc[0]
        greater_aff = list(final_df.loc[(final_df['iupac'] == 1) & (final_df['Relative Affinity'] > user_max_kmer_signal) , 'Kmer']) 
        greater_aff_len = len(greater_aff)

        if greater_aff_len != 0:
            print(f'Warning: The following {greater_aff_len} k-mers that follow the binding site definition have a higher affinity than the define_highest_relative_affinity_sequence: ' + ', '.join(greater_aff))   
        calc_signal = user_max_kmer_signal
        calc_max_kmer = max_kmer 

    # give warning if there are non-iupac kmers with higher affinity than the given max kmer (calculated or given)
    non_iupac = list(final_df.loc[(final_df['iupac'] == 0) & (final_df['Relative Affinity'] > calc_signal), 'Kmer'])
    non_iupac_len = len(non_iupac)
    if non_iupac_len != 0:
        if max_kmer is not None:
            print(f'Warning: The following {non_iupac_len} k-mers that do not follow the binding site definition have a higher affinity than define_highest_relative_affinity_sequence: ' + ', '.join(non_iupac))
        else:
            print(f'Warning: The following {non_iupac_len} k-mers that do not follow the binding site definition have a higher affinity than the k-mer calculated to have the highest relative affinity: ' + ', '.join(non_iupac))
        
    # calculate relative affinities - round and maintain trailing zero float(f'{x:.2f}')
    aff_list = final_df['Relative Affinity'].tolist()
    for i,aff in enumerate(aff_list): 
        norm_aff = aff / calc_signal
        aff_list[i] = float(f'{norm_aff:.2f}') 
    final_df['Relative Affinity'] = aff_list

    # cap affinities at 1.0
    final_df.loc[final_df['Relative Affinity'] > 1, 'Relative Affinity'] = 1.0

    # # report min and max aff sequences
    # min_aff = final_df['rel_aff'].min()
    # print(f'Note: affinity values range between {min_aff} and 1.0')

    # # min normalization  (xi – min(x)) / (max(x) – min(x))
    # if min_norm:
    #     min_aff = final_df['Relative Affinity'].min()
    #     final_df['Relative Affinity'] = round((final_df['Relative Affinity'] - min_aff) / (1 - min_aff), 3)

    # create iupac and non-iupac dataframes from iupac col 
    iupac_df = final_df[final_df['iupac'] == 1]
    non_iupac_df = final_df[final_df['iupac'] == 0]
        
    # write output to csv file - report only iupac k-mers or all k-mers
    if report_sites_only:
        kmer2aff = {kmer:aff for kmer,aff in zip(iupac_df['Kmer'],iupac_df['Relative Affinity'])}
        out_df = pd.DataFrame()
        out_df['Kmer'] = iupac_to_all_possible_sequences('N'*len(iupac))
        rel_aff_list = []
        for kmer in out_df['Kmer']:
            if kmer in kmer2aff:
                rel_aff_list.append(kmer2aff[kmer])
            else:
                rel_aff_list.append(None)
        out_df['Relative Affinity'] = rel_aff_list
    else:
        kmer2aff = {kmer:aff for kmer,aff in zip(final_df['Kmer'],final_df['Relative Affinity'])}
        out_df = pd.DataFrame()
        out_df['Kmer'] = iupac_to_all_possible_sequences('N'*len(iupac))
        rel_aff_list = []
        for kmer in out_df['Kmer']:
            if kmer in kmer2aff:
                rel_aff_list.append(kmer2aff[kmer])
            else:
                rel_aff_list.append(None)
        out_df['Relative Affinity'] = rel_aff_list
    out_df.reset_index(drop=True, inplace=True)

    # # plot histograms
    # fig, ax = plt.subplots(3,1, figsize=(7,10)) 
    # ax[0].hist(final_df['Relative Affinity'], bins=50) 
    # ax[0].set_title('All DNA Sequences')
    # ax[0].set_xlabel("Relative Affinity")
    # ax[0].set_ylabel("Frequency") 

    # ax[1].hist(iupac_df['Relative Affinity'], bins=50) 
    # ax[1].set_title('DNA Sequences Consistent With TF Site Definition')
    # ax[1].set_xlabel("Relative Affinity")
    # ax[1].set_ylabel("Frequency") 

    # ax[2].hist(non_iupac_df['Relative Affinity'], bins=50)
    # ax[2].set_title('DNA Sequences Not Consistent With TF Site Definition')
    # ax[2].set_xlabel("Relative Affinity")
    # ax[2].set_ylabel("Frequency") 
    # fig.tight_layout(pad=2.0) 

    # only plot dna sequences that follow iupac
    fig, ax = plt.subplots(dpi=plot_resolution) 
    ax.hist(iupac_df['Relative Affinity'], bins=50) 
    ax.set_title('DNA Sequences Consistent With TF Site Definition')
    ax.set_xlabel("Relative Affinity")
    ax.set_ylabel("Frequency") 
    fig.tight_layout(pad=2.0) 

    # get names for output file and image 
    if out_directory != './':
        out_directory = out_directory.strip('/') # get rid of trailing slashes
        if output_name is None:
            out_image = out_directory + '/' + f'relative-aff-histogram_site={binding_site_definition}_max={calc_max_kmer}.png'
            out_file = out_directory + '/' + f'relative-aff-table_site={binding_site_definition}_max={calc_max_kmer}.tsv'
        else:
            output_name = output_name.split('.')[0] #remove file extensions just in case
            if output_svg:
                out_image = out_directory + '/' + output_name + '_histogram.svg'
            else:
                out_image = out_directory + '/' + output_name + '_histogram.png'
            out_file = out_directory + '/' + output_name + '_table.tsv'
    
    else:
        if output_name is None:
            out_image = f'relative-aff-histogram_site={binding_site_definition}_max={calc_max_kmer}.png'
            out_file = f'relative-aff-table_site={binding_site_definition}_max={calc_max_kmer}.tsv'
        else:
            output_name = output_name.split('.')[0] #remove file extensions just in case
            if output_svg:
                out_image = output_name + '_histogram.svg'
            else:
                out_image = output_name + '_histogram.png'
            out_file = output_name + '_table.tsv'

    # output image
    plt.savefig(out_image, dpi=plot_resolution)
    print(out_image + ' has been created')

    # output file 
    out_df.to_csv(out_file, sep="\t", index=None)
    print(out_file + ' has been created')

    # close plot
    plt.close('all')

    # # output seq2aff
    # seq2aff = dict(zip(out_df['seq'], out_df['rel_aff']))
    
    # return out_df


##############################################################################################################
# 02 - Download PWMs
##############################################################################################################
# Helper Function List
#    - read_in_chunks
#    - keywords_tsv_to_list
#    - calculate_pseudocounts
#    - PFMtoPWM
##############################################################################################################

def gdb_read_in_chunks(file_path, chunk_size=4):
    """
    Generator to read a file in chunks of lines.
    
    Parameters:
    file_path (str): Path to the file to be read.
    chunk_size (int): Number of lines to read at a time (default is 4).
    
    Yields:
    list: A list of lines.
    """
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
            
def gdb_keywords_tsv_to_list(keywords_file):
    keywords_list = []
    with open(keywords_file) as file:
        file.readline()
        for line in file:
            keyword = line.strip()
            keywords_list.append(keyword)
    return keywords_list

def gdb_calculate_pseudocounts(matrix_values, k_len):
    '''
    Calculate pseudocounts using the product of square root of the 
    column averages and the nucleotide's background frequency.
    return a dictionary of pseudocounts as {'A':0.0,'C':0.0, 'G':0.0,'T':0.0}
    '''
    total=0
    for row in matrix_values:
        total+=sum(row)
    ave_countsPcol = total/k_len
    sqrt_ave = math.sqrt(ave_countsPcol)
    pseudocounts = {}
    for idx, nuc in enumerate('ACGT'):
        pseudocounts[nuc] = background_Freq[idx]*sqrt_ave
    return pseudocounts

def gdb_PFMtoPWM(matrix_values, pseudocounts, background_frequencies):

    # matrix length
    k_len = len(matrix_values[0])

    # calculate pseudocounts
    if pseudocounts:
        pseudoct = gdb_calculate_pseudocounts(matrix_values, k_len)
    else:
        pseudoct = {'A':0.0, 'C':0.0, 'G':0.0, 'T':0.0}

    # get counts matrix
    counts_matrix = np.array(matrix_values, dtype='float64')
    counts_matrix += np.array([pseudoct[nuc] for nuc in 'ACGT']).reshape(-1,1)
    totalOb = counts_matrix.sum(axis=0)
    pfm = counts_matrix/totalOb

    # convert PFM to PWM
    background_Freq = [freq for nt,freq in background_frequencies.items()]
    bgFreq=np.array(background_Freq)
    pwm = np.log2(np.divide(pfm, bgFreq[:, np.newaxis]))
    
    return pwm

def generateMotifDatabase(input, 
                          input_format,
                          keywords_file=None, 
                          pseudocounts=False, 
                          background_frequencies={'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25}, 
                          output_name=None, 
                          out_directory='./'):
    
    pseudocounts = check_bool(pseudocounts)

    ##### account for keywords #####
    if keywords_file is not None:

        # extract list of keywords
        keywords_list = gdb_keywords_tsv_to_list(keywords_file)

        # only add pfm if it follows keyword list
        line_out=''
        for chunk in gdb_read_in_chunks(input, chunk_size=5):
            header=chunk[0]
            for ki in keywords_list:
                if ki.upper() in header.upper():
                    line_out+='\n'.join(chunk)+'\n'
    else:

        # add every line regardless
        line_out=''
        for chunk in gdb_read_in_chunks(input, chunk_size=5):
            line_out+='\n'.join(chunk)+'\n'

    ###### get output file name ######
    if output_name is None:
        base_file = input.split('/')[-1]
        output_name = 'output_PWM-from-' + base_file
    else:
        output_name = output_name.split('.')[0] + '.txt' #just in case there is another extension provided by user
    
    ###### add path if necessary ######
    if out_directory is not None:
        output_name = out_directory.strip('/') + '/' + output_name
    
    ###### convert to pwms #####
    if input_format == 'pfm':
        with open(output_name, 'w') as file:
            matrix_list = line_out.split('>')
            for matrix in matrix_list[1:]: #ignore first empty line
                rows = matrix.split('\n')
                header_info = rows[0]
                # print(header_info)
                file.write(f'>{header_info}' + '\n')
                matrix_values = []
                for row in rows[1:]:
                    if len(row) != 0:
                        row_values = [int(val) for val in row.replace('[', '').strip(']').split()[1:]]
                        matrix_values.append(row_values)
                pwm = gdb_PFMtoPWM(matrix_values, pseudocounts, background_frequencies)
                for nt,row in zip(['A','C','G','T'], pwm):
                    row_list = [str(i) for i in list(row)]
                    new_line = f'{nt} [ ' + ' '.join(row_list) + ' ]'
                    file.write(new_line + '\n')
                # break
        print(f'{output_name} has been created')
    elif input_format == 'pwm':
        with open(output_name, 'w') as file:
            for line in line_out:
                file.write(line)
        print(f'{output_name} has been created')
    else:
        raise ValueError('the value of input_format must be either "pfm" or "pwm"')



###########################################################################################################################
# 03/04 - helper functions shared between annotateAndVisualizeTfSites and annotateAndVisualizeInSilicoSnvs
#     - singleSeqAnnotateTfSites: return table of binding site annotations
#     - plot_sites: plot only the binding sites onto existing plot
###########################################################################################################################

########################################################*
# singleSeqAnnotateTfSites 
########################################################*
# helper function list:
#    - preprocess_seq
#    - load_reference_data
#    - get_kmer_length
#    - check_kmers
#    - assign_existing_kmer_ids
#    - check_kmer_duplicates
#    - apply_zoom_for_sites_df
########################################################*

# preprocess sequence
def preprocess_seq(seq):

    # make seq uppercase
    seq = seq.upper() 

    # check that the seq only contains valid nucleotides ACGT
    check_nt_ACGT(seq) 

    return seq 

# get ref data - either pwm, pbm, or handle none
def load_reference_data(tf_ref_data, tf_ref_data_type):

    if tf_ref_data_type == 'Affinity':
        pwm = None
        seq2aff = tf_ref_data
    elif tf_ref_data_type == 'Score':
        pwm = tf_ref_data
        seq2aff = None
    else:
        pwm, seq2aff = None, None

    return pwm, seq2aff

def check_kmers(seq, tf_kmer_length, tf_iupac, tf_ref_data_type, tf_min_affinity, tf_min_score, seq2aff, pwm):
    
    # define output lists
    kmer_list, start_pos_list, end_pos_list, ref_data_type_list, value_list, site_dir_list = [], [], [], [], [], []

    # if using pwm data, get min and max scores to score kmers
    if pwm is not None:
        max_score = all_funcs_get_max_score_of_pwm(pwm)
        min_score = all_funcs_get_min_score_of_pwm(pwm)

    # if using iupac, get iupac objects 
    if tf_iupac is not None: 
        rev_tf_iupac = rev_comp(tf_iupac)
        rev_iupac_reobj = iupac_to_regex_pattern(rev_tf_iupac)
        fwd_iupac_reobj = iupac_to_regex_pattern(tf_iupac)
    
    # iterate through seq to find all kmers
    for i in range(len(seq) - tf_kmer_length + 1):

        # get pos and kmer seq
        pos_1idx = i + 1
        kmer = seq[i:i+tf_kmer_length] 

        # cases 1, 2, 3: check if an iupac exists 
        if tf_iupac is not None:
    
            # check if we have fwd, rev, or palidrome site
            check_fwd_site = re.search(fwd_iupac_reobj, kmer)
            check_rev_site = re.search(rev_iupac_reobj, kmer)
            check_palindrome = re.search(fwd_iupac_reobj, kmer) and (tf_iupac == rev_tf_iupac)
    
            ### obtain site direction, start pos, and end pos ###
            # check if we have palindrome
            if check_palindrome:
                site_direction = ''
                start_pos_1idx = pos_1idx
                end_pos_1idx = pos_1idx + tf_kmer_length - 1 # subtract 1 to account for converting 0-indexing back to 1-indexing
                
            # check if we have a forward site
            elif check_fwd_site:
                site_direction = '+'
                start_pos_1idx = pos_1idx
                end_pos_1idx = pos_1idx + tf_kmer_length - 1 # subtract 1 to account for converting 0-indexing back to 1-indexing
        
            # check if we have reverse site
            elif check_rev_site:
                site_direction = '-'
                start_pos_1idx = pos_1idx + tf_kmer_length - 1 # subtract 1 to account for converting 0-indexing back to 1-indexing
                end_pos_1idx = pos_1idx

            # if the kmer does not follow the iupac, skip and check the next one
            else:
                continue 

            ### start checking cases ###
            # case 1: no aff or score
            if (seq2aff is None) and (pwm is None):
                # ref_data_type = ''
                value = ''
    
            # case 2: get affinity
            elif seq2aff is not None: 
                # ref_data_type = 'Affinity'
                value = seq2aff[kmer] # get pbm score from loaded seq2aff dictionary 
                if (value < tf_min_affinity):
                    continue # if the score does not meet the minaff, then we don't plot it
    
            # case 3: get score
            elif pwm is not None: 
                # ref_data_type = 'Score'
                value = all_funcs_get_score_of_kmer(pwm, kmer, max_score, min_score) # score kmer using joe's pwm function
        
        # case 4: if iupac is none, then pfm file must exist and we check the score threshold
        else:

            # get forward and rev kmer
            fwd_kmer_score = all_funcs_get_score_of_kmer(pwm, kmer, max_score, min_score)
            rev_kmer = rev_comp(kmer)
            rev_kmer_score = all_funcs_get_score_of_kmer(pwm, rev_kmer, max_score, min_score)
    
            # score the forward kmer and only add site info to df if it is above the threshold
            if fwd_kmer_score >= tf_min_score: 
                start_pos_1idx = pos_1idx 
                end_pos_1idx = pos_1idx + tf_kmer_length - 1 # subtract 1 to account for converting 0-indexing back to 1-indexing
                # ref_data_type = 'Score'
                value = fwd_kmer_score
                site_direction = '+'
    
            # score the reverse kmer and only add site info to df if it is above the threshold
            elif rev_kmer_score >= tf_min_score: 
                kmer = rev_kmer
                start_pos_1idx = pos_1idx + tf_kmer_length - 1 # subtract 1 to account for converting 0-indexing back to 1-indexing
                end_pos_1idx = pos_1idx
                # ref_data_type = 'Score'
                value = rev_kmer_score
                site_direction = '-'

            else:
                continue

        # add site to dataframe lists
        kmer_list.append(kmer)
        start_pos_list.append(int(start_pos_1idx))
        end_pos_list.append(int(end_pos_1idx))
        ref_data_type_list.append(tf_ref_data_type)
        value_list.append(value)
        site_dir_list.append(site_direction)

    return kmer_list, start_pos_list, end_pos_list, ref_data_type_list, value_list, site_dir_list

def assign_existing_kmer_ids(df, tf_name):

    # extract tf name and matrix id (if using pwm -> check by seeing if tfac includes matrix id); also create kmer id 
    if (tf_name[:2] == 'MA') and (tf_name[-2] == '.'):
        matrix_id = tf_name.split('_')[0]
        print(tf_name)
        tf_name = tf_name.split('_')[1].upper()
        df.insert(1, 'TF Name', [tf_name for i in range(len(df))])
        df.insert(2, 'Matrix ID', [matrix_id for i in range(len(df))])
        df.insert(3, 'Kmer ID', [tf_name + ':' + matrix_id + ':' + str(i+1) for i in range(len(df))])
    else:
        tf_name = tf_name.upper()
        df.insert(1, 'TF Name', [tf_name for i in range(len(df))])
        df.insert(2, 'Kmer ID', [tf_name + ':' + str(i+1) for i in range(len(df))])

    return df

def check_kmer_duplicates(df):
    
    # create dict mapping kmer -> kmer_id to see which kmers have duplicates, if any
    kmerseq2id={} 
    for kmerseq,kmerid in zip(df['Kmer'],df['Kmer ID']):
        if kmerseq not in kmerseq2id:
            kmerseq2id[kmerseq]=set()
        kmerseq2id[kmerseq].add(kmerid)

    # extract kmer_ids for duplicate kmers
    kmerDupCol=[]
    for kmerseq in df['Kmer']:
        id_list = list(kmerseq2id[kmerseq])
        if len(id_list) == 1:
            kmerDupCol.append('')
        else:
            kmerDupCol.append(','.join(id_list))
    df['Duplicate Kmer IDs'] = kmerDupCol

    return df 

def apply_zoom_for_sites_df(df, zoom_range):

    lower_bound = zoom_range[0] 
    upper_bound = zoom_range[1]

    # subset fwd direction sites
    pos_strands = (df['Site Direction'] == '+') | (df['Site Direction'] == '')
    pos_strands_bound = (df['Start Position (1-indexed)'] < upper_bound) & (df['End Position (1-indexed)'] > lower_bound)
    pos_strands_plotted = (pos_strands) & (pos_strands_bound)

    # subset rev direction sites
    neg_strands = (df['Site Direction'] == '-')
    neg_strands_bound = (df['End Position (1-indexed)'] < upper_bound) & (df['Start Position (1-indexed)'] > lower_bound)
    neg_strands_plotted = (neg_strands) & (neg_strands_bound)
    
    df = df.loc[pos_strands_plotted | neg_strands_plotted,:]
        
    return df

def singleSeqAnnotateTfSites(seq, 
                             seq_name, 
                             tf_name, 
                             tf_kmer_length,
                             tf_ref_data_type=None,
                             tf_ref_data=None, 
                             tf_iupac=None, 
                             tf_min_affinity=0,
                             tf_min_score=0.7, 
                             tf_pseudocounts=False, 
                             zoom_range=None, 
                             out_file=None, 
                             out_directory='./'): 

    '''
    Given an entire sequence (no zoom or window range applied yet), find all binding sites that exist. 

    Features:
    - Account for both forward, reverse, and palindrome binding sites
    - Allow for 4 possible cases:
        (1) no reference data + iupac given -> no score or affinity assigned
        (2) PBM (or other affinity dataset) ref data + iupac given -> affinity assigned
        (3) PWM ref data + iupac given -> score assigned 
        (4) PWM ref data + NO iupac given -> score assigned according to threshold value (default threshold = 0.7) 
    '''

    # preprocess sequence
    seq = preprocess_seq(seq)

    # get ref data 
    pwm, seq2aff = load_reference_data(tf_ref_data, tf_ref_data_type)
    
    # iterate through seq to find all kmers 
    kmer_list, start_pos_list, end_pos_list, ref_data_type_list, value_list, site_dir_list = check_kmers(seq = seq, 
                                                                                                         tf_kmer_length = tf_kmer_length, 
                                                                                                         tf_iupac = tf_iupac, 
                                                                                                         tf_ref_data_type = tf_ref_data_type, 
                                                                                                         tf_min_affinity = tf_min_affinity, 
                                                                                                         tf_min_score = tf_min_score, 
                                                                                                         seq2aff = seq2aff, 
                                                                                                         pwm = pwm)
    # create output dataframe
    df = pd.DataFrame({
        'Kmer': kmer_list,
        'Start Position (1-indexed)': start_pos_list,
        'End Position (1-indexed)': end_pos_list,
        'Ref Data Type': ref_data_type_list,
        'Value': value_list,
        'Site Direction': site_dir_list})

    # apply zoom boundaries before adding extra modifications/features, depending on site direction
    if zoom_range:
        df = apply_zoom_for_sites_df(df = df, 
                                     zoom_range = zoom_range)

    # assign seq name
    df.insert(0, 'Sequence Name', [seq_name for i in range(len(df))])

    # assign kmer ids
    df = assign_existing_kmer_ids(df = df, 
                                  tf_name = tf_name)

    # add col to check for duplicate kmer ids
    df = check_kmer_duplicates(df)

    # convert pos cols to int
    df['Start Position (1-indexed)'] = df['Start Position (1-indexed)'].astype(int)
    df['End Position (1-indexed)'] = df['End Position (1-indexed)'].astype(int)

    # # output to tsv 
    # if out_file != None:
    #     if out_directory != './':
    #         out_file = out_directory + '/' + out_file
    #     df.to_csv(out_file, sep="\t",index=None)
    #     print(f'{out_file} has been created')
    
    return df

########################################################*
# plot_sites 
########################################################*
# helper function list:
#    - single_kmer_plotting_info
#    - get_plotting_info
#    - get_shift_factor
#    - set_plot_specs
#    - plot_sites_for_single_tf
#    - adjust_plot_size
########################################################*

def single_kmer_plotting_info(og_start_x_pos, og_end_x_pos, site_direction, kmer_length):

    # height of binding sites
    height = 2.4

    # get y pos for polygon, including half height for directionality and half width for the text labels
    y_pos_site = 0
    half_height = (height / 2)
    half_width = (kmer_length - 1) / 2

    # get x pos (depending on dir), coords for polygon, coords for label
    if site_direction == '+':
        x_pos_site = og_start_x_pos - 1 # revert back to 1-indexing
        # print(og_start_x_pos)
        coord = [(x_pos_site,y_pos_site), ((kmer_length-1)+x_pos_site,y_pos_site), ((kmer_length)+x_pos_site,half_height+y_pos_site), ((kmer_length-1)+x_pos_site,height+y_pos_site), (x_pos_site,height+y_pos_site), (x_pos_site,y_pos_site)] 
        x_pos_label = half_width+x_pos_site

    elif site_direction == '-':
        x_pos_site = og_end_x_pos - 1 # revert back to 1-indexing
        # print(og_start_x_pos)
        coord = [((kmer_length)+x_pos_site,y_pos_site), ((kmer_length)+x_pos_site,height+y_pos_site), (1+x_pos_site,height+y_pos_site), (x_pos_site,half_height+y_pos_site), (1+x_pos_site,y_pos_site), ((kmer_length)+x_pos_site,y_pos_site)]  
        x_pos_label = half_width+x_pos_site+1

    elif site_direction == '':
        x_pos_site = og_start_x_pos - 1 # revert back to 1-indexing
        coord = [[x_pos_site,y_pos_site], [kmer_length+x_pos_site,y_pos_site], [kmer_length+x_pos_site,height+y_pos_site], [x_pos_site,height+y_pos_site], [x_pos_site,y_pos_site]] 
        x_pos_label = half_width+x_pos_site+0.5

    return coord, x_pos_site, x_pos_label

# get coordinates for each binding site 
def get_plotting_info(df, start_x, kmer_length, plot_denovo=False):

    # # change start x to be 0-indexed
    # start_x -= 1

    # get cols of interest
    if plot_denovo: 
        col_list = [df['Kmer ID'], df['Alternate Value'], df['Kmer Start Position (1-indexed)'], df['Kmer End Position (1-indexed)'], df['Site Direction']]
    else:
        col_list = [df['Kmer ID'], df['Value'], df['Start Position (1-indexed)'], df['End Position (1-indexed)'], df['Site Direction']]

    # gather plotting info for each site (kmer_id, x pos for site, value, site_direction, x pos for label)  
    plot_list = []
    for kmer_id, value, og_start_x_pos, og_end_x_pos, site_direction in zip(*col_list):

        # get plotting info for single kmer 
        coord, x_pos_site, x_pos_label = single_kmer_plotting_info(og_start_x_pos = og_start_x_pos, 
                                                                   og_end_x_pos = og_end_x_pos, 
                                                                   site_direction = site_direction, 
                                                                   kmer_length = kmer_length)

        # add to running plot list
        plot_list.append((kmer_id, value, coord, x_pos_site, x_pos_label))

    return plot_list

# increase y position to account for overlap
def get_shift_factor(pos2overlap, x_pos, kmer_length):

    # increment each pos for overlap and find max num for shift factor 
    shift_factor_list = []
    for i in range(x_pos, x_pos+kmer_length):
        pos2overlap[i] += 8 # account for height of site and spacing between sites 
        shift_factor_list.append(pos2overlap[i])

    # get max shift factor 
    max_shift_factor = max(shift_factor_list)

    # assign each pos in kmer to new max shift factor
    for i in range(x_pos, x_pos+kmer_length):
        pos2overlap[i] = max_shift_factor 

    return max_shift_factor, pos2overlap

def set_plot_specs(seq_length, plotDims=None, plot_dpi=200, fig=None, ax=None):

    # figure size
    if ax is None: 
        if plotDims is None:
            fig, ax = plt.subplots(1, figsize=(seq_length*.2, 10), dpi=plot_dpi)
            # ax.set_ylim(0,15)
            ax.set_xlim(0,seq_length)
        else: 
            fig, ax = plt.subplots(1, figsize=plotDims, dpi=plot_dpi)

    # remove axes
    for si in ['top','right','left', 'bottom']:
        ax.spines[si].set_visible(False)

    # set ax so proportions do not change
    ax.set_aspect('equal', adjustable='box')
    
    # set x and y labels to none
    ax.set_yticks([])
    ax.set_xticks([])

    return fig,ax

# plot sites for single tf 
def plot_sites_for_single_tf(fig, ax, tfac, tf_color, tf_ref_data_type, tf_kmer_length, tf_plot_list, pos2overlap, reg_or_denovo, row_num, row_num_list):

    # start with no capacity tf
    capacity_tf = None

    # extract information from plot list
    for kmer_id, value, coord, x_pos_site, x_pos_label in tf_plot_list:

        row_num += 1

        # check for overlap
        y_shift_factor, pos2overlap = get_shift_factor(pos2overlap = pos2overlap, 
                                                        x_pos = x_pos_site, 
                                                        kmer_length = tf_kmer_length)
        if y_shift_factor > 140:
            row_num_list.remove(row_num)
            # capacity_tf = tfac
            continue

        # add y shift factor to each y coordinate in the site's coords
        adjusted_coord = []
        for ordered_pair in coord:
            adj_ordered_pair = (ordered_pair[0], ordered_pair[1]+y_shift_factor)
            adjusted_coord.append(adj_ordered_pair)

        # get alpha and affinity label -> depending on if ref data is given (do this for each kmer in case there is missing pbm val)
        # print(value)
        if value == '':
            alph = 1
            value_label = ''
        else:
            alph = value
            if tf_ref_data_type == 'Affinity':
                value_label = f'Aff = {value:.2f}'
            elif tf_ref_data_type == 'Score':
                value_label = f'Score = {value:.2f}'

        # create position label -> get start and end pos of binding site, add 1 to revert to 1-indexed (not for end since we didn't change that)
        start_site_pos = x_pos_site + 1
        end_site_pos = x_pos_site + tf_kmer_length
        range_label = str(start_site_pos) + '-' + str(end_site_pos)

        # set font size
        font_size = 13

        # plot site using ax, instead of plt
        if reg_or_denovo == 'regular':
            polygon = Polygon(adjusted_coord, closed=True, facecolor=tf_color, alpha=alph)
            ax.add_patch(polygon)
        elif reg_or_denovo == 'denovo':
            green_edge_color = mcolors.to_rgba('green', 1)
            denovo_face_color = mcolors.to_rgba(tf_color, alph)
            polygon = Polygon(adjusted_coord, closed=True, edgecolor=green_edge_color, facecolor=denovo_face_color, linewidth=3.5)
            ax.add_patch(polygon)

        # plot labels -> value (aff/score), position range, kmer id
        ax.text(x=x_pos_label, y=y_shift_factor-1.2, s=value_label, fontsize=font_size, horizontalalignment='center', clip_on=True, 
                fontfamily='monospace') #height of text is 1.2, plus 0.3 for extra space
        ax.text(x=x_pos_label, y=y_shift_factor+2.8, s=range_label, fontsize=font_size-0.5, horizontalalignment='center', clip_on=True, 
                fontfamily='monospace') # height of site is 2.4, plus 0.6 for extra space
        ax.text(x=x_pos_label, y=y_shift_factor+4, s=kmer_id, fontsize=font_size, horizontalalignment='center', clip_on=True, 
                fontfamily='monospace') # height of text is 1.2 (so we get 3 + 1.2 = 4.2)

    return fig,ax,pos2overlap,row_num,row_num_list

# adjust height by highest tfbs
def adjust_plot_size(pos2overlap, start_x, end_x, fig, ax):

    # adjust max y axis limit to the highest tfbs
    upper_y = 0
    for pos,overlap in pos2overlap.items():
        if (overlap < 140) and (overlap > upper_y):
            upper_y = overlap
    # upper_y = max(pos2overlap.values())
    # print(pos2overlap)
    ax.set_ylim(-6, upper_y+5.4+1) # one tfbs unit is 5.4 in height, plus 1 for cushion
    # ax.set_ylim(-6, 140)

    # adjust figure size to maintain proportions
    curr_ylim = ax.get_ylim()[1]
    curr_figwidth = fig.get_size_inches()[0]
    new_figheight = (10 * curr_ylim) / 13.4 # first row figheight is y=10 and first row ylim is y=13.4
    fig.set_size_inches(curr_figwidth, new_figheight)

    # cut plot
    ax.set_xlim(start_x,end_x)

    # adjust spacing to make clean
    plt.tight_layout()

    return fig,ax

# main function
def plot_sites(seq, seq_name, start_x, end_x, tf_iupacs, tf_colors, tf_dfs, tf_kmerlens, tf_name_for_snv_analysis=None, snv_df=None,
               plot_denovo=False, plot_dpi=200, fig=None, ax=None):
    
    # create combined df
    all_sites_df = pd.DataFrame()
    for tfac,curr_df in tf_dfs.items():
        if 'Matrix ID' not in curr_df.columns:
            curr_df['Matrix ID'] = ['' for i in range(len(curr_df))]
        all_sites_df = pd.concat([all_sites_df, curr_df], ignore_index=True)
    # print(all_sites_df)
    all_sites_df['Value'] = all_sites_df['Value'].replace('', 100)
    all_sites_df = all_sites_df.sort_values(by='Value', ascending=False) # prioritize plotting sites with higher aff/score
    all_sites_df['Value'] = all_sites_df['Value'].replace(100, '')
    # all_sites_df = all_sites_df[(all_sites_df['Start Position (1-indexed)'] >= start_x) & 
    #                           (all_sites_df['Start Position (1-indexed)'] <= end_x)]

    # get seq length (for zoom range or window size)
    seq_length = end_x - start_x 

    # set plot specs 
    fig, ax = set_plot_specs(seq_length = seq_length, 
                             plotDims = None, 
                             plot_dpi = plot_dpi, 
                             fig = fig, 
                             ax = ax)

    # create dict to account for overlapping sites across all TFs
    pos2overlap = {}
    for pos in range(len(seq)):
        pos2overlap[pos] = -2 # all sites start at -2 because the shift factor is 9 -> we want to start the site at y=7 (-2+9=7)

    # plot sites for each tf
    row_num_list = [i for i in range(len(all_sites_df))]
    row_num = -1
    # pos2capacity = {i:False for i in range(start_x, end_x+1)}
    for tfac, kmer_id, value, og_start_x_pos, og_end_x_pos, \
        site_direction, ref_data_type, matrix_id in zip(all_sites_df['TF Name'],
                                                        all_sites_df['Kmer ID'], 
                                                        all_sites_df['Value'] ,
                                                        all_sites_df['Start Position (1-indexed)'], 
                                                        all_sites_df['End Position (1-indexed)'], 
                                                        all_sites_df['Site Direction'], 
                                                        all_sites_df['Ref Data Type'],
                                                        all_sites_df['Matrix ID']):

        # # adjust tfac for pwm
        # if ref_data_type == 'Score':
        #     print(matrix_id, tfac)
        #     tfac = matrix_id + '_' + tfac

        # get plotting info for single kmer
        coord, x_pos_site, x_pos_label = single_kmer_plotting_info(og_start_x_pos = og_start_x_pos, 
                                                                   og_end_x_pos = og_end_x_pos, 
                                                                   site_direction = site_direction, 
                                                                   kmer_length = tf_kmerlens[tfac])
        plot_list = [[kmer_id, value, coord, x_pos_site, x_pos_label]]
        # print(plot_list)

        # plot single kmer
        fig,ax,pos2overlap,row_num,row_num_list = plot_sites_for_single_tf(fig = fig, 
                                                                            ax = ax, 
                                                                            tfac = tfac, 
                                                                            tf_color = tf_colors[tfac], 
                                                                            tf_ref_data_type = ref_data_type, 
                                                                            tf_kmer_length = tf_kmerlens[tfac],
                                                                            tf_plot_list = plot_list, 
                                                                            pos2overlap = pos2overlap, 
                                                                            reg_or_denovo='regular',
                                                                            row_num = row_num,
                                                                            row_num_list = row_num_list)

        # # check if height capacity has been reached
        # if capacity_tf is not None:
        #     break

    # make new df based on row_num cutoff
    image_sites_df = all_sites_df.iloc[row_num_list].reset_index(drop=True)
    remaining_row_num_list = [i for i in range(len(all_sites_df)) if i not in row_num_list]
    excluded_sites_df = all_sites_df.iloc[remaining_row_num_list].reset_index(drop=True)

    # adjust plot by highest binding site, fix figure proportions
    fig,ax = adjust_plot_size(pos2overlap = pos2overlap, 
                              start_x = start_x, 
                              end_x = end_x, 
                              fig = fig, 
                              ax = ax)

    return fig,ax,image_sites_df,excluded_sites_df,all_sites_df

########################################################*
# append_images_with_top_border: stitch together images 
########################################################*

def append_images_with_top_border(images, outfile):
    """
    Append multiple images horizontally, aligning them by their bottom edges and filling the top of shorter images with white space.

    :param images: List of image paths or PIL Images
    :return: A single image appended from the list of images with an even top border
    """

    # create image objects
    imgs = [Image.open(img) if isinstance(img, str) else img for img in images]

    # Calculate the total width and maximum height
    total_width = sum(img.width for img in imgs)
    max_height = max(img.height for img in imgs)

    # Create a new image with the total width and max height
    new_img = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))

    # Paste images into the new image, aligning the bottom, and fill the top with white space
    x_offset = 0
    for img in imgs:
        y_offset = max_height - img.height
        # Add white space on top of the image to align it with the top border
        padded_img = ImageOps.expand(img, border=(0, y_offset, 0, 0), fill='white')
        new_img.paste(padded_img, (x_offset, 0))
        x_offset += img.width

    # save image to file
    new_img.save(outfile)
    print(outfile + ' has been created')

########################################################*
# extract_tf_info_from_batch_pwm 
########################################################*

def extract_tf_info_from_batch_pwm(batch_custom_pwm, batch_pwm_min_score, batch_pwm_tf_color, pwm_pseudocounts, tf_colors, tf_data, pwm_file_format='jaspar'):

    # convert pwms to dict id2matrix
    id2matrix = pwm_to_pwmobj_batch(pwm_input=batch_custom_pwm, pwm_file_format=pwm_file_format)

    # compile list of all tfs
    tf_list = []

    # create min score dict
    tf_minscores = {}

    for id,matrix in id2matrix.items():

        # get tf name
        tfac_id = id[0] + '_' + id[1].upper()

        # add to list of all tfs
        tf_list.append(tfac_id)

        # get pwm data
        tf_data[tfac_id] = matrix

        # get threshold
        tf_minscores[tfac_id] = batch_pwm_min_score

        # assign color
        tf_colors[tfac_id] = batch_pwm_tf_color

    return tf_colors, tf_data, tf_minscores, tf_list

########################################################*
# convert_tf_info_to_dicts
########################################################*
def convert_tf_info_to_dicts(input_file):

    # read in file 
    df = pd.read_csv(input_file, sep='\t')

    # define dicts
    tf2iupac, tf2color, tf2data, tf2minaff = {}, {}, {}, {}

    # get col names
    df_cols = df.columns
    tf_name_col = df_cols[0]
    color_col = df_cols[1]
    site_def_col = df_cols[2]
    ref_data_col = df_cols[3]
    minaff_data_col = df_cols[4]

    # make sure there is a color for each tf listed 
    for tfac,color in zip(df[tf_name_col], df[color_col]):
        if pd.isnull(tfac) or pd.isnull(color):
            raise ValueError('The TF name and color must be populated for every entry in the input TF information file.')
    
    # populate dictionaries
    for tf_name,iupac,color,ref_data,minaff in zip(df[tf_name_col], df[site_def_col], df[color_col], df[ref_data_col], df[minaff_data_col]):

        tf_name = tf_name.upper()

        # check if tf name already exists in dict
        if pd.isnull(tf_name):
            raise ValueError(f'TF names must be populated.')
        else:
            if tf_name in tf2data:
                raise ValueError('Please make sure all TF names are unique.')

        # check color validity AND make sure each color is unique
        if pd.isnull(color):
            raise ValueError(f'Please assign {tf_name} a color.')
        else:
            check_color(color)
            # if color in tf2color.values():
            #     raise ValueError('Please make sure each TF has a unique assigned color.')
            tf2color[tf_name] = color
        
        # check iupac validity; if null then there must be ref data
        if pd.isnull(iupac):
            raise ValueError(f'Please assign {tf_name} a binding site definition.')
        else:
            iupac = check_nt_iupac(iupac)
            tf2iupac[tf_name] = iupac
        # elif (pd.isnull(ref_data)) and (pd.isnull(iupac)): #not allowing pfm data in tf info file anymore so iupac is required
        #     raise ValueError('If no site definition is provided, then there must be at least PFM data given.')

        # add optional parameters (accounting for the value being null)
        if not pd.isnull(ref_data):
            tf2data[tf_name] = ref_data
        if not pd.isnull(minaff):
            if (type(minaff) != float) and (type(minaff) != int): 
                raise ValueError('Minimum affinity must be a valid number')
            if (minaff < 0) or (minaff > 1):
                raise ValueError('Minimum affinity must be between 0 and 1')
            tf2minaff[tf_name] = minaff

    return tf2color, tf2iupac, tf2data, tf2minaff


##############################################################################################################
# 03 - Annotate and Visualize TF Sites (main function: annotateAndVisualizeTfSites) 
##############################################################################################################
# Helper Function List (shared with annotateTfSites)
#    - singleSeqAnnotateTfSites: annotate one seq w/ table output
#    - plot_sites: plot polygons as binding sites
#    - append_images_with_top_border: stitch images together to output full seq
#    - extract_tf_info_from_batch_pfm: add pfm tfs to tf info dictionaries
#    - convert_tf_info_to_dicts: get tf info dicts from input tsv file 

# Helper Function List: 
#    - add_ones_numline: use in plot_number_table to adjust for non-multiples of 10
#    - plot_number_table: add tens numline and nts in seq
#    - singleSeqVisualizeTfSites: create visualization for one seq 
#    - singleSeqAnnotateAndVisualizeTfSites: create table + visualization for one seq
#    - annotateAndVisualizeTfSites_dict_inputs: use tf info dictionaries as input
#    - annotateAndVisualizeTfSites_file_input: use tf info tsv input
##############################################################################################################

# given a table matrix, add number line to it  
def add_ones_numline(label_matrix, start_pos):

    # get seq length 
    seq_len = len(label_matrix[0])

    # add digits until first multiple of 10 is reached
    ones_numline = []
    last_digit = start_pos % 10
    for i in range(last_digit, 10):
        ones_numline.append(i)
    seq_len -= len(ones_numline) 

    # number of times to repeat 0-9
    repeats = int(seq_len / 10)
    ones_numline.append(0)
    for i in range(repeats): 
        for num in [1,2,3,4,5,6,7,8,9,0]:
            ones_numline.append(num) 
    
    # add extra nums from remainder 
    remainder = seq_len % 10 
    if remainder != 0: # if number is not multiple of 10 
        for num in range(1,remainder+1): 
            ones_numline.append(num)

    # add ones number line to matrix 
    ones_numline = ones_numline[:-1]
    label_matrix.append(ones_numline)

    return label_matrix


# plot table 
def plot_number_table(fig, ax, seq, start_x, end_x):

    # # change start and end x coordinates to 0-indexed
    # start_x -= 1 
    # end_x -= 1

    # length of seq
    seq_length = end_x - start_x 

    # adjust start x coordinate by rounding to nearest 10 
    start_x_1idx = start_x + 1
    diff = 10 - (start_x_1idx % 10)  
    if diff == 10: 
        diff = 0
    adj_start_x = start_x_1idx + diff 

    # obtain y pos for tens numline
    y_pos_of_numline = fig.get_size_inches()[1]
    
    # tens numline 
    for num in range(adj_start_x, end_x+1, 10): 

        # get x label 
        # x_label = num-adj_start_x+diff 

        # plot tens number 
        ax.text(x=num-1, y=0, s=str(num), horizontalalignment='left', c='grey', fontfamily='monospace', fontsize=13) # subtract x pos by 1 to account for 0-indexing 
    
    # add ones number line 
    label_matrix = []
    label_matrix.append(list(seq[start_x:end_x])) 
    label_matrix = add_ones_numline(label_matrix = label_matrix, 
                                    start_pos = start_x+1) # change start x to be 1-indexed so the number line starts at 1

    # add nucleotides and number line
    range_list = [i for i in range(start_x,end_x)]
    # print(range_list)
    for x_pos, nt, ones_num in zip(range_list, label_matrix[0], label_matrix[1]):
        # ones_num = label_matrix[1][x_pos]
        ax.text(x=x_pos, y=1.25, s=ones_num, horizontalalignment='left', c='black', fontfamily='monospace', fontsize=13)
        ax.text(x=x_pos, y=2.5, s=nt, horizontalalignment='left', c='black', fontfamily='monospace', fontsize=13)

    # remove extra space
    plt.tight_layout(pad=0)

    return fig, ax

def plot_red_carrot_boxes(fig, ax, seq, start_x, end_x, all_sites_df):

    # adjust seq
    new_seq = seq[start_x : end_x + 1]

    # check which positions will have red box
    pos2count = {pos:0 for pos in range(start_x, end_x+1)}
    # print(all_sites_df)
    for start,end in zip(all_sites_df['Start Position (1-indexed)'], all_sites_df['End Position (1-indexed)']):
        if start < end:
            for i in range(start,end+1):
                if  i in pos2count: # binding site could exceed current range
                    pos2count[i] += 1
        else:
            for i in range(end,start+1):
                if  i in pos2count: # binding site could exceed current range
                    pos2count[i] += 1
    # print(all_sites_df)
    # print(pos2count)
    pos2redbox = {}
    for pos in range(start_x, end_x+1):
        if pos2count[pos] > 15:
            pos2redbox[pos] = True
        else:
            pos2redbox[pos] = False
    # print(pos2redbox)

    # plot red boxes
    # pos2redbox = {i:True for i in range(start_x, end_x+1)}
    # print(pos2redbox)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    x_width = xmax - xmin
    # print(x_width)    
    for pos,val in pos2redbox.items():

        if val:

            # add rectangle
            rect = patches.Rectangle(
                (pos, ymax - 1),  # x in axes coords (normalized), y near top
                1 - 0.25,          # width in axes coords
                0.75,                      # height of the box
                color='red',
                clip_on=False
            )
            ax.add_patch(rect)

            # add carrot
            ax.text(
                pos, ymax - 1, '^',
                ha='left', va='center',
                fontsize=13, color='white',
                clip_on=False
            )

    # for pos,val in pos2redbox.items():
    #     if val:

    #         # add rectangle
    #         rect = patches.Rectangle(
    #             (pos / len(new_seq), 1.02),  # x in axes coords (normalized), y near top
    #             1 / len(new_seq) - 0.001,          # width in axes coords
    #             0.03,                      # height of the box
    #             transform=ax.transAxes,
    #             color='red',
    #             clip_on=False
    #         )
    #         ax.add_patch(rect)

    #         # add carrot
    #         ax.text(
    #             pos / len(new_seq), 1.04, '^',
    #             ha='left', va='top',
    #             transform=ax.transAxes,
    #             fontsize=13, color='white',
    #             clip_on=False
    #         )

    return fig,ax
       


def singleSeqVisualizeTfSites(seq, seq_name, out_image_format, tf_iupacs, tf_colors, tf_dfs, tf_kmerlens,
                           plot_dpi=200, zoom=None, plot_window=None, out_directory='./', output_svg=None, 
                           output_name=None):

    # if zoom chosen as out image format (potentially reassign to windows format instead)
    if out_image_format == 'Zoom':
        zoom_length = zoom[1] - zoom[0]
        if zoom is None:
            plot_window = 500
            out_image_format == 'Windows'
            print('Warning: Zoom range was not given. Sequence(s) will be plotted in windows.')
        if zoom_length > 500:
            zoom = (zoom[0], zoom[0]+499)
            print('Warning: Zoom range must be less than 500. Output was returned with a zoom range of 500 with the given starting coordinate.')
        if zoom[0] == 0:
            print('Warning: Zoom coordinates must be 1-indexed (numbering starts at 1).')
            
    # if window is chosen as the out image format 
    if out_image_format == 'Windows':
        if plot_window is None:
            plot_window = 500 # default window size
        # if plot_window > 500:
        #     plot_window = 500
        #     print('Warning: Number of bases included per plot must be less than 500. Output was returned with a window size of 500.')

    # zoom into portion of the seq
    if out_image_format == 'Zoom':

        # create new var for tf dictionaries
        new_tf_iupacs = tf_iupacs
        new_tf_colors = tf_colors
        new_tf_dfs = tf_dfs
        new_tf_kmerlens = tf_kmerlens

        # subset sequence based on zoom (assume that zoom coords are given as 1-indexed) 
        start_pos = zoom[0] - 1
        end_pos = zoom[1]
        # adj_seq = seq[start_pos:end_pos] 

        # plot sites and then table underneath 
        fig1, ax1, image_sites_df, excluded_sites_df, all_sites_df = plot_sites(seq = seq, 
                                                                                seq_name = seq_name, 
                                                                                start_x = start_pos, 
                                                                                end_x = end_pos, 
                                                                                tf_iupacs = new_tf_iupacs, 
                                                                                tf_colors = new_tf_colors, 
                                                                                tf_dfs = new_tf_dfs, 
                                                                                tf_kmerlens = new_tf_kmerlens, 
                                                                                plot_dpi = plot_dpi) 
        
        fig2, ax2 = plot_number_table(fig = fig1, 
                                      ax = ax1, 
                                      seq = seq, 
                                      start_x = start_pos, 
                                      end_x = end_pos)
        
        fig3, ax3 = plot_red_carrot_boxes(fig=fig2, 
                                          ax=ax2, 
                                          seq = seq, 
                                          start_x = start_pos, 
                                          end_x = end_pos,
                                          all_sites_df = all_sites_df)
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        new_output_name = output_name + '_image' + '_seq=' + seq_name + '_zoom=' + str(start_pos+1) + '-' + str(end_pos) + '.png'
        if out_directory != './':
            out_directory = out_directory.strip('/') # get rid of trailing slashes
            new_output_name = out_directory + '/' + new_output_name 
        fig3.savefig(new_output_name, bbox_inches='tight', pad_inches=0)
        print(new_output_name + ' has been created')

        if output_svg:
            svg_outfile = new_output_name[:-3] + 'svg'
            fig3.savefig(svg_outfile, bbox_inches='tight', pad_inches=0)
            print(svg_outfile + ' has been created')
        
    
    # new figure plotted for size of plot window (min = 500 bp) 
    if out_image_format == 'Windows':    

        # get window list
        window_list = [i for i in range(0, len(seq), plot_window)] + [len(seq)]

        # plot sites for each window
        all_vis_files = []
        for i in range(len(window_list[:-1])):

            # create new var for tf dictionaries
            new_tf_iupacs = tf_iupacs
            new_tf_colors = tf_colors
            new_tf_dfs = tf_dfs
            new_tf_kmerlens = tf_kmerlens

            # take window of seq (assume that zoom coords are given as 1-indexed) 
            start_pos = window_list[i]
            end_pos = window_list[i+1]
            # adj_seq = seq[start_pos:end_pos]

            fig1, ax1, image_sites_df, excluded_sites_df, all_sites_df = plot_sites(seq=seq, 
                                                                                    seq_name=seq_name, 
                                                                                    start_x = start_pos, 
                                                                                    end_x = end_pos, 
                                                                                    tf_iupacs = new_tf_iupacs, 
                                                                                    tf_colors = new_tf_colors, 
                                                                                    tf_dfs = new_tf_dfs, 
                                                                                    tf_kmerlens = new_tf_kmerlens, 
                                                                                    plot_dpi = plot_dpi) 
            
            fig2, ax2 = plot_number_table(fig = fig1, 
                                            ax = ax1, 
                                            seq = seq, 
                                            start_x = start_pos, 
                                            end_x = end_pos)
            
            fig3, ax3 = plot_red_carrot_boxes(fig=fig2, 
                                            ax=ax2, 
                                            seq = seq, 
                                            start_x = start_pos, 
                                            end_x = end_pos,
                                            all_sites_df = all_sites_df)
            
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            new_output_name = output_name + '_image' + '_seq=' + seq_name + '_pos=' + str(start_pos+1) + '-' + str(end_pos) + '.png'
            if out_directory != './':
                out_directory = out_directory.strip('/') # get rid of trailing slashes
                new_output_name = out_directory + '/' + new_output_name 
            fig3.savefig(new_output_name, bbox_inches='tight', pad_inches=0)
            print(new_output_name + ' has been created')
            all_vis_files.append(new_output_name)

            if output_svg:
                svg_outfile = new_output_name[:-3] + 'svg'
                fig3.savefig(svg_outfile, bbox_inches='tight', pad_inches=0)
                print(svg_outfile + ' has been created')

        ## create stitched image 
        if len(all_vis_files) > 1:
            full_seq_image = output_name + '_image' + '_seq=' + seq_name + '_full_range.png'
            if out_directory != './':
                out_directory = out_directory.strip('/') # get rid of trailing slashes
                full_seq_image = out_directory + '/' + full_seq_image
            append_images_with_top_border(all_vis_files, full_seq_image)

    # close plots
    plt.close('all')

    return image_sites_df, excluded_sites_df
    
    # if outfile is not None:
    #     return new_outfile


def singleSeqAnnotateAndVisualizeTfSites(seq, seq_name, out_image_format, tf_iupacs, tf_colors, tf_data={}, tf_minscores={}, 
                                         tf_minaffs={}, plot_dpi=200, zoom_range=None, plot_window=None, out_directory="./", 
                                         tf_pseudocounts=False, input2tfs=None, output_svg=None, output_name=None): 

    # iterate for each tf
    # min_df_length = float('inf')
    tf_dfs = {}
    tf_kmerlens = {}
    for tf_name in tf_colors: # color must exist for each tf

        # get iupac for this tf, if exists
        if tf_name in tf_iupacs:
            tf_binding_site_definition = tf_iupacs[tf_name]
        else:
            tf_binding_site_definition = None

        # get ref data for this tf, if exists; if pfm data, then convert to pwm object
        if tf_name in tf_data: 
            tf_reference_data = tf_data[tf_name]
        else: 
            tf_reference_data = None

        # get binding site min score for this tf, if exists
        if tf_name in tf_minscores: 
            tf_min_score = tf_minscores[tf_name]
        else: 
            tf_min_score = 0.7

        # get min affinity to plot for this tf, if exists 
        if tf_name in tf_minaffs: 
            tf_min_affinity = tf_minaffs[tf_name]
        else: 
            tf_min_affinity = 0

        # get ref data type and process reference data 
        if tf_reference_data is None: 
            tf_ref_data_type = None
        elif isinstance(tf_reference_data, np.ndarray):
            tf_ref_data_type = 'Score'
        else:
            tf_ref_data_type = 'Affinity'
            tf_reference_data = loadNormalizedFile(tf_reference_data)
        
        # annotate tf sites for one seq and one tf
        tf_kmer_length = get_kmer_length(tf_iupac = tf_binding_site_definition, 
                                         tf_ref_data_type = tf_ref_data_type, 
                                         tf_ref_data = tf_reference_data)
        
        df = singleSeqAnnotateTfSites(seq = seq, 
                                      seq_name = seq_name, 
                                      tf_name = tf_name, 
                                      tf_kmer_length = tf_kmer_length,
                                      tf_ref_data_type = tf_ref_data_type,
                                      tf_ref_data = tf_reference_data, 
                                      tf_iupac = tf_binding_site_definition, 
                                      tf_min_affinity = tf_min_affinity,
                                      tf_min_score = tf_min_score, 
                                      tf_pseudocounts = tf_pseudocounts, 
                                      zoom_range = zoom_range)
        tf_dfs[tf_name] = df
        tf_kmerlens[tf_name] = tf_kmer_length

        # # check if there are zero binding sites
        # if len(df) < min_df_length: 
        #     min_df_length = len(df) 

    # print(tf_colors)
    # print(tf_kmerlens)

    # # get name of output visualization file
    # vis_out_file = output_name + '_seq=' + seq_name # distinguish files by seq 

    # visualize sites for all tfs on this seq 
    # if min_df_length != 0: 
    image_sites_df, excluded_sites_df = singleSeqVisualizeTfSites(seq = seq, 
                                                                    seq_name = seq_name, 
                                                                    out_image_format = out_image_format, 
                                                                    tf_iupacs = tf_iupacs, 
                                                                    tf_colors = tf_colors, 
                                                                    tf_dfs = tf_dfs, 
                                                                    tf_kmerlens = tf_kmerlens, 
                                                                    plot_dpi = plot_dpi, 
                                                                    zoom = zoom_range, 
                                                                    plot_window = plot_window,
                                                                    out_directory = out_directory, 
                                                                    output_svg = output_svg,
                                                                    output_name = output_name)
    
    
    # # separate tf aff and pwm data
    # pwm_final_df = final_df[final_df['Ref Data Type'] == 'Score']
    # pwm_final_df = pwm_final_df[['Sequence Name', 'TF Name', 'Matrix ID', 'Kmer ID', 'Kmer', 'Start Position (1-indexed)',	
    #                  'End Position (1-indexed)', 'Ref Data Type', 'Value', 'Site Direction', 'Duplicate Kmer IDs']]
    # aff_final_df = final_df[(final_df['Ref Data Type'] == 'Affinity') | (final_df['Ref Data Type'] == '')]
    # aff_final_df = aff_final_df.drop('Matrix ID', axis=1)
    # final_df = final_df[['Sequence Name', 'TF Name', 'Matrix ID'] + list(final_df.columns)[2:10]]

    # create output files
    for curr_df,file_suffix in zip([image_sites_df, excluded_sites_df], ['', '_sites-excluded-from-image']):

        if len(curr_df) != 0:
        
            # annotated file
            output_name = output_name.split('.')[0]
            if zoom_range is not None:
                annotated_out_file = output_name + '_table' + '_seq=' + seq_name + '_zoom=' + str(zoom_range[0]) + '-' + str(zoom_range[1]) + file_suffix + '.tsv'
            else:
                annotated_out_file = output_name + '_table' + '_seq=' + seq_name + file_suffix + '.tsv' # distinguish files by seq

            # output df to file
            if out_directory != './':
                out_directory = out_directory.strip('/') # get rid of trailing slashes
                annotated_out_file = out_directory + '/' + annotated_out_file
            curr_df.to_csv(annotated_out_file, sep="\t",index=None)
            print(f'{annotated_out_file} has been created')
    
    
    # # create output annotated table files 
    # output_name = output_name.split('.')[0]
    # for input,tf_list in input2tfs.items():

    #     # get name for output annotated table file
    #     if zoom_range is not None:
    #         annotated_out_file = output_name + '_table' + '_seq=' + seq_name + '_zoom=' + str(zoom_range[0]) + '-' + str(zoom_range[1]) + f'_{input}' + '.tsv'
    #     else:
    #         annotated_out_file = output_name + '_table' + '_seq=' + seq_name + f'_{input}' + '.tsv' # distinguish files by seq

    #     # # concatenate dfs corresponding to tf list
    #     # all_sites_df = pd.DataFrame()
    #     # for tfac in tf_list:
    #     #     curr_df = tf_dfs[tfac]
    #     #     all_sites_df = pd.concat([all_sites_df, curr_df], ignore_index=True)

    #     # # add necessary columns
    #     # df.insert(1, 'Kmer ID', [seq_name + ':' + str(i+1) for i in range(len(df))])

    #     # output df to file
    #     if out_directory != './':
    #         out_directory = out_directory.strip('/') # get rid of trailing slashes
    #         annotated_out_file = out_directory + '/' + annotated_out_file
    #     all_sites_df.to_csv(annotated_out_file, sep="\t",index=None)
    #     print(f'{annotated_out_file} has been created')

    # else: 
    #     return "No binding sites found..." 


def annotateAndVisualizeTfSites_dict_inputs(dna_sequences_to_annotate,
                                            binding_site_color={},
                                            binding_site_definition={}, 
                                            binding_site_reference_data={},
                                            binding_site_min_aff={},
                                            pwm_reference_data=None,
                                            pwm_min_score=0.7,
                                            pwm_tf_color='grey',
                                            pwm_pseudocounts=False,
                                            output_svg=False,
                                            plot_resolution=150,
                                            zoom_range=None,
                                            output_name=None,
                                            out_directory="./",
                                            pwm_file_format = 'jaspar'):
    
    '''
    Predict transcription factor binding sites across a DNA sequence. Use dictionaries to store TF information. \n
    Parameters: \n
    - dna_sequences_to_annotate (.tsv): File containing one or more DNA sequences to be annotated. \n 
        - columns (strict order, flexible names) -> sequence name, sequence \n
    - binding_site_color (dict): Dictionary mapping each TF to its plotted color. (key = TF name, value = color) \n
    - binding_site_definition (dict): Dictionary mapping each TF to its core binding site IUPAC definition (key = TF name, value = binding site definition). \n
    - binding_site_reference_data (dict, default = {}): Dictionary mapping each TF to its affinity data (key = TF name, value = PBM data). \n
    - binding_site_min_aff (dict, default = {}): Dictionary mapping each TF to its affinity threshold for reporting binding sites. \n
    - pwm_reference_data (.txt, default = None): File containing PWMs to predict and score binding sites. \n 
    - pwm_min_score (float, default = 0.7): PWM score required to predict a site. \n 
    - pwm_tf_color (string, default = 'grey'): Color of sites scored by PWMs on the output visualization. \n 
    - pwm_pseudocounts (boolean, default = False): Whether to use pseudocounts to calculate PWM scores. \n 
    - output_svg (boolean, default = False): Option to output images as `.svg` in addition to `.png`. For manuscript preparation, `.svg` format is preferable. \n 
    - plot_resolution (integer, default = 150): Resolution of the plot, in dots (pixels) per inch. Manuscripts require 300 DPI. The DPI does not affect the resolution of `.svg` files. \n 
    - zoom_range (tuple, default = None): Indicates the region of the DNA sequence to visualize, given a start and end coordinate. The numbers in the range are inclusive and 1-indexed. For example, the first 200 nucleotides of the sequence would be specified as: (1,200). \n
    - out_directory (string, default = ./): Directory to contain all output files. \n
    '''

    # check bools
    pwm_pseudocounts = check_bool(pwm_pseudocounts)
    output_svg = check_bool(output_svg)

    # set internal variables
    input_file = dna_sequences_to_annotate
    tf_iupacs = binding_site_definition
    tf_colors = binding_site_color
    tf_data = binding_site_reference_data
    tf_minaffs = binding_site_min_aff
    plot_dpi = plot_resolution
    zoom = zoom_range

    # get output name, just in case file extension is included in the file name by the user
    output_name = output_name.split('.')[0]

    # # set value error for out_image_format
    # if out_image_format is None:
    #     raise ValueError('The value of out_image_format must be either “Zoom” or “Windows.”')

    # # create tf fam to color dictionary
    # tfam2color = {}
    # for tfac,color in tf_colors.items():
    #     stripped_tfac = tfac.upper().strip('1234567890-')
    #     tfam2color[stripped_tfac] = color

    # convert all pwm files to pwm objects -> and it will be consistent with batch pwm custom input
    if tf_data != {}:
        for tfac,data in tf_data.items():
            with open(data) as file:
                if file.readline()[0] == '>':
                    pwm = pwm_to_pwmobj(pwm_input=data)
                    tf_data[tfac] = pwm

    # map each input with tf info to list of tfs (if it exisits) -> for tf info tsv, get list of tfs using color dict since every tf must be assigned a color
    input2tfs = {}
    if tf_data != {}:
        input2tfs['tf-info-table'] = []
        for tfac in binding_site_color:
            input2tfs['tf-info-table'].append(tfac)
    if pwm_reference_data is not None:
        input2tfs['batch-custom-pwm'] = []

    # get tf information from batch pwm input (and map each input to list of tfs)
    tf_minscores = {}
    if pwm_reference_data is not None:
        tf_colors, tf_data, tf_minscores, tf_list = extract_tf_info_from_batch_pwm(batch_custom_pwm = pwm_reference_data, 
                                                                                   batch_pwm_min_score = pwm_min_score, 
                                                                                   batch_pwm_tf_color = pwm_tf_color,
                                                                                   pwm_pseudocounts = pwm_pseudocounts, 
                                                                                   tf_colors = tf_colors, 
                                                                                   tf_data = tf_data, 
                                                                                   pwm_file_format=pwm_file_format)
        input2tfs['batch-custom-pwm'] = tf_list

    # # assign kmer_id to each kmer
    # df.insert(1, 'Kmer ID', [seq_name + ':' + str(i+1) for i in range(len(df))])

    # print(tf_iupacs, tf_colors, tf_data)

    # set out_image_format
    if zoom_range is not None:
        out_image_format = 'Zoom'
    else:
        out_image_format = 'Windows'
        plot_window = None
    
    # iterate through each seq
    with open(input_file) as file:
        file.readline()
        file = csv.reader(file, delimiter='\t')
        for line in file: 

            # obtain seq and seq_name
            input_seq_name = line[0]
            input_seq = line[1]

            # run annotate and visualize on one sequence
            singleSeqAnnotateAndVisualizeTfSites(seq = input_seq, 
                                                 seq_name = input_seq_name, 
                                                 out_image_format = out_image_format,
                                                 tf_iupacs = tf_iupacs, 
                                                 tf_colors = tf_colors, 
                                                 tf_data = tf_data, 
                                                 tf_minscores = tf_minscores, 
                                                 tf_minaffs = tf_minaffs, 
                                                 plot_dpi = plot_dpi, 
                                                 zoom_range = zoom, 
                                                 plot_window = None, #fixed variable
                                                 out_directory = out_directory, 
                                                 tf_pseudocounts = pwm_pseudocounts, 
                                                 input2tfs = input2tfs, 
                                                 output_svg = output_svg,
                                                 output_name = output_name)


def annotateAndVisualizeTfSites_file_input(dna_sequences_to_annotate,
                                           tf_info=None,
                                           pwm_reference_data=None,
                                           pwm_min_score=0.7,
                                           pwm_tf_color='grey',
                                           pwm_pseudocounts=False,
                                           plot_resolution=150,
                                           output_svg=False,
                                           zoom_range=None,
                                           output_name=None,
                                           out_directory="./",
                                           pwm_file_format='jaspar'):

    '''
    Predict transcription factor binding sites across a DNA sequence. Use file to store TF information. \n
    Parameters: \n
    - dna_sequences_to_annotate (.tsv): File containing one or more DNA sequences to be annotated. \n 
        - columns (strict order, flexible names) -> sequence name, sequence. \n
    - tf_info (.tsv): File containing all information for the transcription factors being analyzed. \n
        - columns (strict order, flexible names) -> TF name, color, binding site definition, reference data, and minimum affinity. \n
    - pwm_reference_data (.txt, default = None): File containing PWMs to predict and score binding sites. \n 
    - pwm_min_score (float, default = 0.7): PWM score required to predict a binding site. \n 
    - pwm_tf_color (string, default = 'grey'): Color of sites scored by PWMs on the output visualization. \n 
    - pwm_pseudocounts (boolean, default = False): Whether to use pseudocounts to calculate PWM scores. \n 
    - plot_resolution (integer, default = 150): Resolution of the plot, in dots (pixels) per inch. Manuscripts require 300 DPI. The DPI does not affect the resolution of `.svg` files. \n 
    - output_svg (boolean, default = False): Option to output images as `.svg` in addition to `.png`. For manuscript preparation, `.svg` format is preferable. \n 
    - zoom_range (tuple, default = None): Indicates the region of the DNA sequence to visualize, given a start and end coordinate. The numbers in the range are inclusive and 1-indexed. For example, the first 200 nucleotides of the sequence would be specified as: (1,200). \n
    - out_directory (string, default = ./): Directory to contain all output files. \n
    '''

    # check that inputs are valid (the checks for tf_info are done in convert_tf_info_to_dicts) 
    if type(pwm_min_score) != float: raise ValueError('pwm_min_score must be a valid float number.')
    if (pwm_min_score < 0) or (pwm_min_score > 1): raise ValueError('pwm_min_score must be between 0 and 1')
    pwm_pseudocounts = check_bool(pwm_pseudocounts)
    if type(plot_resolution) != int: raise ValueError('plot_resolution must be a valid integer')
    output_svg = check_bool(output_svg)
    if (type(zoom_range) != tuple) and (zoom_range is not None): raise ValueError('zoom_range must be a valid tuple.')

    # convert tf info file to dicts
    if tf_info is not None:
        tf2color, tf2iupac, tf2data, tf2minaff = convert_tf_info_to_dicts(tf_info)
    else:
        tf2color, tf2iupac, tf2data, tf2minaff = {}, {}, {}, {}

    # run dict input function
    annotateAndVisualizeTfSites_dict_inputs(dna_sequences_to_annotate = dna_sequences_to_annotate,
                                            binding_site_color = tf2color,
                                            binding_site_definition = tf2iupac, 
                                            binding_site_reference_data = tf2data,
                                            binding_site_min_aff = tf2minaff,
                                            pwm_reference_data = pwm_reference_data,
                                            pwm_min_score = pwm_min_score,
                                            pwm_tf_color = pwm_tf_color,
                                            pwm_pseudocounts = pwm_pseudocounts,
                                            plot_resolution = plot_resolution,
                                            output_svg = output_svg,
                                            zoom_range = zoom_range,
                                            output_name = output_name,
                                            out_directory = out_directory,
                                            pwm_file_format = pwm_file_format)
    
# final function name for tfsites website
def visualizeTfSitesOnSequences(dna_sequences_to_annotate,
                                binding_site_color={},
                                binding_site_definition={}, 
                                binding_site_reference_data={},
                                binding_site_min_aff={},
                                pwm_reference_data=None,
                                pwm_min_score=0.7,
                                pwm_tf_color='grey',
                                pwm_pseudocounts=False,
                                output_svg=False,
                                plot_resolution=150,
                                zoom_range=None,
                                output_name=None,
                                out_directory="./"):
    
    annotateAndVisualizeTfSites_dict_inputs(dna_sequences_to_annotate=dna_sequences_to_annotate,
                                            binding_site_color=binding_site_color,
                                            binding_site_definition=binding_site_definition, 
                                            binding_site_reference_data=binding_site_reference_data,
                                            binding_site_min_aff=binding_site_min_aff,
                                            pwm_reference_data=pwm_reference_data,
                                            pwm_min_score=pwm_min_score,
                                            pwm_tf_color=pwm_tf_color,
                                            pwm_pseudocounts=pwm_pseudocounts,
                                            output_svg=output_svg,
                                            plot_resolution=plot_resolution,
                                            zoom_range=zoom_range,
                                            output_name=output_name,
                                            out_directory=out_directory)


##############################################################################################################
# 04 - Annotate and Visualize In Silico SNV Analysis (main function: annotateAndVisualizeInSilicoSnvs)
##############################################################################################################
# Helper Function List (shared with annotateTfSites)
#    - singleSeqAnnotateTfSites: annotate one seq w/ table output
#    - plot_sites: plot polygons as binding sites
#    - append_images_with_top_border: stitch images together to output full seq
#    - convert_tf_info_to_dicts: get tf info dicts from input tsv file 

# Helper Function List (unique to annotateAndVisualizeInSilicoSnvs)
#    - find_snv_effect_no_ref: snv effects for no ref data
#    - find_snv_effect_pbm: get all possible snv effects for PBM
#    - find_snv_effect_pfm: get all possible snv effects for PFM
#    - annotateInSilicoSnvs: return table of all annotated snv effects for one seq and one TF
#    - create_snv_dict: create dictionary with the mapping: pos -> nucleotide (ref and alts) -> effect 
#    - plot_snv_table: plot the table displaying the effect of each SNV with its color
#    - singleSeqVisualizeInSilicoSnvs: create visualization for one seq
#    - singleSeqAnnotateAndVisualizeInSilicoSnvs: create table + visualization for one seq
#    - annotateAndVisualizeInSilicoSnvs_dict_inputs: use tf info dictionaries as input
#    - annotateAndVisualizeTfSites_file_input: use tf info tsv input
##############################################################################################################

# snv effects for no reference data
def find_snv_effect_no_ref(ref_kmer_input, alt_kmer_input, iupac, mut_type):

    # iupac is mandatory
    if iupac is None: 
        raise ValueError('IUPAC must be given if there is no reference data.')

    # get list of snv effects to look for
    mut_list = mut_type.split(',')

    for ref_kmer,alt_kmer,strand in [(ref_kmer_input,alt_kmer_input,'+'),(rev_comp(ref_kmer_input),rev_comp(alt_kmer_input), '-')]:

        # create iupac object 
        iupac_reobj = iupac_to_regex_pattern(iupac)
    
        # determine when IUPAC is present
        iupacRef = re.search(iupac_reobj, ref_kmer)
        iupacAlt = re.search(iupac_reobj, alt_kmer)

        # denovo snvs
        if ('denovo' in mut_list or 'all' in mut_list or 'all'==mut_list):
            
            # check for de novo 
            if iupacRef==None and iupacAlt:
                return [ref_kmer, alt_kmer, strand, '', '', '', 'denovo']
    
        # del snvs 
        if ('del' in mut_list or 'all' in mut_list or 'all'==mut_list):
            
            if iupacRef and iupacAlt==None:
                return [ref_kmer, alt_kmer, strand, '', '', '', 'del']
    
    return ''

# find snv effects given the ref and alt kmer
def find_snv_effect_pbm(ref_kmer_input, alt_kmer_input, iupac, seq2aff, mut_type, opt_thres=1, subopt_thres=1):

    # get list of snv effects to look for
    mut_list = mut_type.split(',')

    for ref_kmer,alt_kmer,strand in [(ref_kmer_input,alt_kmer_input,'+'),(rev_comp(ref_kmer_input),rev_comp(alt_kmer_input), '-')]:

        # create iupac object 
        iupac_reobj = iupac_to_regex_pattern(iupac)
    
        # determine when IUPAC is present
        iupacRef = re.search(iupac_reobj, ref_kmer)
        iupacAlt = re.search(iupac_reobj, alt_kmer)

        # find affinity (same for fwd and rev kmers)
        ref_aff = seq2aff[ref_kmer_input] 
        alt_aff = seq2aff[alt_kmer_input]

        # inc or dec snvs 
        if ('inc' in mut_list) or ('dec' in mut_list) or ('all' in mut_list):
            
            # if iupac is present in both ref and alt
            if iupacRef and iupacAlt:
                
                # account for kmers that have no affinity (not contained in og pbm data)
                if (alt_aff is None) or (ref_aff is None):
                    return [ref_kmer, alt_kmer, strand, ref_aff, alt_aff, None, 'none']
                
                # if kmers have aff, proceed normally to calculate fold change
                else:
                    fold_change = round(alt_aff / ref_aff, 3)
                    inc_condition = fold_change > opt_thres     # does snv optimize? 
                    dec_condition = fold_change < subopt_thres  # does snv suboptimize? 

                    if inc_condition and (('inc' in mut_list) or ('all' in mut_list) or ('all'==mut_list)):
                        return [ref_kmer, alt_kmer, strand, ref_aff, alt_aff, fold_change, 'inc']

                    if dec_condition and (('dec' in mut_list) or ('all' in mut_list) or ('all'==mut_list)):
                        return [ref_kmer, alt_kmer, strand, ref_aff, alt_aff, fold_change, 'dec']
        
        # denovo snvs
        if ('denovo' in mut_list or 'all' in mut_list or 'all'==mut_list):
            
            # check for de novo 
            if iupacRef==None and iupacAlt:
                return [ref_kmer, alt_kmer, strand, '', alt_aff, '', 'denovo']

        # del snvs 
        if ('del' in mut_list or 'all' in mut_list or 'all'==mut_list):
            
            if iupacRef and iupacAlt==None:
                return [ref_kmer, alt_kmer, strand, ref_aff, '', '', 'del']
        
    return ''


# find snv effects given the ref and alt kmer
def find_snv_effect_pfm(ref_kmer_input, alt_kmer_input, iupac, minscore, pwm, mut_type, opt_thres=1, subopt_thres=1):

    # get list of snv effects to look for
    mut_list = mut_type.split(',')

    for ref_kmer,alt_kmer,strand in [(ref_kmer_input,alt_kmer_input,'+'),(rev_comp(ref_kmer_input),rev_comp(alt_kmer_input), '-')]:

        # get pfm score of ref and alt
        max_score = all_funcs_get_max_score_of_pwm(pwm)
        min_score = all_funcs_get_min_score_of_pwm(pwm)
        ref_score = all_funcs_get_score_of_kmer(pwm, ref_kmer, max_score, min_score)
        alt_score = all_funcs_get_score_of_kmer(pwm, alt_kmer, max_score, min_score)

        # instances where ref and alt score are zero
        if (ref_score == 0) or (alt_score == 0):
            continue

        # determine binding site condition based on iupac or min score criteria
        if iupac is not None:
            
            # create iupac object 
            iupac_reobj = iupac_to_regex_pattern(iupac)
        
            # determine when IUPAC is present
            iupacRef = re.search(iupac_reobj, ref_kmer)
            iupacAlt = re.search(iupac_reobj, alt_kmer)
            inc_dec_condition = (iupacRef and iupacAlt)
            denovo_condition = (iupacRef==None) and (iupacAlt)
            del_condition = (iupacRef) and (iupacAlt==None)

        else:
            inc_dec_condition = (ref_score >= minscore) and (alt_score >= minscore)
            denovo_condition = (ref_score <= minscore) and (alt_score >= minscore)
            del_condition = (ref_score >= minscore) and (alt_score <= minscore)

        # inc or dec snvs 
        if ('inc' in mut_list) or ('dec' in mut_list) or ('all' in mut_list):

            # check if kmer follows threshold (aka it is considered a site)
            if inc_dec_condition:

                fold_change = round(alt_score / ref_score, 3)
                inc_condition = fold_change > opt_thres     # does snv optimize? 
                dec_condition = fold_change < subopt_thres  # does snv suboptimize? 

                if inc_condition and (('inc' in mut_list) or ('all' in mut_list) or ('all'==mut_list)):
                    return [ref_kmer, alt_kmer, strand, ref_score, alt_score, fold_change, 'inc']

                if dec_condition and (('dec' in mut_list) or ('all' in mut_list) or ('all'==mut_list)):
                    return [ref_kmer, alt_kmer, strand, ref_score, alt_score, fold_change, 'dec']
        
        # denovo snvs
        if ('denovo' in mut_list or 'all' in mut_list or 'all'==mut_list):
            
            # check that ref is not a binding site but alt is 
            if denovo_condition:
                return [ref_kmer, alt_kmer, strand, '', alt_score, '', 'denovo']

        # del snvs 
        if ('del' in mut_list or 'all' in mut_list or 'all'==mut_list):

            # check that ref is a binding site but alt is not
            if del_condition:
                return [ref_kmer, alt_kmer, strand, ref_score, '', '', 'del']
        
    return ''


########################################################*
# annotateInSilicoSnvs
########################################################*
# helper function list
#    - create_alt_kmer
#    - get_start_and_end_pos
#    - assign_snv_kmer_ids
#    - report_snv_effect
#    - annotate_snvs_in_single_kmer
#    - apply_zoom_for_snvs
#    - output_tsv
########################################################*

def create_alt_kmer(ref_kmer, pos, base):
    kmer_list = list(ref_kmer)
    kmer_list[pos] = base
    alt_kmer = ''.join(kmer_list)
    return alt_kmer

def get_start_and_end_pos(site_dir, start_pos_1idx, kmer_length):

    # neg site direction
    if site_dir == '-':
        start_pos = start_pos_1idx + kmer_length - 1
        end_pos = start_pos_1idx

    # pos site direction or palindrome
    else: 
        start_pos = start_pos_1idx
        end_pos = start_pos_1idx + kmer_length - 1

    return start_pos, end_pos

def assign_snv_kmer_ids(pos2kmerid, start_pos, tf_name_for_snv_analysis):
    if start_pos in pos2kmerid:
        kmer_id = pos2kmerid[start_pos]
    else:
        kmer_id = tf_name_for_snv_analysis + ':' + str(len(pos2kmerid) + 1)
        pos2kmerid[start_pos] = kmer_id
    return pos2kmerid, kmer_id

def report_snv_effect(df, seq_name, tf_name_for_snv_analysis, kmer_length, ref_kmer, alt_kmer, start_pos_1idx, pos, pos2kmerid, snv_info):

    # if the snv effect is not none, then report it
    if snv_info != '':

        # get ref and alt nts
        ref = ref_kmer[pos]
        alt = alt_kmer[pos]

        # get site direction
        site_dir = snv_info[2]

        # get kmer start and end position 
        start_pos, end_pos = get_start_and_end_pos(site_dir = site_dir, 
                                                   start_pos_1idx = start_pos_1idx, 
                                                   kmer_length = kmer_length)

        # get snv position
        snv_pos = pos+start_pos_1idx

        # assign kmer id 
        pos2kmerid, kmer_id = assign_snv_kmer_ids(pos2kmerid = pos2kmerid, 
                                                  start_pos = start_pos, 
                                                  tf_name_for_snv_analysis = tf_name_for_snv_analysis)

        # add snv to current df 
        df.loc[len(df.index)] = [seq_name, kmer_id, snv_pos, ref, alt, start_pos, end_pos, ref_kmer, alt_kmer] + snv_info[2:]

    return pos2kmerid, df

def report_snv_effect(df, seq_name, tf_name_for_snv_analysis, kmer_length, ref_kmer, alt_kmer, start_pos_1idx, pos, pos2kmerid, snv_info):

    # if the snv effect is not none, then report it
    if snv_info != '':

        # get ref and alt nts
        ref = ref_kmer[pos]
        alt = alt_kmer[pos]

        # get site direction
        site_dir = snv_info[2]

        # get kmer start and end position
        start_pos, end_pos = get_start_and_end_pos(site_dir = site_dir, 
                                                   start_pos_1idx = start_pos_1idx, 
                                                   kmer_length = kmer_length)

        # get snv position
        snv_pos = pos+start_pos_1idx

        # assign kmer id
        pos2kmerid, kmer_id = assign_snv_kmer_ids(pos2kmerid = pos2kmerid, 
                                                  start_pos = start_pos, 
                                                  tf_name_for_snv_analysis = tf_name_for_snv_analysis)

        # add snv to current df 
        df.loc[len(df.index)] = [seq_name, kmer_id, snv_pos, ref, alt, start_pos, end_pos, ref_kmer, alt_kmer] + snv_info[2:]

    return pos2kmerid, df

# annotate_snvs_in_single_kmer(df, tf_name, seq_name, tf_kmer_length, ref_kmer, start_pos_1idx, tf_iupac, seq2aff, 
#                                           tf_min_score, pwm, tf_ref_data_type, mut_type, opt_thres, subopt_thres, pos2kmerid)
def annotate_snvs_in_single_kmer(df, 
                                 seq_name, 
                                 tf_name, 
                                 kmer_length, 
                                 ref_kmer, 
                                 start_pos_1idx, 
                                 iupac, 
                                 seq2aff, 
                                 minscore, 
                                 pwm, 
                                 ref_data_type, 
                                 mut_type, 
                                 opt_thres, 
                                 subopt_thres, 
                                 pos2kmerid):
    
    # list of all possible bases
    base_list = ['A', 'G', 'C', 'T'] 
    
    # introduce SNV at every position in the kmer 
    for pos in range(len(ref_kmer)):
        for base in base_list:
            if ref_kmer[pos] != base: # only use the alt nt 

                # create alt kmer 
                alt_kmer = create_alt_kmer(ref_kmer = ref_kmer, 
                                           pos = pos, 
                                           base = base)

                # search for snv effect by comparing ref and alt kmer 
                if (ref_data_type == 'Affinity'):
                    snv_info = find_snv_effect_pbm(ref_kmer_input = ref_kmer, 
                                                   alt_kmer_input = alt_kmer, 
                                                   iupac = iupac, 
                                                   seq2aff = seq2aff, 
                                                   mut_type = mut_type, 
                                                   opt_thres = opt_thres, 
                                                   subopt_thres = subopt_thres)
                elif ref_data_type == 'Score':
                    snv_info = find_snv_effect_pfm(ref_kmer_input = ref_kmer, 
                                                   alt_kmer_input = alt_kmer, 
                                                   iupac = iupac, 
                                                   minscore = minscore, 
                                                   pwm = pwm, 
                                                   mut_type = mut_type, 
                                                   opt_thres = opt_thres, 
                                                   subopt_thres = subopt_thres)
                else:
                    snv_info = find_snv_effect_no_ref(ref_kmer_input = ref_kmer, 
                                                      alt_kmer_input = alt_kmer, 
                                                      iupac = iupac, 
                                                      mut_type = mut_type)

                # report snv effect, if not none
                pos2kmerid, df = report_snv_effect(df = df, 
                                                   seq_name = seq_name, 
                                                   tf_name_for_snv_analysis = tf_name, 
                                                   kmer_length = kmer_length,
                                                   ref_kmer = ref_kmer, 
                                                   alt_kmer = alt_kmer, 
                                                   start_pos_1idx = start_pos_1idx, 
                                                   pos = pos, 
                                                   pos2kmerid = pos2kmerid, 
                                                   snv_info = snv_info)

    return df  

# apply zoom boundaries - only keep snvs that are within bounds 
def apply_zoom_for_snvs(zoom_range, df):
    lower_bound = zoom_range[0]
    upper_bound = zoom_range[1]
    zoom_subset = (df['SNV Position (1-indexed)'] <= upper_bound) & (df['SNV Position (1-indexed)'] >= lower_bound)
    df = df.loc[zoom_subset,:]
    return df 

def output_tsv(df, out_file, out_directory):

    # check if there are any snv effects 
    if len(df) != 0:

        # clean up df
        df = df.sort_values(by='SNV Position (1-indexed)', ignore_index=True)
        df.drop_duplicates(inplace=True)

        # if name is provided, then output the file 
        if out_file != None:
            if out_directory != './':
                out_directory = out_directory.strip('/') # get rid of trailing slashes
                out_file = out_directory + '/' + out_file
            df.to_csv(out_file, sep="\t",index=None)
            print(out_file + ' has been created') 
        return df 
    
    else:
        raise ValueError('No SNVs for specified threshold...')

def annotateInSilicoSnvs(seq, 
                         seq_name, 
                         tf_name, 
                         tf_kmer_length, 
                         tf_ref_data_type=None,
                         tf_ref_data=None,
                         tf_iupac=None,
                         tf_min_score=0.7,
                         snv_effects_to_report='all', 
                         optimization_threshold=1, 
                         suboptimization_threshold=1, 
                         pos2kmerid={}, 
                         zoom_range=None,
                         output_name=None,
                         out_directory='./'):

    # preprocess sequence
    seq = preprocess_seq(seq)

    # get ref data - either pwm, pbm, or handle none
    pwm, seq2aff = load_reference_data(tf_ref_data, tf_ref_data_type)
    
    # create output dataframe
    df = pd.DataFrame(columns=['Sequence Name', 'Kmer ID', 'SNV Position (1-indexed)', 'Reference Nucleotide', 'Alternate Nucleotide', 
                               'Kmer Start Position (1-indexed)', 'Kmer End Position (1-indexed)', 'Reference Kmer', 
                               'Alternate Kmer', 'Site Direction', 'Reference Value', 'Alternate Value', 'Fold Change', 'SNV Effect'])
    
    # iterate over all possible kmers in the seq
    for i in range(len(seq) - tf_kmer_length + 1):

        # get ref kmer and start position 
        start_pos_1idx = i + 1
        ref_kmer = seq[i:i+tf_kmer_length]

        # single kmer
        df = annotate_snvs_in_single_kmer(df = df, 
                                          seq_name = seq_name,
                                          tf_name = tf_name,
                                          kmer_length = tf_kmer_length, 
                                          ref_kmer = ref_kmer, 
                                          start_pos_1idx = start_pos_1idx, 
                                          iupac = tf_iupac, 
                                          seq2aff = seq2aff, 
                                          minscore = tf_min_score, 
                                          pwm = pwm, 
                                          ref_data_type = tf_ref_data_type, 
                                          mut_type = snv_effects_to_report, 
                                          opt_thres = optimization_threshold, 
                                          subopt_thres = suboptimization_threshold, 
                                          pos2kmerid = pos2kmerid)

    # apply zoom boundaries - only keep snvs that are within bounds 
    if zoom_range:
        df = apply_zoom_for_snvs(zoom_range = zoom_range, 
                                 df = df)

    # get output file name
    if zoom_range is not None:
        output_name = output_name + '_table_seq=' + seq_name + '_zoom=' + str(zoom_range[0]) + '-' + str(zoom_range[1]) + '.tsv'
    else:
        output_name =  output_name + '_table_seq=' + seq_name + '.tsv' # distinguish files by seq

    # output to tsv
    return output_tsv(df = df, 
                      out_file = output_name, 
                      out_directory = out_directory)


########################################################*
# create_snv_dict
########################################################*
# helper function list
#    - fill_snv_dict
#    - calc_min_max_snv_effects
#    - initialize_snv_dict
#    - scale_denovo
#    - scale_subopt
#    - scale_opt
#    - scale_alpha
#    - create_snv_dict
########################################################*

def fill_snv_dict(snv_dict, snv_df):
    # color key
    color_dict = {'denovo':'green', 'del':'red', 'inc':'blue', 'dec':'orange'}

    # initialize lists 
    denovo_list, inc_list, dec_list, del_list = [], [], [], []
    snv_list = set()

    # get all snvs with an effect
    effect_df = snv_df[snv_df['SNV Effect'] != 'none']
    
    # use annotated snv effects table to update snv_dict
    for snv_pos_1idx, ref_nt, alt_nt, site_dir, ref_aff, alt_aff, fold_change, snv_effect in zip(effect_df['SNV Position (1-indexed)'],
        effect_df['Reference Nucleotide'], effect_df['Alternate Nucleotide'], effect_df['Site Direction'], effect_df['Reference Value'], 
        effect_df['Alternate Value'], effect_df['Fold Change'], effect_df['SNV Effect']): 

            # convert position coords to be 0-indexed
            snv_pos_0idx = snv_pos_1idx - 1

            # add to list 
            snv_list.add((snv_pos_0idx, alt_nt)) 

            # snv effect coloring
            if snv_effect == 'denovo':
                denovo_list.append(alt_aff)
                if alt_aff == '':
                    alt_aff = 1
                val = alt_aff 
            elif snv_effect == 'inc':
                inc_list.append(fold_change)
                val = fold_change
            elif snv_effect == 'dec':
                dec_list.append(fold_change)
                val = fold_change
            elif snv_effect == 'del':
                val = 1 

            # add to snv dict
            color = color_dict[snv_effect]
            snv_dict[snv_pos_0idx][alt_nt][color] = val

    return snv_dict, denovo_list, inc_list, dec_list, snv_list

def calc_min_max_snv_effects(denovo_list, inc_list, dec_list):
    alpha_stats_dict = {}
    for name,lst in zip(['denovo_list', 'inc_list', 'dec_list'], [denovo_list, inc_list, dec_list]):
        mut = name.split('_')[0]
        if len(lst) != 0:
            alpha_stats_dict[mut] = {'min':min(lst), 'max':max(lst)}
    return alpha_stats_dict

def initialize_snv_dict(snv_df, seq):
    
    # initialize snv dictionary (pos -> nt -> snv effect + alpha) 
    snv_dict = {}
    for i in range(len(seq)): 
        mut_dict = { nt:{} for nt in ['A', 'C', 'G', 'T'] }
        snv_dict[i] = mut_dict 
    
    # fill snv dictionary
    snv_dict, denovo_list, inc_list, dec_list, snv_list = fill_snv_dict(snv_dict, snv_df)

    # calculate max and min for each snv effect 
    alpha_stats_dict = calc_min_max_snv_effects(denovo_list, inc_list, dec_list)

    return snv_dict, snv_list, alpha_stats_dict

def scale_denovo(value, alpha_stats_dict):
    diff = alpha_stats_dict['denovo']['max'] - alpha_stats_dict['denovo']['min']
    if (diff == 0):
        scaled_value = value
    else:  
        scaled_value = (0.9) * ( (value - alpha_stats_dict['denovo']['min']) / (diff) ) 
    scaled_value = max(0.3, scaled_value) # make 0.3 the minimum alpha in snv table
    return scaled_value

def scale_subopt(value, alpha_stats_dict): 
    diff = alpha_stats_dict['dec']['max'] - alpha_stats_dict['dec']['min']
    if diff == 0:
        scaled_value = value
    else:
        scaled_value = 0.9 - ( (0.9) * ( (value - alpha_stats_dict['dec']['min']) / (diff) ) )  
    scaled_value = max(0.3, scaled_value) # make 0.3 the minimum alpha in snv table
    return scaled_value

def scale_opt(value, alpha_stats_dict):
    diff = alpha_stats_dict['inc']['max'] - alpha_stats_dict['inc']['min']
    if diff == 0:
        if (alpha_stats_dict['inc']['max'] > 1): # opt fold change will be above 1 so change it a constant below 1 (so it can still be visualized)
            scaled_value = 0.8 
    else:
        scaled_value = (0.9) * ( (value - alpha_stats_dict['inc']['min']) / (diff) ) 
    scaled_value = max(0.3, scaled_value) # make 0.3 the minimum alpha in snv table
    return scaled_value

def scale_alpha(snv_dict, snv_list, alpha_stats_dict, snv_reference_data):
    
    # normalize values for snvs 
    for snv in snv_list:

        # get position and alt nt to access values in snv_dict
        pos_0idx = snv[0]
        alt_nt = snv[1]

        # get the color and alpha value
        for color,value in snv_dict[pos_0idx][alt_nt].items():

            # scale the alpha value according to snv effect (between 0.3 and 0.9) 
            if (snv_reference_data is None): # account for no pbm/pfm data, set alpha to 1
                scaled_value = 1
                continue 
                
            if color == 'green':
                scaled_value = scale_denovo(value, alpha_stats_dict)
                
            elif color == 'orange':
                scaled_value = scale_subopt(value, alpha_stats_dict)

            elif color == 'blue':
                scaled_value = scale_opt(value, alpha_stats_dict)

            elif color == 'red': # deletions don't need to be scaled bc they are always 1
                scaled_value = 1

            # add back to snv dict 
            snv_dict[pos_0idx][alt_nt][color] = round(scaled_value, 3)

    return snv_dict

def create_snv_dict(snv_df, seq, snv_reference_data):

    # initialize snv dictionary 
    snv_dict, snv_list, alpha_stats_dict = initialize_snv_dict(snv_df = snv_df, 
                                                               seq = seq)
    
    # scale alpha values 
    snv_dict = scale_alpha(snv_dict = snv_dict, 
                           snv_list = snv_list, 
                           alpha_stats_dict = alpha_stats_dict, 
                           snv_reference_data = snv_reference_data)

    return snv_dict


########################################################*
# plot_snv_table
########################################################*
# helper function list
#    - plot_one_box
#    - plot_two_boxes
#    - plot_three_boxes
#    - plot_four_boxes
#    - plot_snv_box
#    - plot_main_table
#    - plot_tens_numline
#    - plot_additional_features
#    - plot_snv_table
########################################################*

# plot box for 1 snv effect
def plot_one_box(x, y, color2alpha, ax):

    # box specs
    width_spacer = 0.1
    height_spacer = 0.15
    total_box_height = 1
    total_box_width = 0.75

    # plot single box
    for color,alpha in color2alpha.items():
        rectangle = patches.Rectangle((x-width_spacer, y-height_spacer), total_box_width, total_box_height, linewidth=0, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rectangle) 

# plot boxes for 2 snv effects
def plot_two_boxes(x, y, color2alpha, ax):

    # box specs
    width_spacer = 0.1
    height_spacer = 0.15
    total_box_height = 1
    total_box_width = 0.75
    half_box_height = total_box_height / 2

    # list of colors in dict
    color_list = list(color2alpha.keys())

    # top box
    first_color = color_list[0]
    first_alpha = color2alpha[first_color]
    rectangle1 = patches.Rectangle((x-width_spacer, y-height_spacer+half_box_height), total_box_width, half_box_height, linewidth=0, edgecolor=first_color, facecolor=first_color, alpha=first_alpha) 
    ax.add_patch(rectangle1)

    # bottom box 
    second_color = color_list[1]
    second_alpha = color2alpha[second_color]
    rectangle2 = patches.Rectangle((x-width_spacer, y-height_spacer), total_box_width, half_box_height, linewidth=0, edgecolor=second_color, facecolor=second_color, alpha=second_alpha) 
    ax.add_patch(rectangle2)

# plot boxes for 3 snv effects
def plot_three_boxes(x, y, color2alpha, ax):

    # box specs
    width_spacer = 0.1
    height_spacer = 0.15
    total_box_height = 1
    total_box_width = 0.75
    third_box_height = total_box_height / 3

    # list of colors in dict
    color_list = list(color2alpha.keys())

    # top box
    first_color = color_list[0]
    first_alpha = color2alpha[first_color]
    rectangle1 = patches.Rectangle((x-width_spacer, y-height_spacer + 2*third_box_height), total_box_width, third_box_height, linewidth=0, edgecolor=first_color, facecolor=first_color, alpha=first_alpha)
    ax.add_patch(rectangle1)

    # middle box 
    second_color = color_list[1]
    second_alpha = color2alpha[second_color]
    rectangle2 = patches.Rectangle((x-width_spacer, y-height_spacer+third_box_height), total_box_width, third_box_height, linewidth=0, edgecolor=second_color, facecolor=second_color, alpha=second_alpha)
    ax.add_patch(rectangle2)

    # bottom box 
    third_color = color_list[2]
    third_alpha = color2alpha[third_color]
    rectangle3 = patches.Rectangle((x-width_spacer, y-height_spacer), total_box_width, third_box_height, linewidth=0, edgecolor=third_color, facecolor=third_color, alpha=third_alpha)
    ax.add_patch(rectangle3)

# plot boxes for 3 snv effects
def plot_four_boxes(x, y, color2alpha, ax):
    
    # box specs
    width_spacer = 0.1
    height_spacer = 0.15
    total_box_height = 1
    total_box_width = 0.75
    quarter_box_height = total_box_height / 4

    # list of colors in dict
    color_list = list(color2alpha.keys())
    
    # 1st box
    first_color = color_list[0]
    first_alpha = color2alpha[first_color]
    rectangle1 = patches.Rectangle((x-width_spacer, y-height_spacer + 3*quarter_box_height), total_box_width, quarter_box_height, linewidth=0, edgecolor=first_color, facecolor=first_color, alpha=first_alpha)
    ax.add_patch(rectangle1)

    # 2nd box 
    second_color = color_list[1]
    second_alpha = color2alpha[second_color]
    rectangle2 = patches.Rectangle((x-width_spacer, y-height_spacer + 2*quarter_box_height), total_box_width, quarter_box_height, linewidth=0, edgecolor=second_color, facecolor=second_color, alpha=second_alpha)
    ax.add_patch(rectangle2)

    # 3rd box 
    third_color = color_list[2]
    third_alpha = color2alpha[third_color]
    rectangle3 = patches.Rectangle((x-width_spacer, y-height_spacer + quarter_box_height), total_box_width, quarter_box_height, linewidth=0, edgecolor=third_color, facecolor=third_color, alpha=third_alpha)
    ax.add_patch(rectangle3)

    # 4th box
    fourth_color = color_list[3]
    fourth_alpha = color2alpha[fourth_color]
    rectangle4 = patches.Rectangle((x-width_spacer, y-height_spacer), total_box_width, quarter_box_height, linewidth=0, edgecolor=fourth_color, facecolor=fourth_color, alpha=fourth_alpha)
    ax.add_patch(rectangle4)

# function to plot box around text 
def plot_snv_box(x, y, text, color2alpha, ax):

    # create rectangle around text, accounting for mixed effects
    if len(color2alpha) == 1: 
        plot_one_box(x, y, color2alpha,  ax)
    elif len(color2alpha) == 2:
        plot_two_boxes(x, y, color2alpha, ax)
    elif len(color2alpha) == 3:
        plot_three_boxes(x, y, color2alpha, ax)
    elif len(color2alpha) == 4:
        plot_four_boxes(x, y, color2alpha, ax)

def plot_main_table(seq, start_pos, end_pos, snv_dict, ax):
    
    # plot text for wildtype sequence and all possible snvs; plot box if there is a snv effect
    for num in range(start_pos, end_pos):
        
        # plot wt nucleotide
        wt_nt = str(seq[num])
        wt_text = ax.text(x=num, y=2.5, s=wt_nt, horizontalalignment='left', weight='bold', fontfamily='monospace', fontsize=13, clip_on=True)
        
        # plot snv nucleotides (in order A,C,G,T with pre-calculated descending heights)
        for curr_nt,height in zip(['A', 'C', 'G', 'T'], [1.25, 0, -1.25, -2.5]):
            if wt_nt != curr_nt:
                nt_text = ax.text(x=num, y=height, s=curr_nt, horizontalalignment='left', fontfamily='monospace', color='grey', fontsize=13, clip_on=True)
                color2alpha = snv_dict[num][curr_nt]
                if snv_dict[num][curr_nt] != {}: 
                    plot_snv_box(x = num, 
                                 y = height, 
                                 text = nt_text, 
                                 color2alpha = color2alpha, 
                                 ax = ax)
    
        # plot ones number line
        ax.text(x=num, y=-3.75, s=str(num+1)[-1], horizontalalignment='left', fontfamily='monospace', fontsize=13, clip_on=True)

def plot_tens_numline(start_pos, end_pos, ax):
    
    # adjust start x coordinate by rounding to nearest 10 
    start_x_1idx = start_pos + 1
    diff = 10 - (start_x_1idx % 10)  
    if diff == 10: 
        diff = 0
    adj_start_x = start_x_1idx + diff

    # plot tens numline
    for num in range(adj_start_x, end_pos+1, 10): 
        ax.text(x=num-1, y=-5, s=str(num), horizontalalignment='left', fontfamily='monospace', fontsize=13, color='grey') # subtract x pos by 1 to account for 0-indexing

def plot_additional_features(start_pos, seq_len, ax):

    # plot line between snvs and number line
    ax.axhline(y=-2.85, xmin=start_pos, xmax=seq_len, linewidth=1, color='black', alpha=0.15)
    
    # plot box highlighting wildtype sequence
    single_box_height = 1.15
    single_box_width = 0.75
    x_spacer = 1
    y_spacer = 0.25
    rectangle = patches.Rectangle((start_pos-x_spacer, 2.5-y_spacer), seq_len+single_box_width, single_box_height, facecolor='lightgrey', alpha=0.25, clip_on=True)
    ax.add_patch(rectangle)

# main function
def plot_snv_table(seq, start_pos, end_pos, snv_dict, plot_dpi=None, fig=None, ax=None, only_table=False):

    # length of seq
    seq_len = end_pos - start_pos + 1

    # set plot specs (if only plotting snv table and not using prior figure for sites)
    if only_table:
        fig,ax = set_plot_specs(seq_length = seq_len, 
                                # plotDims = None,  # not really needed
                                plot_dpi = plot_dpi, 
                                fig = fig, 
                                ax = ax)

    # plot text for wildtype sequence and all possible snvs; plot box if there is a snv effect
    plot_main_table(seq = seq, 
                    start_pos = start_pos, 
                    end_pos = end_pos, 
                    snv_dict = snv_dict, 
                    ax = ax)

    # plot tens numline
    plot_tens_numline(start_pos = start_pos, 
                      end_pos = end_pos, 
                      ax = ax)
    
    # plot additional features
    plot_additional_features(start_pos = start_pos, 
                             seq_len = seq_len, 
                             ax = ax)

    # adjust 
    if only_table:
        
        # adjust figure size to maintain proportions
        ax.set_ylim(-6, 4) # one tfbs unit is 5.4 in height, plus 1 for cushion
        curr_ylim = ax.get_ylim()[1]
        curr_figwidth = fig.get_size_inches()[0]
        new_figheight = (10 * curr_ylim) / 13.4 # first row figheight is y=10 and first row ylim is y=13.4
        fig.set_size_inches(curr_figwidth, new_figheight)

    # cut plot
    ax.set_xlim(start_pos,end_pos)
    
    # fix spacing of letters inside the boxes
    plt.tight_layout(pad=0)

    return fig, ax


########################################################*
# singleSeqVisualizeInSilicoSnvs
########################################################*

def singleSeqVisualizeInSilicoSnvs(seq, 
                                   seq_name,
                                   out_image_format, 
                                   tf_iupacs, 
                                   tf_dfs, 
                                   tf_kmerlens, 
                                   tf_name_for_snv_analysis, 
                                   snv_dict, 
                                   snv_df, 
                                   snv_reference_data_type, 
                                   plot_denovo, 
                                   only_table=False,
                                   plot_dpi=200, 
                                   output_svg=False,
                                   zoom=None, 
                                   plot_window=None, 
                                   output_name=None, 
                                   out_directory='./'):

    # if zoom chosen as out image format (potentially reassign to windows format instead)
    if out_image_format == 'Zoom':
        zoom_length = zoom[1] - zoom[0]
        if zoom is None:
            plot_window = 500
            # plot_window = 1000
            out_image_format == 'Windows'
            print('Warning: Zoom range was not given. Sequence(s) will be plotted in windows.')
        if zoom_length > 500:
            zoom = (zoom[0], zoom[0]+499)
            print('Warning: Zoom range must be less than 500. Output was returned with a zoom range of 500 with the given starting coordinate.')
        if zoom[0] == 0:
            print('Warning: Zoom coordinates must be 1-indexed (numbering starts at 1).')
            
    # if window is chosen as the out image format 
    if out_image_format == 'Windows':
        if plot_window is None:
            plot_window = 500 # default window size
        elif plot_window > 500:
            plot_window = 500
            print('Warning: Number of bases included per plot must be less than 500. Output was returned with a window size of 500.')
    
    # zoom into portion of the seq 
    batch_check = False
    if out_image_format == 'Zoom':

        # subset sequence based on zoom (assume that zoom coords are given as 1-indexed) 
        start_pos = zoom[0] - 1
        end_pos = zoom[1]
        # adj_seq = seq[start_pos:end_pos] 

        # plot sites and then table underneath 
        capacity_tf = 'starting'
        batch_num = 0
        while capacity_tf is not None:
            batch_num += 1

            # plot sites depending on only_table
            if not only_table:
                fig1, ax1, capacity_tf = plot_sites(seq = seq, 
                                                    seq_name = seq_name, 
                                                    start_x = start_pos, 
                                                    end_x = end_pos, 
                                                    tf_iupacs = tf_iupacs, 
                                                    tf_dfs = tf_dfs, 
                                                    tf_kmerlens = tf_kmerlens, 
                                                    tf_name_for_snv_analysis = tf_name_for_snv_analysis, 
                                                    snv_df = snv_df, 
                                                    plot_denovo = plot_denovo, 
                                                    plot_dpi = plot_dpi) 
            else:
                fig1, ax1, capacity_tf = None, None, None
    
            # plot snv table
            fig2, ax2 = plot_snv_table(seq = seq, 
                                       start_pos = start_pos, 
                                       end_pos = end_pos, 
                                       snv_dict = snv_dict, 
                                       plot_dpi = plot_dpi, 
                                       fig = fig1, 
                                       ax = ax1, 
                                       only_table = only_table)
    
            # output plot
            if output_name is not None:
                new_outfile = output_name + '_image_seq=' + seq_name + '_zoom=' + str(start_pos+1) + '-' + str(end_pos) + '.png' # add 1 to start pos to revert back to 1-indexed 
                if out_directory != './':
                    out_directory = out_directory.strip('/') # get rid of trailing slashes
                    new_outfile = out_directory + '/' + new_outfile
                fig2.savefig(new_outfile, bbox_inches='tight')
                print(new_outfile + ' has been created') 
    
                if output_svg:
                    svg_outfile = new_outfile[:-3] + 'svg'
                    fig2.savefig(svg_outfile, bbox_inches='tight', pad_inches=0)
                    print(svg_outfile + ' has been created')

            # trim dictionaries
            if capacity_tf is not None:

                # images printed in batches
                batch_check = True

                # create new tf_iupacs dict 
                pos = 0
                for i,tfac in enumerate(tf_iupacs):
                    if tfac == capacity_tf:
                        pos = i
                        break
                tf_iupacs_trimmed_keys = list(tf_iupacs.keys())[pos+1:]
                new_tf_iupacs = {id:tf_iupacs[id] for id in tf_iupacs_trimmed_keys}
                tf_iupacs = new_tf_iupacs
    
                # create new tf_dfs dict
                pos = 0
                for i,tfac in enumerate(tf_dfs):
                    if tfac == capacity_tf:
                        pos = i
                        break
                tf_dfs_trimmed_keys = list(tf_dfs.keys())[pos+1:]
                new_tf_dfs = {id:tf_dfs[id] for id in tf_dfs_trimmed_keys}
                tf_dfs = new_tf_dfs
    
                # create new tf_kmerlens dict
                pos = 0
                for i,tfac in enumerate(tf_kmerlens):
                    if tfac == capacity_tf:
                        pos = i
                        break
                tf_kmerlens_trimmed_keys = list(tf_kmerlens.keys())[pos+1:]
                new_tf_kmerlens = {id:tf_kmerlens[id] for id in tf_kmerlens_trimmed_keys}
                tf_kmerlens = new_tf_kmerlens
        
    
    # new figure plotted for size of plot window (min = 500 bp) 
    if out_image_format == 'Windows':    

        # get window list
        window_list = [i for i in range(0, len(seq), plot_window)] + [len(seq)]

        # plot sites for each window
        all_vis_files = []
        for i in range(len(window_list[:-1])):
    
            # take window of seq 
            start_pos = window_list[i]
            end_pos = window_list[i+1]
    
            # plot sites and then table underneath 
            capacity_tf = 'starting'
            batch_num = 0
            while capacity_tf is not None:
                batch_num += 1

                # plot sites depending on only_table
                if not only_table:
                    fig1, ax1, capacity_tf = plot_sites(seq = seq, 
                                                        seq_name = seq_name, 
                                                        start_x = start_pos, 
                                                        end_x = end_pos, 
                                                        tf_iupacs = tf_iupacs, 
                                                        tf_dfs = tf_dfs, 
                                                        tf_kmerlens = tf_kmerlens, 
                                                        tf_name_for_snv_analysis = tf_name_for_snv_analysis, 
                                                        snv_df = snv_df, 
                                                        plot_denovo = plot_denovo, 
                                                        plot_dpi = plot_dpi) 
                else:
                    fig1, ax1, capacity_tf = None, None, None
    
                # plot snv table
                fig2, ax2 = plot_snv_table(seq = seq, 
                                           start_pos = start_pos, 
                                           end_pos = end_pos, 
                                           snv_dict = snv_dict, 
                                           plot_dpi = plot_dpi, 
                                           fig = fig1, 
                                           ax = ax1, 
                                           only_table = only_table)
    
                # output plot
                if output_name is not None:
                    new_outfile = output_name + '_image_seq=' + seq_name + '_pos=' + str(start_pos+1) + '-' + str(end_pos) + '.png' # add 1 to start pos to revert back to 1-indexed 
                    if out_directory != './':
                        out_directory = out_directory.strip('/') # get rid of trailing slashes
                        new_outfile = out_directory + '/' + new_outfile
                    fig2.savefig(new_outfile, bbox_inches='tight')
                    print(new_outfile + ' has been created')
                    all_vis_files.append(new_outfile)
    
                    if output_svg:
                        svg_outfile = new_outfile[:-3] + 'svg'
                        fig2.savefig(svg_outfile, bbox_inches='tight', pad_inches=0)
                        print(svg_outfile + ' has been created')

                # trim dictionaries
                if capacity_tf is not None:

                    # images printed in batches
                    batch_check = True
    
                    # create new tf_iupacs dict 
                    pos = 0
                    for i,tfac in enumerate(tf_iupacs):
                        if tfac == capacity_tf:
                            pos = i
                            break
                    tf_iupacs_trimmed_keys = list(tf_iupacs.keys())[pos+1:]
                    new_tf_iupacs = {id:tf_iupacs[id] for id in tf_iupacs_trimmed_keys}
                    tf_iupacs = new_tf_iupacs
        
                    # create new tf_dfs dict
                    pos = 0
                    for i,tfac in enumerate(tf_dfs):
                        if tfac == capacity_tf:
                            pos = i
                            break
                    tf_dfs_trimmed_keys = list(tf_dfs.keys())[pos+1:]
                    new_tf_dfs = {id:tf_dfs[id] for id in tf_dfs_trimmed_keys}
                    tf_dfs = new_tf_dfs
        
                    # create new tf_kmerlens dict
                    pos = 0
                    for i,tfac in enumerate(tf_kmerlens):
                        if tfac == capacity_tf:
                            pos = i
                            break
                    tf_kmerlens_trimmed_keys = list(tf_kmerlens.keys())[pos+1:]
                    new_tf_kmerlens = {id:tf_kmerlens[id] for id in tf_kmerlens_trimmed_keys}
                    tf_kmerlens = new_tf_kmerlens

        # create stitched image 
        if len(seq) > 500:
            outfile = output_name + '_image_seq=' + seq_name + '_full_range' + '.png'
            if out_directory != './':
                out_directory = out_directory.strip('/') # get rid of trailing slashes
                outfile = out_directory + '/' + outfile
            append_images_with_top_border(all_vis_files, outfile)

    # close plot 
    plt.close('all')
    
    # if outfile is not None:
    #     return new_outfile


########################################################*
# singleSeqAnnotateAndVisualizeInSilicoSnvs
########################################################*

def singleSeqAnnotateAndVisualizeInSilicoSnvs(seq, 
                                              seq_name, 
                                              tf_name,
                                              tf_kmer_length,
                                              tf_ref_data_type,
                                              tf_ref_data, 
                                              tf_iupac,
                                              tf_min_affinity,
                                              tf_min_score,
                                              tf_pseudocounts,
                                              snv_effects_to_report, 
                                              plot_denovo_sites,
                                              optimization_threshold, 
                                              suboptimization_threshold, 
                                              only_snv_table, 
                                              plot_resolution,
                                              output_svg,
                                              out_image_format,
                                              zoom_range,
                                              window_size,
                                              output_name,
                                              out_directory):

    # create annotated df for tf -> to use for plotting sites    
    df = singleSeqAnnotateTfSites(seq = seq, 
                                  seq_name = seq_name, 
                                  tf_name = tf_name, 
                                  tf_kmer_length = tf_kmer_length,
                                  tf_ref_data_type = tf_ref_data_type,
                                  tf_ref_data = tf_ref_data, 
                                  tf_iupac = tf_iupac, 
                                  tf_min_affinity = tf_min_affinity,
                                  tf_min_score = tf_min_score, 
                                  tf_pseudocounts = tf_pseudocounts, 
                                  zoom_range = zoom_range)
    
    # create kmer id dictionary from annotateTfSites so it can be used/expanded in annotateInSilicoSnvs
    pos2kmerid = {pos:id for pos,id in zip(df['Start Position (1-indexed)'], df['Kmer ID'])}
    
    # create in silico snvs table
    snv_df = annotateInSilicoSnvs(seq = seq, 
                                  seq_name = seq_name, 
                                  tf_name = tf_name, 
                                  tf_kmer_length = tf_kmer_length, 
                                  tf_ref_data_type = tf_ref_data_type,
                                  tf_ref_data = tf_ref_data,
                                  tf_iupac = tf_iupac, 
                                  tf_min_score = tf_min_score, 
                                  snv_effects_to_report = snv_effects_to_report, 
                                  optimization_threshold = optimization_threshold, 
                                  suboptimization_threshold = suboptimization_threshold, 
                                  pos2kmerid = pos2kmerid, 
                                  zoom_range = zoom_range, 
                                  output_name = output_name,
                                  out_directory = out_directory)

    # create snv dict
    snv_dict = create_snv_dict(snv_df = snv_df, 
                               seq = seq, 
                               snv_reference_data = tf_ref_data)

    # # get name of output visualization file
    # vis_out_file = 'visualizeInSilicoSnvs-image_seq=' + seq_name + '.png' # distinguish files by seq 

    # create tf dicts for singleSeqVisualizeInSilicoSnvs input
    tf_iupacs = {tf_name:tf_iupac}
    tf_dfs = {tf_name:df}
    tf_kmerlens = {tf_name:tf_kmer_length}

    # visualize sites for all tfs on this seq 
    if len(snv_df) != 0: 
        return singleSeqVisualizeInSilicoSnvs(seq = seq, 
                                              seq_name = seq_name,
                                              out_image_format = out_image_format, 
                                              tf_iupacs = tf_iupacs,
                                              tf_dfs = tf_dfs,
                                              tf_kmerlens = tf_kmerlens,
                                              tf_name_for_snv_analysis = tf_name,
                                              snv_dict = snv_dict,
                                              snv_df = snv_df,
                                              snv_reference_data_type = tf_ref_data_type,
                                              plot_denovo = plot_denovo_sites,
                                              only_table = only_snv_table,
                                              plot_dpi = plot_resolution,
                                              output_svg = output_svg,
                                              zoom = zoom_range,
                                              plot_window = window_size, 
                                              output_name = output_name,
                                              out_directory = out_directory)
    else: 
        return "No binding sites found..." 


########################################################*
# annotateAndVisualizeInSilicoSnvs_dict_inputs
########################################################*

def annotateAndVisualizeInSilicoSnvs_dict_inputs(dna_sequences_to_annotate,
                                                 tf_name,
                                                 tf_binding_site_definition=None,
                                                 tf_aff_data=None,
                                                 # tf_reference_data=None,
                                                 tf_min_affinity=0,
                                                 tf_pwm_data=None,
                                                 tf_min_score=0.7,
                                                 tf_pseudocounts=False,
                                                 snv_effects_to_report='all',
                                                 plot_denovo_sites=False,
                                                 optimization_threshold=1,
                                                 suboptimization_threshold=1,
                                                 only_snv_table=True,
                                                 plot_resolution=150,
                                                 output_svg=False,
                                                 zoom_range=None,
                                                 output_name=None,
                                                 out_directory="./"):
    '''
    Reports the effects of all possible single-nucleotide variants on the TF binding sites in a sequence.\n
    Parameters: \n
    - dna_sequences_to_annotate (.tsv): File containing one or more DNA sequences to be annotated.\n
        - columns (strict order, flexible names) -> sequence name, sequence \n
    - tf_name (string): Name of the transcription factor to use for SNV analysis. \n
    - tf_binding_site_definition (string, default = None): IUPAC definition of core TF binding site. Only optional if using PFM data but required if using affinity data. \n
    - tf_aff_data (string, default = None): PBM affinity dataset used to assign value to each binding site. \n
    - tf_min_affinity (float, default = 0): Affinity threshold required to predict a binding site. \n
    - tf_pfm_data (string, default = None): PFM affinity dataset used to assign value to each binding site. \n
    - tf_min_score (float, default = 0.7): PFM score required to predict a binding site. \n
    - tf_pseudocounts (boolean, default = False): Whether to use pseudocounts when calculating PFM score. \n
    - snv_effects_to_report (comma-separated string, default = all): Specify one or more mutation types to analyze. The possible mutation types are `inc`, `dec`, `denovo`, and `del`. By default, this value is `all` and analyzes all of the listed mutation types.\n
    - plot_denovo_sites (boolean, default = False): If `True`, plot the binding sites that would be created from denovo SNVs, in addition to existing binding sites. If `False`, only plot existing binding sites.\n
    - optimization_threshold (float, default = 1): Only SNVs with a fold change above this threshold will be reported. By default, all SNVs will be reported.\n
    - suboptimization_threshold (float, default = 1): Only SNVs with a fold change below this threshold will be reported. By default, all SNVs will be reported.\n
    - only_snv_table (boolean, default = True): If `True`, only print the table containing SNV effects with no plotting binding site. If `False`, plot both binding sites and SNV table.  \n
    - plot_resolution (integer, default = 200): Resolution of the plot, in dots (pixels) per inch. Manuscripts require 300 DPI. The DPI does not affect the resolution of .svg files. \n
    - output_svg (boolean, default = False): Option to output images as .svg in addition to .png. For manuscript preparation, .svg format is preferable. \n
    - zoom_range (tuple, default = None): Indicates the region of the DNA sequence to visualize, given a start and end coordinate. The numbers in the range are inclusive and 1-indexed. For example, the first 200 nucleotides of the sequence would be specified as: (1,200) \n
    - out_directory (string, default = ./): Directory to contain all output files \n
    '''

    # check that inputs are valid
    if tf_binding_site_definition is not None: check_nt_iupac(tf_binding_site_definition)
    if (type(tf_min_affinity) != float) and (type(tf_min_affinity) != int): raise ValueError('tf_min_affinity must be a valid number')
    if (tf_min_affinity < 0) or (tf_min_affinity > 1): raise ValueError('tf_min_affinity must be between 0 and 1')
    if (type(tf_min_score) != float): raise ValueError('tf_min_score must be a valid float')
    if (tf_min_score < 0) or (tf_min_score > 1): raise ValueError('tf_min_score must be between 0 and 1')
    tf_pseudocounts = check_bool(tf_pseudocounts)
    for snv_effect in snv_effects_to_report.split(','):
        if snv_effect not in ['inc', 'dec', 'del', 'denovo', 'all']:
            raise ValueError('snv_effects_to_report must only contain one or more values in [inc, dec, del, denovo, all]')
    plot_denovo_sites = check_bool(plot_denovo_sites)
    if (type(optimization_threshold) != float) and (type(optimization_threshold) != int): raise ValueError('optimization_threshold must be a valid number')
    if (type(suboptimization_threshold) != float) and (type(suboptimization_threshold) != int): raise ValueError('suboptimization_threshold must be a valid number')
    only_snv_table = check_bool(only_snv_table)
    if type(plot_resolution) != int: raise ValueError('plot_resolution must be a valid integer')
    output_svg = check_bool(output_svg)
    if type(zoom_range) != tuple and (zoom_range is not None): raise ValueError('zoom_range must be a valid tuple.')

    # get output name, just in case file extension is included in the file name by the user
    output_name = output_name.split('.')[0]

    # # set error for out_image_format
    # if out_image_format is None:
    #     raise ValueError('The value of out_image_format must be either “Zoom” or “Windows.”')

    # handle reference data
    if (tf_aff_data is None) and (tf_pwm_data is None):
        tf_ref_data_type = None
        tf_reference_data = None
    elif (tf_aff_data is not None) and (tf_pwm_data is None):
        tf_ref_data_type = 'Affinity'
        tf_reference_data = loadNormalizedFile(tf_aff_data)
    elif (tf_aff_data is None) and (tf_pwm_data is not None):
        tf_ref_data_type = 'Score'
        tf_reference_data = pwm_to_pwmobj(pwm_input=tf_pwm_data)
    elif (tf_aff_data is not None) and (tf_pwm_data is not None):
        raise ValueError('Only one reference data file can be given, not both tf_aff_data and tf_pfm_data.')

    # check that ref data is present for no iupac
    if (tf_binding_site_definition is None) and (tf_reference_data is None):
        raise ValueError('If no binding site definition is given, then there must be reference data. Alternatively, if there is no reference data, then the binding site definition must be given.')
    
    # # get ref data type and process reference data 
    # if tf_reference_data != None:
    #     with open(tf_reference_data) as file:
    #         if file.readline()[0] == '>':
    #             tf_ref_data_type = 'Score'
    #             tf_reference_data = pwm_to_pwmobj(pwm_input=tf_reference_data)
    #         else:
    #             tf_ref_data_type = 'Affinity'
    #             tf_reference_data = loadNormalizedFile(tf_reference_data)
    # else:
    #     tf_ref_data_type = None

    # get kmer length 
    tf_kmer_length = get_kmer_length(tf_iupac = tf_binding_site_definition, 
                                     tf_ref_data_type = tf_ref_data_type, 
                                     tf_ref_data = tf_reference_data)

    # set out_image_format
    if zoom_range is not None:
        out_image_format = 'Zoom'
    else:
        out_image_format = 'Windows'

    # iterate through each seq
    with open(dna_sequences_to_annotate) as input_file:
        input_file.readline()
        input_file = csv.reader(input_file, delimiter='\t')
        for line in input_file: 

            # obtain seq and seq_name
            seq_name = line[0]
            seq = line[1]

            # run annotate and visualize on one sequence
            singleSeqAnnotateAndVisualizeInSilicoSnvs(seq = seq, 
                                                      seq_name = seq_name, 
                                                      tf_name = tf_name,
                                                      tf_kmer_length = tf_kmer_length,
                                                      tf_ref_data_type = tf_ref_data_type,
                                                      tf_ref_data = tf_reference_data,
                                                      tf_iupac = tf_binding_site_definition,
                                                      tf_min_affinity = tf_min_affinity,
                                                      tf_min_score = tf_min_score,
                                                      tf_pseudocounts = tf_pseudocounts,
                                                      snv_effects_to_report = snv_effects_to_report, 
                                                      plot_denovo_sites = plot_denovo_sites,
                                                      optimization_threshold = optimization_threshold, 
                                                      suboptimization_threshold = suboptimization_threshold, 
                                                      only_snv_table = only_snv_table, 
                                                      plot_resolution = plot_resolution,
                                                      output_svg = output_svg,
                                                      out_image_format = out_image_format,
                                                      zoom_range = zoom_range,
                                                      window_size = None,
                                                      output_name = output_name,
                                                      out_directory = out_directory)
            
# final function name for tfsites website
def findTfSitesAlteredBySequenceVariation(dna_sequences_to_annotate,
                                            tf_name,
                                            tf_binding_site_definition=None,
                                            tf_aff_data=None,
                                            tf_min_affinity=0,
                                            tf_pwm_data=None,
                                            tf_min_score=0.7,
                                            tf_pseudocounts=False,
                                            snv_effects_to_report='all',
                                            plot_denovo_sites=False,
                                            optimization_threshold=1,
                                            suboptimization_threshold=1,
                                            only_snv_table=True,
                                            plot_resolution=150,
                                            output_svg=False,
                                            zoom_range=None,
                                            output_name=None,
                                            out_directory="./"):
    
    annotateAndVisualizeInSilicoSnvs_dict_inputs(dna_sequences_to_annotate=dna_sequences_to_annotate,
                                                 tf_name=tf_name,
                                                 tf_binding_site_definition=tf_binding_site_definition,
                                                 tf_aff_data=tf_aff_data,
                                                 tf_min_affinity=tf_min_affinity,
                                                 tf_pwm_data=tf_pwm_data,
                                                 tf_min_score=tf_min_score,
                                                 tf_pseudocounts=tf_pseudocounts,
                                                 snv_effects_to_report=snv_effects_to_report,
                                                 plot_denovo_sites=plot_denovo_sites,
                                                 optimization_threshold=optimization_threshold,
                                                 suboptimization_threshold=suboptimization_threshold,
                                                 only_snv_table=only_snv_table,
                                                 plot_resolution=plot_resolution,
                                                 output_svg=output_svg,
                                                 zoom_range=zoom_range,
                                                 output_name=output_name,
                                                 out_directory=out_directory)


###################################################################################################
# 05 and 06 - Compare TF Sites Across Sequences From DNA Sequences and Genomic Variants 
###################################################################################################

#############################################################################
# copied from various js packages
#############################################################################

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], 
distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

def fdr_correction(pvals, alpha=0.05, method='indep'):
    """P-value correction with False Discovery Rate (FDR).
    Correction for multiple comparison using FDR :footcite:`GenoveseEtAl2002`.
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.
    Parameters
    ----------
    pvals : array_like
        Set of p-values of the individual tests.
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.
    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
        P-values adjusted for multiple hypothesis testing to limit FDR.
    References
    ----------
    .. footbibliography::
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return [min(1,p) for  p in pvals_corrected]

def mkdir_if_dir_not_exists(out_dir):
    '''Make a directory only if that directory doesnt exist'''
    if not os.path.exists(out_dir): os.mkdir(out_dir)

def clearspines(ax,sides=['top','right']):
    for s in sides:
        ax.spines[s].set_visible(False)

def clearticks(ax,sides=['x','y']):
    if 'x' in sides: ax.set_xticks([])
    if 'y' in sides: ax.set_yticks([])

def quickfig(x=5,y=5,dpi=150):
    return plt.subplots(1,figsize=(x,y),dpi=dpi,facecolor='white')

def zipdf(df,cols):
    return zip(*[df[c] for c in cols])

def read_tsv(fn,pc,header,breakBool=False,sep='\t',pc_list=False):
    '''Read a tsv file'''
    with open(fn,'r') as f:
        
        # If printing columns, skip header
        if header:
            if (pc==True): pass
            else:          next(f)
                
        for i,line in enumerate(f):
            a=line.strip().split(sep)
            if pc:
                if pc_list==False:
                    if i==0:
                        for i,c in enumerate(a):
                            print(i,c)
                        print()
                        if breakBool: break
                        continue
                else:
                    if i==0:
                        print(', '.join([i.replace('-','_').replace(' ','_') for i in a]))
                        if breakBool: break
                        continue
            yield a
            
def read_in_chunks(file_path, chunk_size=4):
    """
    Generator to read a file in chunks of lines.
    
    Parameters:
    file_path (str): Path to the file to be read.
    chunk_size (int): Number of lines to read at a time (default is 4).
    
    Yields:
    list: A list of lines.
    """
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
            
Iupac2AllNt= {
        'A':['A'],
        'C':['C'],
        'G':['G'],
        'T':['T'],
        'R':['A','G'],
        'Y':['C','T'],
        'S':['G','C'],
        'W':['A','T'],
        'K':['G','T'],
        'M':['A','C'],
        'B':['C','G','T'],
        'D':['A','G','T'],
        'H':['A','C','T'],
        'V':['A','C','G'],
        'N':['A','C','G','T'],
}

ATGC=set(['A','T','G','C'])

def revcomp(dna): 
	'''Takes DNA sequence as input and returns reverse complement'''
	inv={'A':'T','T':'A','G':'C','C':'G', 'N':'N','W':'W','-':'-'}
	revcomp_dna=[]
	for nt in dna:
		revcomp_dna.append(inv[nt])
	return ''.join(revcomp_dna[::-1])

def IupacToRegexPattern(dna):
	'''Takes DNA with IUPAC letters and returns a regex object that can search DNA with only A/T/G/C for the corresponding IUPAC DNA string.'''
	p=''
	for iupac in dna:
		if iupac in ['A','T','G','C']:
			p+=iupac
		else:
			p+='('
			for nt in Iupac2AllNt[iupac]:
				p+=nt
				p+='|'
			p=p[:-1]
			p+=')'
	return p


def hamming(str1, str2):
	'''Takes 2 strings as input and returns hamming distance'''
	if len(str1) != len(str2):
		raise ValueError("Strand lengths are not equal!")
	else:
		return sum(1 for (a, b) in zip(str1, str2) if a != b)


##############################################################################
# Maggies helper functions
##############################################################################

seqDict = {'A':0, 'C':1, 'G':2, 'T': 3}
background_Freq= [0.25, 0.25, 0.25, 0.25]

def uniprobe_parser(input_pfm_jaspar):
    #check the file format
    '''
    Warning: The input file must have the same format if have more than one PWMs in the order of ACGT. Ensure the TF name is included in the file header (Can either have 5 or 6 lines)
    '''
    filename = os.path.basename(input_pfm_jaspar)
    motif_name = os.path.splitext(filename)[0].split('_')[0]
    motifs = {}
    with open (input_pfm_jaspar, 'r') as f:
        motif_id = []
        is_number = 0
        for line in f:
            if line.strip() == "":
                continue
        
            if len(re.findall(r'[A-Za-z]', line)) < 2:
                if is_number == 0:
                    motif_id = motif_name+' '+' '.join(motif_id)
                if (motif_id, motif_name) not in motifs:
                    motifs[(motif_id, motif_name)] = []
                motifs[(motif_id, motif_name)].append(list(map(float, re.findall(r'-?\d*\.\d+|-?\d+|[-+]?inf', line))))
                
                is_number+=1
                if is_number == 4:
                    is_number = 0
                    motif_id = []
            else: 
                motif_id.append(line.strip())
    return motifs

def motif_parser(input_pfm_jaspar, pwm_file_format = 'jaspar'):
    motifs = {}
    if pwm_file_format == 'uniprobe':
        return uniprobe_parser(input_pfm_jaspar)
    with open (input_pfm_jaspar, 'r') as f:
        current_motif = None
        count = 1
        for line in f:
            
            if line[0]=='>':
              
                header_info = line.strip().replace('>', '').split()
                #if len(header_info) == 2:
                if re.match(r'^MA\d{4}\.\d+$', header_info[0]):
                    motif_id = header_info[0] 
                    motif_name = header_info[1] if len(header_info) > 1 else header_info[0]
                #elif the input is from hocomoco
                elif re.search(r'([^.\s]+\.){2,}[^.\s]+',  line):
                    header = header_info[0].split('.')
                    motif_name = header[0]
                    motif_id = line.strip()
                    count += 1
                else:
                    raise ValueError('Input .jaspar PWM file is not from Jaspar or Hocomoco.')
                motif_name  = motif_name.strip('>').strip()
                motif_id = motif_id.strip('>').strip()
                motifs[(motif_id, motif_name)] = []
                current_motif = (motif_id, motif_name)
                continue

            numbers = list(map(float, re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf', line)))
            # numbers = list(map(float, line.strip('ACGT []\n').split()))
            motifs[current_motif].append(numbers)

    # make all pwms into numpy arrays
    array_motifs = {}
    for motif,numbers in motifs.items():
        array_motifs[motif] = np.array(numbers)
    return array_motifs

            
def get_trimmed_dict(input_pwm, trimStart, trimEnd):
    '''Trims pwm to desired start and end (0-idx)
    
    Parameters
    - trimStart - if you want to only use a subset of the bases in the pfm, this is the 0idx start of that trimmed pfm
    - trimEnd - end of the trimmed pfm 0idx'''
    trimmed_pwm = input_pwm[:, trimStart:trimEnd+1]
    return trimmed_pwm

def calculate_pseudocounts(input_pfm, k_len):
    '''
    Calculate pseudocounts using the product of square root of the 
    column averages and the nucleotide's background frequency.
    return a dictionary of pseudocounts as {'A':0.0,'C':0.0, 'G':0.0,'T':0.0}
    '''
    total= sum(sum(row) for row in input_pfm)

    ave_countsPcol = total/k_len
    sqrt_ave = math.sqrt(ave_countsPcol)
    pseudocounts = []
    for idx in range(4):
        pseudocounts.append(background_Freq[idx]*sqrt_ave)
    return pseudocounts

def get_FracMatrix_from_Jaspar(input_pfm, isFraction, pseudocounts=False):
    '''
    Convert Count matrix to PWM and return the PWM as np array
    '''
    # Convert counts matrix to PFM
    # Calculate pseudocounts if not given
    if isFraction and pseudocounts==False:
        return input_pfm
    k_len = len(input_pfm[0])
    PseudoCt = [0.0, 0.0, 0.0, 0.0]#{'A':0.0,'C':0.0, 'G':0.0,'T':0.0}
    if pseudocounts == True:
        PseudoCt = calculate_pseudocounts(input_pfm, k_len)

    counts_matrix = np.array(input_pfm)
    counts_matrix += np.array([PseudoCt[idx] for idx in range(4)]).reshape(-1,1)
    totalOb = counts_matrix.sum(axis=0)
    pfm = counts_matrix/totalOb
    return pfm
    
def get_PWM_from_PFM(input_pfm, isFraction, pseudocounts=False):
    '''
    Convert Count matrix to PWM and return the PWM as np array
    '''
    # Convert counts matrix to PFM
    # Calculate pseudocounts if not given
    #input_pfm = {nuc:input_pfm[i] for i, nuc in enumerate('ACGT')}
    if isFraction and pseudocounts:
        input_pfm = (np.array(input_pfm)*1e3).tolist()
        #input_pfm = {nuc:temp_mat[i].tolist() for i, nuc in enumerate('ACGT')}

    pfm = get_FracMatrix_from_Jaspar(input_pfm, isFraction, pseudocounts)

    # convert PFM to PWM
    bgFreq=np.array(background_Freq)
    pwm = np.log2(np.divide(pfm, bgFreq[:, np.newaxis]))
    
    return pwm

def get_pwm_python_object(input_pfm_jaspar_formatted, isAlreadyPwm, isFraction, trimStart=None, trimEnd=None, pseudocounts=False, pwm_file_format='jaspar'):
    '''
    get_pfm_python_object takes an input pfm and outputs a pythonic object to score kmers
    
    Parameters:
    - input_pfm_jaspar_formatted - pfm formatted from jaspar
    - trimStart - if you want to only use a subset of the bases in the pfm, this is the 0idx start of that trimmed pfm
    - trimEnd - end of the trimmed pfm 0idx
    - pseudocounts - if you want to use Jaspar pseudocounts, set this to True

    Returns: returns python object that represents the pfm
    '''
    #with open(input_pfm_jaspar_formatted) as handle:
       # motif = motifs.parse(handle, "jaspar")
    motif = motif_parser(input_pfm_jaspar_formatted, pwm_file_format=pwm_file_format)

    if isAlreadyPwm:    
        pwm= np.array(list(motif.values())[0]) #np.array([motif[0].counts[nt] for nt in 'ACGT'])
    else:               
        pwm= get_PWM_from_PFM(list(motif.values())[0], isFraction, pseudocounts)        
        
    if trimStart==None and trimEnd==None:
        return pwm
    
    else:
        trimmed_pwm = pwm[:, trimStart:trimEnd+1]
        return trimmed_pwm     
                
def get_pwm_python_object_batch(input_pfm_jaspar_formatted_batch_file, isAlreadyPwm, isFraction, pseudocounts=False, pwm_file_format='jaspar'):
    '''
    get_pfm_python_object takes an input pfm and outputs a pythonic object to score kmers
    
    Parameters:
    - input_pfm_jaspar_formatted_batch_file - pfm batch file from jaspar
    - pseudocounts - if you want to use jaspar pseudocounts, set this to True

    Returns: returns python object that represents the pfms
    '''
    #with open(input_pfm_jaspar_formatted_batch_file) as handle:
        #motif_s = motifs.parse(handle, "jaspar")
    motif_s = motif_parser(input_pfm_jaspar_formatted_batch_file, pwm_file_format=pwm_file_format)
    pwmObj = {}
    for matrix_info, matrix in motif_s.items():
        if isAlreadyPwm: 
            pwmObj[matrix_info] = np.array(matrix)
            if isFraction:
                print('[WARNING] isFraction is ignored when a motif database is inputted as PWM.')
                print('[WARNING] pseudocounts is ignored when a motif database is inputted as PWM.')
        else:            pwmObj[matrix_info] = get_PWM_from_PFM(matrix, isFraction, pseudocounts)
    return pwmObj

##############################################################################
# Joes helper functions 
##############################################################################

def filter_pwm_file_keywords(tfKeywordList,in_pfm_file,out_pfm_file):
    
    tfKeywordList=[i.upper() for i in tfKeywordList]
    line_out=''
    for pfm in read_in_chunks(in_pfm_file,5):
        
        header,_,_,_,_=pfm
        header=header.upper()
        
        for tf in tfKeywordList:
            if tf in header:
                line_out+='\n'.join(pfm)
                line_out+='\n'
                
    with open(out_pfm_file,'w') as f: f.write(line_out)
    
def fasta_to_tsv(fasta_file, tsv_file):
    """
    Convert a FASTA file to a TSV file with sequence ID and sequence.

    Parameters:
    fasta_file (str): Path to the input FASTA file.
    tsv_file (str): Path to the output TSV file.
    """
    # Read the sequences from the FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    
    # Create a list of dictionaries with sequence ID and sequence
    data = [{"id": seq.id, "sequence": str(seq.seq)} for seq in sequences]
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a TSV file
    df.to_csv(tsv_file, sep='\t', index=False)
    
def tsv_to_fasta(tsv_in,fasta_out):
    '''Converts tsv file to fasta file.
    
    Parameters:
    - input_enhancer_fn - input file (tsv header) with first two columns being enhancer name and sequence'''
    
    line_out=''
    for row in read_tsv(tsv_in,pc=False,header=True):    
        name,seq=row[0],row[1].upper()
        name=name.replace(' ','_')
        line_out+=f'>{name}\n{seq}\n'
    with open(fasta_out,'w') as f: f.write(line_out)
    
def read_in_chunks(file_path, chunk_size=4):
    """
    Generator to read a file in chunks of lines.
    
    Parameters:
    file_path (str): Path to the file to be read.
    chunk_size (int): Number of lines to read at a time (default is 4).
    
    Yields:
    list: A list of lines.
    """
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def filter_batch_pfm_file(keywordList,in_batch_pfm,out_batch_pfm_filtered):
    '''filter_batch_pfm_file will output a subset of the pfms in which the tf name contains one of the kewords provided
    in the keword list
    
    Parameters:
     - keywordList - list - kewords that if found in pfm header the pfm is written to the filtered file
     - in_batch_pfm - file containing all pfm. Formatted similar to fasta where header line is ">"
       and the matrix of that pfm is directly below.
     - out_batch_pfm_filtered - name of file to output filtered batch pfm.'''
    
    line_out=''
    for chunk in read_in_chunks(in_batch_pfm, chunk_size=5):
        header=chunk[0]
        for ki in keywordList:
            if ki.upper() in header.upper():
                line_out+='\n'.join(chunk)+'\n'
    with open(out_batch_pfm_filtered,'w') as f: f.write(line_out)
    
def get_lowercase_diffs(wt,dna):
    if len(wt)!=len(dna): raise ValueError(f'this shouldnt happen wt={wt}, dna={dna}')
    out=''
    for wi,di in zip(wt,dna):
        
        if wi==di: out+=di.upper()
        else:      out+=di.lower()
    return out
    
##############################################################################
# Joes compare seqs used in optimized  AC analysis 
##############################################################################

def trim_alignment_tsv_file(infile,grouplabelfile,outfile,start0idx,end0idx,removeSameAsWt=False):
    '''trims alignemnt of file
    
    Parameters:
     - infile (str) - input tsv file with columns (in order) name, alignment, group. Must have header.
     - outfile (str) - name ouf output tsv file
     - start0idx (int) - 0-index of start position to trim
     - end0idx (int) - 0-index of end position to trim'''
    
    # get group labels
    group2label={}
    with open(grouplabelfile,'r') as f:
        # skip header
        f.readline()
        for l in f: 
            l=l.strip().split('\t')
            # print(l)
            group,label=l
            group2label[group]=label
            
    #
    groupCol=3
    groupColIndex=groupCol-1
    # get wt sequence
    if removeSameAsWt:
        wtfound=False
        for row in read_tsv(infile,pc=False,header=True):
            name,al=row[:2]
            group=row[groupColIndex]
            label=group2label[group]
            al=al[max(start0idx,0):end0idx]
            if label=='wild-type': 
                wtseq=al
                wtfound=True
            break
        if wtfound==False: raise ValueError('if removeSameAsWt is True, then there must be an alignment labeled "wild-type"')
        
    line_out='name\talignment\tgroup\n'
    nseqsignored=0
    for row in read_tsv(infile,pc=False,header=True):
        name,al=row[:2]
        group=row[groupColIndex]
        label=group2label[group]
        al=al[max(start0idx,0):end0idx]
        
        # if seq is same as wt and is not the wt sequence, then skip it
        if removeSameAsWt:
            if label!='wild-type':
                if al==wtseq: 
                    nseqsignored+=1
                    continue
                

        line_out+=f'{name}\t{al}\t{group}\n'
    with open(outfile,'w') as f: f.write(line_out)
    print(nseqsignored,'squences ignored due to being same as wt')
    
def scan_seqs(  enhancer_tsv,
                group_tsv,
                tfname,
                quantifyTfbsType,
                ldf=pd.DataFrame(),
                tfbsCore=None,seq2aff=None,
                nt2weight=None,tfid=None,pwm_min_score=None,
                scoreRound=3,
                containsWildType=True):
    
    '''scan_seqs analyzes multiple aligned enhancers grouped by function to predict which sites may
    be causing the difference in function. 
    
    Parameters:
    - quantifyTfbsType - must be core-only, core-affinity or score-pwm
        - if core-only, you must specifiy "tfname". 
        - if core-affinity, you must specifiy "tfname" and "seq2aff"
        - if score-pwm, you must specify "nt2weight","tfid", and "pwm_min_score"
    - enhancer_tsv - enhancer input file (tsv with header) 
        col1 - enhancer name
        col2 - enhancer seq
        col3 - enhancer group (e.g. wild-type, phenotype, no-phenotype)
    - group_tsv - group definition file (tsv with header)
        col1 - enhancer group
        col2 - group definition/label (either "test" or "control", no other values allowed)
    - tfname - name of tf as it appears in the "nt2weight" dictionary
    - tfid - jaspar id of tf as it appears in the "nt2weight" dictionary
    - pwm_min_score - minimum score to call a site
    - groupCol - 1-index of the column you want to use for groups
    - scoreRound - how many decimal points to round the pwm score, default=3'''
    
    ####################################################
    # read groupings, make sure none are missing
    ####################################################
    
    # make sure quantifyTfbsType is correct
    if quantifyTfbsType not in ['score-pwm','core-only','core-affinity']:
        raise ValueError('quantifyTfbsType={quantifyTfbsType} is not allowed. quantifyTfbsType must be either score-pwm,core-only, or core-affinity.')
        
    # ensure no empty rows
    groupdf=pd.read_csv(group_tsv,sep='\t')
    if groupdf.isna().any().any(): raise ValueError('There is an NA or empty cell in your group.tsv file. check it.')
    
    # determine which groups are conterols and which are tests
    group2type={}
    for gi,ti in read_tsv(group_tsv,pc=False,header=True):
        group2type[gi]=ti
    allGroupLabels=set(group2type.values())
    
    # check that group labels are allowed
    notAllowedGroupLabels=set(allGroupLabels)-set(['control','test','na','wild-type'])
    if len(notAllowedGroupLabels)>0: 
        raise ValueError(f'Group labels have to be either control, test, na or wild-type. The following group labels are not allowed: {notAllowedGroupLabels}')
    
    # at least one control or wild type must be included
    if ('control' not in allGroupLabels) and ('wild-type' not in allGroupLabels):
        raise ValueError('group labels as provided in "group_tsv" must have at least one "control" and or "wild-type".')
    
    # if wild type is supposed to be provided, ensure it is provided
    if containsWildType and ('wild-type' not in allGroupLabels):
        raise ValueError(f'containsWildType={containsWildType} but no "wild-type" is provided in the groups.tsv file')
        
    # make sure only one wt is provided
    wtCounter=list(group2type.values()).count('wild-type')
    if containsWildType and (wtCounter>1):
        raise ValueError(f'Only one wild-type is allowed in the group-label column of group.tsv. {wtCounter} wild-types are currently provided.')
        
    # there must be at least one test group
    if ('test' not in allGroupLabels):
        raise ValueError('group labels as provided in "group_tsv" must have at least one "test".')
    
    ####################################################
    # load enhancer.tsv
    ####################################################
    
    # load all seqs
    edf=pd.read_csv(enhancer_tsv,sep='\t')
    otherCols=list(edf.columns)[3:]
    if otherCols!=[]: edf.columns=['enhancer-name','enhancer-sequence','group-name']+otherCols
    else:             edf.columns=['enhancer-name','enhancer-sequence','group-name']
    
    # ensure no empty rows
    if edf.isna().any().any(): raise ValueError('There is an NA or empty cell in your enhancer.tsv file. check it.')
    
    # remove any group names that are "na" sequences
    edf['group-label']=edf['group-name'].apply(lambda g: group2type[g])
    edf=edf.loc[edf['group-label'].isin(['control','test','wild-type']),:]
    
    # define groups
    allGroups=edf.loc[:,'group-name'].unique()
    
    # check all seqs are same length
    enseqLengths=[len(s) for s in edf.loc[:,'enhancer-sequence']]
    if len(set(enseqLengths))>1: raise ValueError('Some enhancer sequences are different lengths.')
    seqLen=enseqLengths[0]

    ####################################################
    # Begin table
    ####################################################    
    
    c2v={c:[] for c in ['enhancer-name','group','group-label','tf-name','tf-definition',
                        'tf-strength-type','tf-strength-value',
                        'strand','start','start-set','end','kmer','kmer-with-insertions']}#,'count-control','count-test']+[f'count-{gi}' for gi in allGroups]

    ####################################################
    # scan for pwm
    ####################################################    
    
    if quantifyTfbsType=='score-pwm':
        
        # check that all required parameters are passed
        if (str(nt2weight)=='None') or (str(tfid)=='None') or (pwm_min_score==None):
            raise ValueError(f'if quantifyTfbsType={quantifyTfbsType}, nt2weight, tfid and pwm_min_score must all be specified.')
            
        pwmLen     =all_funcs_get_pwm_length(nt2weight)
        maxpwmscore=all_funcs_get_max_score_of_pwm(nt2weight)
        minpwmscore=all_funcs_get_min_score_of_pwm(nt2weight)

        # iterate over every bp and save location of all

        kmer2score={}
        skipKmers=set()
        pes2type2members={} # maps position/strand to groups and sequences

        for name,seq,group in zipdf(edf,['enhancer-name','enhancer-sequence','group-name']):

            grouplabel=group2type[group]

            # for each position
            for pos0 in range(seqLen):

                # skip kmers that start with a gap.
                # if seq[pos0]=='-': continue

                for strand,end0 in [('+',pwmLen),('-',-pwmLen)]:

                    end0=pos0+end0

                    # if not full kmers, skip
                    if end0<0 or end0>seqLen: continue

                    # get kmer
                    if strand=='+': kmer=seq[pos0:end0]
                    else:           kmer=revcomp(seq[end0+1:pos0+1])

                    # keep progressing forward til you get all nucleotides and no dashes
                    if '-' in kmer:

                        numRealNt=len(kmer)-kmer.count('-')

                        while numRealNt!=pwmLen:
                            if strand=='+': 
                                end0+=1
                                kmer=seq[pos0:end0]
                            else:           
                                end0-=1
                                kmer=revcomp(seq[end0+1:pos0+1])

                            numRealNt=len(kmer)-kmer.count('-')

                            # give up if you reach the end 
                            if end0<0 or end0>seqLen: break

                    # Determine number of total dashes for adjustting possible start positions
                    insertionCount=kmer.count('-')
                    originalKmer=kmer

                    # get final kmer wihtout dashes, skip if all seqs not A/T/G/C
                    kmer=kmer.replace('-','')
                    if 'N' in kmer: continue

                    # skip if too youve seen this kmer before with too low score
                    if kmer in skipKmers: continue 

                    # get pwm score
                    if kmer in kmer2score:  
                        score=kmer2score[kmer]
                    else:                   
                        score=all_funcs_get_score_of_kmer(nt2weight,kmer,maxpwmscore,minpwmscore)
                        kmer2score[kmer]=score

                    score=round(score,scoreRound)

                    # skip if too low
                    if score<pwm_min_score: 
                        skipKmers.add(kmer)
                        continue

                    # if the kmer has - in it, include these possible starts as having a - meanss you need to be flexible with the start to correctly map the kmer in other enhancers
                    # will add the original start and the number of dashes
                    #  so if you have 1 -, it will be {start,start+1}. if 2-, it will be {start,start+1,start+2}
                    startset=set()
                    for startAdjust in range(0,insertionCount+1): 
                        if strand=='+': startset.add(pos0+startAdjust)
                        else:           startset.add(pos0-startAdjust)

                    # at this point the site is sufficient to call, so we add it to the table
                    c2v['enhancer-name'].append(name)
                    c2v['group'].append(group)
                    c2v['group-label'].append(grouplabel)
                    c2v['tf-name'].append(tfname)
                    c2v['tf-definition'].append(tfid)
                    c2v['tf-strength-type'].append(quantifyTfbsType)
                    c2v['tf-strength-value'].append(score)
                    c2v['strand'].append(strand)
                    c2v['start'].append(pos0)
                    c2v['start-set'].append(startset)
                    c2v['end'].append(end0)
                    c2v['kmer'].append(kmer)
                    c2v['kmer-with-insertions'].append(originalKmer)
                    # c2v['kmer-length'].append(len(kmer))

                    
    ####################################################
    # scan for core
    ####################################################
    
    if (quantifyTfbsType=='core-only') or (quantifyTfbsType=='core-affinity'):
        
        
        # some checks that all necesarry parameters were given
        if (tfbsCore==None): 
            raise ValueError('if quantifyTfbsType={quantifyTfbsType}, tfbsCore must be provided.')
        if (quantifyTfbsType=='core-affinity') and (seq2aff==None): 
            raise ValueError('if quantifyTfbsType={quantifyTfbsType}, seq2aff must be provided.')
                    
        coreRePattern=IupacToRegexPattern(tfbsCore)

        coreLen=len(tfbsCore)
        
        # make sure core length is consistent with the affinity dictionary
        if (quantifyTfbsType=='core-affinity'):
            lenSet=set([len(kmer) for kmer in seq2aff.keys()])
            if len(lenSet)>1: 
                raise ValueError('kmers are of different lengths in the sequence->affinity dataset. All kmers in this file must be the same length')
                
            # make sure length of kmer is lenght of core
            lengthOfRefAffKmer=list(lenSet)[0]
            if lengthOfRefAffKmer!=coreLen:
                raise ValueError('tfbsCore={tfbsCore} with length {coreLen}. Kmers in the sequence->affinity dataset are length {lengthOfRefAffKmer}. The length of tfbsCore and the length of the kmers in the affinity file must be the same length.')
            
        for name,seq,group in zipdf(edf,['enhancer-name','enhancer-sequence','group-name']):

            grouplabel=group2type[group]

            # for each position
            for pos0 in range(seqLen):

                # skip kmers that start with a gap.
                if seq[pos0]=='-': continue

                for strand,end0 in [('+',coreLen),('-',-coreLen)]:

                    end0=pos0+end0

                    # if not full kmers, skip
                    if end0<0 or end0>seqLen: continue

                    # get kmer
                    if strand=='+': kmer=seq[pos0:end0]
                    else:           kmer=revcomp(seq[end0+1:pos0+1])

                    # keep progressing forward til you get all nucleotides and no dashes
                    skipThisSeq=False
                    if '-' in kmer:

                        numRealNt=len(kmer)-kmer.count('-')

                        while numRealNt!=coreLen:
                            if strand=='+': 
                                end0+=1
                                kmer=seq[pos0:end0]
                            else:           
                                end0-=1
                                kmer=revcomp(seq[end0+1:pos0+1])

                            numRealNt=len(kmer)-kmer.count('-')

                            # give up if you reach the end 
                            if end0<0 or end0>seqLen: 
                                skipThisSeq=True
                                break
                                
                    if skipThisSeq or end0<0 or end0>seqLen: continue

                    # Determine number of total dashes for adjustting possible start positions
                    insertionCount=kmer.count('-')
                    originalKmer=kmer

                    # get final kmer wihtout dashes, skip if all seqs not A/T/G/C
                    kmer=kmer.replace('-','')
                    if 'N' in kmer: continue
                    
                    # skip if kmer is not a core site
                    if not re.search(coreRePattern,kmer): 
                        continue
                    
                    # get affinity if need be
                    if quantifyTfbsType=='core-only':     
                        score='.'
                    if quantifyTfbsType=='core-affinity': 
                        if kmer in seq2aff:
                            score=seq2aff[kmer]
                        else:
                            raise ValueError('kmer={kmer} not detected in the sequence->affinity dataset.')
                            
                        score=round(score,scoreRound)


                    # if the kmer has - in it, include these possible starts as having a - meanss you need to be flexible with the start to correctly map the kmer in other enhancers
                    # will add the original start and the number of dashes
                    #  so if you have 1 -, it will be {start,start+1}. if 2-, it will be {start,start+1,start+2}
                    startset=set()
                    for startAdjust in range(0,insertionCount+1): 
                        if strand=='+': startset.add(pos0+startAdjust)
                        else:           startset.add(pos0-startAdjust)

                    # at this point the site is sufficient to call, so we add it to the table
                    c2v['enhancer-name'].append(name)
                    c2v['group'].append(group)
                    c2v['group-label'].append(grouplabel)
                    c2v['tf-name'].append(tfname)
                    c2v['tf-definition'].append(tfbsCore)
                    c2v['tf-strength-type'].append(quantifyTfbsType)
                    c2v['tf-strength-value'].append(score)
                    c2v['strand'].append(strand)
                    c2v['start'].append(pos0)
                    c2v['start-set'].append(startset)
                    c2v['end'].append(end0)
                    c2v['kmer'].append(kmer)
                    c2v['kmer-with-insertions'].append(originalKmer)
                    # c2v['kmer-length'].append(len(kmer))

    
    ####################################################
    # output dataframe
    ####################################################
                
    # if nothing added, return false
    if len(c2v['enhancer-name'])==0: 
        return ldf

    # if ther are hits, continue
    longdf=pd.DataFrame(c2v)
    longdf=longdf.loc[(longdf['end']>=0) & (longdf['end']<=seqLen),:]
        
    
    ldf=pd.concat([ldf,longdf])
        
    return ldf 

def map_sites_across_seqs(ldf,
                          enhancer_tsv,
                          group_tsv,
                          containsWildType=True,
                          out_tsv=None):
    '''
    hypothesis can be LOF, GOF or both'''
    
    df=ldf.reset_index(drop=True)
    
    ####################################################
    # read groupings, make sure none are missing
    ####################################################
        
    # ensure no empty rows
    groupdf=pd.read_csv(group_tsv,sep='\t')
    if groupdf.isna().any().any(): raise ValueError('There is an NA or empty cell in your group.tsv file. check it.')
    
    # determine which groups are conterols and which are tests
    group2type={}
    type2group={}
    for gi,ti in read_tsv(group_tsv,pc=False,header=True):
        group2type[gi]=ti
        if ti not in type2group:
            type2group[ti]=set()
        type2group[ti].add(gi)
    allGroupLabels=set(group2type.values())
    
    # check that group labels are allowed
    notAllowedGroupLabels=set(allGroupLabels)-set(['control','test','na','wild-type'])
    if len(notAllowedGroupLabels)>0: 
        raise ValueError(f'Group labels have to be either control, test, na or wild-type. The following group labels are not allowed: {notAllowedGroupLabels}')
    
    # at least one control or wild type must be included
    if ('control' not in allGroupLabels) and ('wild-type' not in allGroupLabels):
        raise ValueError('group labels as provided in "group_tsv" must have at least one "control" and or "wild-type".')
    
    # if wild type is supposed to be provided, ensure it is provided
    if containsWildType and ('wild-type' not in allGroupLabels):
        raise ValueError(f'containsWildType={containsWildType} but no "wild-type" is provided in the groups.tsv file')
        
    # make sure only one wt is provided
    wtCounter=list(group2type.values()).count('wild-type')
    if containsWildType and (wtCounter>1):
        raise ValueError(f'Only one wild-type is allowed in the group-label column of group.tsv. {wtCounter} wild-types are currently provided.')
        
    # there must be at least one test group
    if ('test' not in allGroupLabels):
        raise ValueError('group labels as provided in "group_tsv" must have at least one "test".')
    
    ####################################################
    # load enhancer.tsv
    ####################################################
    
    # load all seqs
    edf=pd.read_csv(enhancer_tsv,sep='\t')
    
    # standardize column names
    otherCols=list(edf.columns)[3:]
    if otherCols!=[]: edf.columns=['enhancer-name','enhancer-sequence','group-name']+otherCols
    else:             edf.columns=['enhancer-name','enhancer-sequence','group-name']
    edf['enhancer-sequence']=edf['enhancer-sequence'].str.upper()
    
    # ensure no empty rows
    if edf.isna().any().any(): raise ValueError('There is an NA or empty cell in your enhancer.tsv file. check it.')
    
    # remove any group names that are "na" sequences
    edf['group-label']=edf['group-name'].apply(lambda g: group2type[g])
    edf=edf.loc[edf['group-label'].isin(['control','test','wild-type']),:]
    
    grouplabel2seqnameset={}
    for seqname,group in zipdf(edf,['enhancer-name','group-name']):
        # print(seqname,group)
        grouplabel=group2type[group]
        if grouplabel not in grouplabel2seqnameset:
            grouplabel2seqnameset[grouplabel]=set()
        grouplabel2seqnameset[grouplabel].add(seqname)
    

    if ('test' not in grouplabel2seqnameset): 
        raise ValueError(f'No sequences with "test" group label are in the table.')
        
    if ('control' not in grouplabel2seqnameset) and ('wild-type' not in grouplabel2seqnameset) : 
        raise ValueError(f'No sequences with "wild-type" nor "control" group labels are in the table.')

    if containsWildType:
        if 'wild-type' not in type2group: raise ValueError('if "containsWt" is True, "wild-type" must be one of the group labels.')
        
        groupNameWT=type2group['wild-type'] # this will be a set
        if len(groupNameWT)==1: groupNameWT=list(groupNameWT)[0]
        else: raise ValueError('Only one group can be assigned to wild-type')
        
        # print(groupNameWT)
        wtalignment=edf.loc[edf['group-name']==groupNameWT,'enhancer-sequence']
        # print(wtalignment)
        # print(len(wtalignment))
        
        if len(wtalignment)==0:
            raise ValueError(f'containsWt=={containsWt} but no sequences have a group "wild-type" designated.')
            
        elif len(wtalignment)>1:
            raise ValueError(f'more than one sequence is designated to "wild-type". only one sequence can be of this group.')
            
        else:
            wtalignment=wtalignment.tolist()[0].upper()
    
    ####################################################
    # group sites across sequences
    ####################################################
    
    siteid=0
    strengthType2tfname2pfmid2strand2start2siteid={}
    
    siteIdColumn=[]
    for tfname,tfpfmid,strand,startset,detectionType in zipdf(df,['tf-name','tf-definition','strand','start-set','tf-strength-type']):
        
        # if any of these are not observed yet, add them
        if detectionType not in strengthType2tfname2pfmid2strand2start2siteid: strengthType2tfname2pfmid2strand2start2siteid[detectionType]={}
        if tfname not in strengthType2tfname2pfmid2strand2start2siteid[detectionType]: strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname]={}
        if tfpfmid not in strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname]: strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid]={}
        if strand not in strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid]: strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid][strand]={}
            
        # first check what starts are already observed to see if this is a part of an existing start id
        startAlreadyObserved=False
        for preObsStart in strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid][strand]:
            
            # if previous observed start is within possible starts for this site, link the two to the same site id
            if preObsStart in startset:
                startAlreadyObserved=True
                preObsSiteId=strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid][strand][preObsStart]
                siteIdColumn.append(preObsSiteId)
                
                # for all of the current possible starts, link to the same id
                for start in startset:
                    strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid][strand][start]=preObsSiteId
                    
                break
                
        if startAlreadyObserved==True: continue
                
        # if this start hasn't been seen yet,
        if startAlreadyObserved==False:
            
            # add site to dataframe
            siteIdColumn.append(siteid)
            
            # map all possible starts to the same site id
            for start in startset:
                if start in strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid][strand]: # by this point, youve alrady made sure its a new start
                    raise ValueError('this shouldnt happen')
                else:
                    strengthType2tfname2pfmid2strand2start2siteid[detectionType][tfname][tfpfmid][strand][start]=siteid
            
            # move on to next assignable site id
            siteid+=1
        
    df['site-id']=siteIdColumn
    
    ####################################################
    # flatten sites
    ####################################################
    
    return df

def prioritize_sites(mdf,
                     enhancer_tsv,
                     group_tsv,
                     containsWildType=True,
                     out_tsv=None,deltamin=.05):
    
    df=mdf
    
    df['kmer-length']=df['kmer'].apply(lambda s: len(s))
    
    ####################################################################################
    # start new df
    ####################################################################################
    
    newcols=['site-id','tf-strength-type','tf-name','tf-definition','kmer-length','strand','start-set']
    newcols+=['wt-max-taken','wt-name','wt-group','wt-kmer','wt-strength']
    newcols+=['test-name-list','test-group-list','test-kmer-list','test-strength-list','test-strength-min','test-strength-max']
    newcols+=['control-name-list','control-group-list','control-kmer-list','control-strength-list','control-strength-min','control-strength-max']
    
    c2v={c:[] for c in newcols}
    
    ####################################################################################
    # collapse on sites to get df where each row is a siteid
    ####################################################################################
    
    
    
    for siteid in df['site-id'].unique():
        
        allStartSetValues=set()
        thisdf=df.loc[df['site-id']==siteid,:].reset_index(drop=False)
        
        ####################################
        # get things that should be constant
        
        # verify all values are same
        for unicol in ['tf-strength-type','tf-name','tf-definition','strand','kmer-length']:
            if thisdf[unicol].nunique()!=1: raise ValueError(f'the {unicol} column should have the same value for site id {siteid}')
            
        # since all are same, take the first one
        dettype  = thisdf.at[0,'tf-strength-type']
        tfname   = thisdf.at[0,'tf-name']
        pfmid    = thisdf.at[0,'tf-definition']
        strand   = thisdf.at[0,'strand']
        pfmLen   = thisdf.at[0,'kmer-length']
        
        c2v['site-id'].append(siteid)
        c2v['tf-strength-type'].append(dettype)
        c2v['tf-name'].append(tfname)
        c2v['tf-definition'].append(pfmid)
        c2v['kmer-length'].append(pfmLen)
        c2v['strand'].append(strand)

        ############################################################
        # get things for just the test and control subgroups

        for label in ['test','control']:
            labeldf=thisdf.loc[thisdf['group-label']==label,:]
            
            if len(labeldf)==0:
                names=[]
                groups=[]
                kmers=[]
                strengths=[]
                startsets=[]
                stmin='.'
                stmax='.'
                
            else:
                names=labeldf['enhancer-name'].tolist()
                groups=labeldf['group'].tolist()
                kmers=labeldf['kmer'].tolist()
                strengths=labeldf['tf-strength-value'].tolist()
                strengths=[i for i in strengths if i!='.']
                startsets=labeldf['start-set'].tolist() # also get starts to add
                
                if strengths!=[]:
                    stmin=min(strengths)
                    stmax=max(strengths)
                else:
                    stmin='.'
                    stmax='.'
                
            c2v[f'{label}-name-list'].append(names)
            c2v[f'{label}-group-list'].append(groups)
            c2v[f'{label}-kmer-list'].append(kmers)
            c2v[f'{label}-strength-list'].append(strengths)
            c2v[f'{label}-strength-min'].append(stmin)
            c2v[f'{label}-strength-max'].append(stmax)
            
            for startset in startsets:
                allStartSetValues = startset | allStartSetValues # add all current starts
                    

        ############################################################
        # get wt - you dont need to check if containsWt is true s
        #          because the df will just be empty otherwise
        labeldf=thisdf.loc[thisdf['group-label']=='wild-type',:].reset_index(drop=True)
        wtMaxTaken='.'

        # if more than one hit for wt, take the max
        if len(labeldf)>1: 
            wtMaxTaken=True
            maxStrength=labeldf['tf-strength-value'].max()
            for name,group,kmer,strength,starts in zipdf(labeldf,['enhancer-name','group','kmer','tf-strength-value','start-set']):
                if strength==maxStrength:
                    wtname=name
                    wtgroup=group
                    wtkmer=kmer
                    wtstrength=strength
                    wtstarts=starts
                    break

        # if only one hit for wt, take the one
        elif len(labeldf)>0:
            wtname=labeldf.at[0,'enhancer-name']
            wtgroup=labeldf.at[0,'group']
            wtkmer=labeldf.at[0,'kmer']
            wtstrength=labeldf.at[0,'tf-strength-value']
            wtstarts=labeldf.at[0,'start-set']

        # otherwise, add nothing
        else:
            wtname='.'
            wtgroup='.'
            wtkmer='.'
            wtstrength='.'
            wtstarts=set()
                
        c2v['wt-max-taken'].append(wtMaxTaken)
        c2v['wt-name'].append(wtname)
        c2v['wt-group'].append(wtgroup)
        c2v['wt-kmer'].append(wtkmer)
        c2v['wt-strength'].append(wtstrength)
            
        # add starts
        allStartSetValues = wtstarts | allStartSetValues
            
            
        ############################################################
        # add all start sets seen
        c2v['start-set'].append(allStartSetValues)
    
    cmdf=pd.DataFrame(c2v)
    
    ####################################################################################
    # prioritize
    ####################################################################################
    
    deltaGofList=[]
    deltaLofList=[]
    typeList=[]
    
    for wtval, testvalmin, testvalmax, controlvalmin,controlvalmax in zipdf(cmdf,['wt-strength','test-strength-min','test-strength-max','control-strength-min','control-strength-max']):
        
        testvalgof=testvalmax
        testvallof=testvalmin
        
        # determine reference value (will be wt if exists, otherwise will be control)
        if wtval!='.': 
            refvalgof=wtval
            refvallof=wtval
        else:          
            refvalgof=controlvalmax
            refvallof=controlvalmin
            
        ####################################
        # compare strengths
        
        testSiteExists = testvalgof!='.'
        refSiteExists  = refvalgof !='.'
        
        # de novo
        if testSiteExists and not refSiteExists:
            deltaGof='de novo'
            deltaLof=np.NaN
            
            typeVar='dnv'
            
        # deletion
        elif not testSiteExists and refSiteExists:
            deltaGof=np.NaN
            deltaLof='ablation'
            
            typeVar='abl'
            
            
        # opt or subopt
        elif testSiteExists and refSiteExists:
            
            deltaGof=testvalgof-refvalgof
            if   deltaGof-deltamin<=0: # deltaGof-deltamin>=0 - maggie-modification, > to <
                deltaGof=np.NaN

            deltaLof=testvallof-refvallof
            if deltaLof >= 0.0 or (abs(deltaLof)<=deltamin and deltaLof < 0.0):    
            #elif abs(deltaLof)<=deltamin and deltaLof < 0.0: #maggie-modification
                deltaLof=np.NaN

            #if not np.isnan(deltaGof) and not np.isnan(deltaLof): typeVar='Mix' #Maggie modification 
            if pd.notnull(deltaGof):   typeVar='inc' # true even if deltaGof is null # if not np.isnan(deltaGof)
            elif pd.notnull(deltaLof):   typeVar='dec'
            else:                        typeVar=np.NaN
            
        elif not testSiteExists and not refSiteExists:
            deltaGof=np.NaN
            deltaLof=np.NaN
            typeVar=np.NaN

        deltaGofList.append(deltaGof)
        deltaLofList.append(deltaLof)
        typeList.append(typeVar)
    
    cmdf['gof-delta']=deltaGofList
    cmdf['lof-delta']=deltaLofList
    cmdf['variant-effect']=typeList
    
    ############################################
    # get rank of columns for gof
    cmdf = cmdf.sort_values(by=[f'gof-delta','test-strength-max'], key=lambda col: col.map(lambda x: (isinstance(x, (float, int)), x)),
                            ascending=[True,False]).reset_index(drop=True)
    cmdf[f'gof-rank']=cmdf.index
    cmdf[f'gof-rank']=cmdf[f'gof-rank']+1
    cmdf.loc[cmdf['gof-delta'].isna(),'gof-rank']=np.NaN # nullifies any gof rank values with no gof effect

    ############################################
    # get rank of columns for gof
    
    # if wt exists, then enforce that anything missing in test/wt should not be considered LOF. 
    if containsWildType:
        ablatedrows=cmdf['lof-delta']=='ablation'
        wtMissingRows=cmdf['wt-strength']=='.'
        cmdf.loc[ ablatedrows & wtMissingRows,'lof-delta']=np.NaN
    cmdf = cmdf.sort_values(by=['lof-delta','wt-strength','control-strength-max'], 
                            key=lambda col: col.map(lambda x: (isinstance(x, (float, int)), -x if isinstance(x, (float, int)) else x)),
                            ascending=[True,True,True]).reset_index(drop=True)
    cmdf[f'lof-rank']=cmdf.index
    cmdf[f'lof-rank']=cmdf[f'lof-rank']+1
    cmdf.loc[cmdf['lof-delta'].isna(),'lof-rank']=np.NaN 
    
    ####################################################################################
    # merge dataframes
    ####################################################################################
    
    siteid2info2value={}
    for siteid,gofdelta,gofrank,lofdelta,lofrank in zipdf(cmdf,['site-id','gof-delta','gof-rank','lof-delta','lof-rank']):
        
        siteid2info2value[siteid]={'gofd':gofdelta,'lofd':lofdelta,'gofr':gofrank,'lofr':lofrank}    
        
    df['gof-rank'] =df['site-id'].apply(lambda si: siteid2info2value[si]['gofr'])    
    df['gof-delta']=df['site-id'].apply(lambda si: siteid2info2value[si]['gofd'])    
    df['lof-rank'] =df['site-id'].apply(lambda si: siteid2info2value[si]['lofr'])    
    df['lof-delta']=df['site-id'].apply(lambda si: siteid2info2value[si]['lofd'])
    
    ####################################################################################
    # remove na values from both (neither gof nor lof )
    ####################################################################################
    
    lofHits=~cmdf['lof-rank'].isna()
    gofHits=~cmdf['gof-rank'].isna()
    cmdf=cmdf.loc[lofHits | gofHits,:]
    
    lofHits=~df['lof-rank'].isna()
    gofHits=~df['gof-rank'].isna()
    df=df.loc[lofHits | gofHits,:]
    
    ####################################################################################
    # get some last minute info
    ####################################################################################
    
    # print(df.head())
    if len(df)==0: return False,False
    df['kmer-oriented-with-wt']=df.apply(lambda row: row['kmer'] if row['strand']=='+' else revcomp(row['kmer']),axis=1)
    # df['kmer-oriented-with-wt']=df.apply(lambda row: get_lowercase_diffs(row['',axis=1)
        
    df=df.sort_values('lof-rank')
    cmdf=cmdf.sort_values('lof-rank')
    return cmdf,df

def check_input_tf_sheet_tsv(input_tf_sheet_tsv):
    
    alltfs=[]
    with open(input_tf_sheet_tsv,'r') as f:
        f.readline()
        for l in f:
            l=l.strip().split('\t')
            alltfs.append(l[0])
    if len(alltfs)!=len(set(alltfs)):
        raise ValueError('the tf-name column of the input_tf_sheet_tsv must be unique for every row.')

def check_input_tf_sheet_df(input_tf_sheet_df):

    alltfs = list(input_tf_sheet_df['tfname'])
    if len(alltfs)!=len(set(alltfs)):
        raise ValueError('the tf-name column of the input_tf_sheet_df must be unique for every row.')
            
    
def compare_seqs_pipeline(enhancer_tsv,
                          containsWildType,
                          group_tsv,
                          isAlreadyPwm,
                          isFraction,
                          pseudocounts,
                          input_tf_sheet_df=None,
                          input_pwm_batch_fn=None,
                          deltamin=.05,
                          plotMsa=False,
                          trimAlignment=(None,None),
                          ifTrim_removeSameAsWt=False,
                          outdir='./',
                          scoreRound=3,
                          pwm_min_score=.6,
                          betaPwm=False,
                          pwmPrintProgress=False,ignorePythonWarnings=False,
                          pwm_file_format='jaspar'):
    
    if ignorePythonWarnings:
        warnings.filterwarnings('ignore')
    if input_tf_sheet_df is not None: check_input_tf_sheet_df(input_tf_sheet_df)
    
    # if input_pwm_batch_fn==None and input_pwm_batch_fn==None:
    #     raise ValueError('one of the following (or borth) must be specificed: input_pwm_batch_fn or input_pwm_batch_fn.')
    
    ############################################################
    print('trimming alignments...')
    ############################################################
    
    if trimAlignment[0]!=None:
        outtrimfile=f'{outdir}/trim_align.tsv'
        trim_alignment_tsv_file(infile=enhancer_tsv,
                                    outfile=outtrimfile,
                                    start0idx=trimAlignment[0],
                                    end0idx=trimAlignment[1],
                                    removeSameAsWt=ifTrim_removeSameAsWt)
        
        enhancer_tsv=outtrimfile

    if plotMsa:
        plot_msa(enhancer_tsv,
                 trim=None,
                 group2color={'human':'black','mouse':'black','na':'black','serpintized':'black','wt':'black'},
                 sortByType=None,
                 sortList=None,textSize=16,plotTrimBox=False,onlyColorNtDifferentThanWildType=None,dpi=300,
                 outdir=outdir)
    
    #######################################################################################
    print('Scanning sites for cores...')
    #######################################################################################

    ldf=pd.DataFrame()
    
    # do core only and pbm affinities
    if input_tf_sheet_df is not None:
        for name,core,aff_ref_fn in zip(input_tf_sheet_df['tfname'], input_tf_sheet_df['core'], input_tf_sheet_df['aff-ref']):

            if core.upper()!=core: raise ValueError('Lowercase letters detected in core. Only use uppercase in core binding site definition using IUPAC nomenclature.')

            for i in core: 
                if i not in Iupac2AllNt.keys(): raise ValueError(f'{i} is not a valid nt. the core site can only contain the following sequences {Iupac2AllNt.keys()}')

            if aff_ref_fn=='na': 
                searchType='core-only'
                seq2aff=None

            elif aff_ref_fn!='na':
                searchType='core-affinity'
                seq2aff=loadNormalizedFile(aff_ref_fn)

            ldf= scan_seqs( ldf=ldf,
                            enhancer_tsv=enhancer_tsv,
                            group_tsv=group_tsv,
                            tfname=name,
                            quantifyTfbsType=searchType, 
                            tfbsCore=core, 
                            seq2aff=seq2aff,
                            scoreRound=3,
                            containsWildType=containsWildType)
        
        
    #######################################################################################
    print('Scanning sites for pwms...')
    #######################################################################################

    if input_pwm_batch_fn:
        name2nt2counts=get_pwm_python_object_batch(input_pwm_batch_fn,isAlreadyPwm=isAlreadyPwm,isFraction=isFraction,pseudocounts=pseudocounts, pwm_file_format=pwm_file_format)
        searchType='score-pwm'

        if betaPwm:
            totalPwmToScreen=betaPwm
        else:
            totalPwmToScreen=len(name2nt2counts)

        for lc,((tfid,tfname),nt2counts) in enumerate(name2nt2counts.items()):

            if pwmPrintProgress:
                if lc % pwmPrintProgress==0: print(round(100*lc/totalPwmToScreen),'%',sep='',end=', ')

            ldf = scan_seqs(        ldf=ldf,
                                    enhancer_tsv=enhancer_tsv,
                                    group_tsv=group_tsv,
                                    tfname=tfname,
                                    quantifyTfbsType=searchType, 
                                    tfid=tfid, 
                                    nt2weight=name2nt2counts[tfid,tfname], 
                                    pwm_min_score=pwm_min_score,
                                    scoreRound=scoreRound,
                                    containsWildType=containsWildType)

            if betaPwm:
                if lc>betaPwm: 
                    print('[[WARNING]] not all pwms scanned due to beta testing. ')
                    break

    if len(ldf) == 0:
        print('[WARNING] No hits found! Exiting.')
        return False,False,False,False
    
    # ldf.to_csv(f'{outdir}/0-ldf.tsv',sep='\t',index=None)
    #######################################################################################
    print('map sites across enhancers...')
    #######################################################################################

    mdf  =map_sites_across_seqs(ldf,
                                enhancer_tsv=enhancer_tsv,
                                group_tsv=group_tsv,
                                containsWildType=containsWildType,
                                out_tsv=None)

    if len(mdf)==0: 
        print('[WARNING] No hits found! Exiting.')
        return False,False,False,False
    
    # mdf.to_csv(f'{outdir}/1-mdf.tsv',sep='\t',index=None)
    #######################################################################################
    print('prioritize gof and lof...')
    #######################################################################################

    cmdf,lmdf=prioritize_sites(  mdf,
                                 enhancer_tsv=enhancer_tsv,
                                 group_tsv=group_tsv,
                                 containsWildType=containsWildType,
                                 out_tsv=None)
    
    if type(cmdf)==bool:
        print('[WARNING] No hits found! Exiting.')
        return False,False,False,False
    if type(lmdf)==bool: 
        print('[WARNING] No hits found! Exiting.')
        return False,False,False,False

    cmdf.to_csv(f'{outdir}/differential-binding-sites.tsv',sep='\t',index=None)
    # lmdf.to_csv(f'{outdir}/2-lmdf.tsv',sep='\t',index=None)
    
    print('done!')
    return ldf,mdf,lmdf,cmdf


################################################################
# Visualizations
################################################################

def visualize_site(siteid,pldf=None,pldf_fn=None,plotType='swarm',ylim=(.6,1.05),title=None,highlightSeqs=[],outfn=None):
    '''
    - plotType can be swarm or violin or strip'''
    
    theseSiteIds=pldf['site-id']==siteid
    if sum(theseSiteIds)==0:
        print(f'[WARNING] no sites detected for {siteid}')
        return False
    
    plotdf=pldf.loc[theseSiteIds,:]
    # print(plotdf)
    
    highlightRows=plotdf['enhancer-name'].isin(highlightSeqs)
    
    fig,ax=quickfig()
    
    if plotType=='swarm':    sns.swarmplot( data=plotdf.loc[~highlightRows,:],x='group-label',y='tf-strength-value',palette=['black','grey','purple'],order=['wild-type','control','test'])
    elif plotType=='violin': sns.violinplot(data=plotdf.loc[~highlightRows,:],x='group-label',y='tf-strength-value',palette=['black','grey','purple'],order=['wild-type','control','test'])
    elif plotType=='strip':  sns.stripplot( data=plotdf.loc[~highlightRows,:],x='group-label',y='tf-strength-value',palette=['black','grey','purple'],order=['wild-type','control','test'])
       
    # which seqs to highlight
    hdf=plotdf.loc[highlightRows,:]
    grouplabel2x={'wild-type':0,'control':1,'test':2}
    for score,label in zipdf(hdf,['tf-strength-value','group-label']):
        ax.scatter(grouplabel2x[label],score,color='red',zorder=1e9)
        
    ax.set_ylim(ylim)
    ax.set_xlim(-.5,2.5)
    
    if title:
        ax.set_title(title)
        
    if outfn:
        plt.savefig(outfn)
        plt.close(fig)
    else:
        return fig,ax
        

def create_pdf(out_pdf_fn,pdf_fn,pldf_fn,plotType='strip',highlightSeqs=[],seqnameFilter=None,seqnameFilter_hypothesis=None):
    '''
    - if "seqnameFilter" is provided, only those sites with that seqname will be provided. 
        - if "seqnameFilter" is provided, "seqnameFilter_hypothesis" must be provided as well as gof or log
    - if "highlightSeqs" is  provided, those names will come up as red dots in all the visualizations
    '''
    
    if seqnameFilter_hypothesis not in ['gof','lof']: raise ValueError('seqnameFilter_hypothesis must be "gof" or "lof"')
    
    pdf=pd.read_csv(pdf_fn,sep='\t')
    pldf=pd.read_csv(pldf_fn,sep='\t')
    
    if len(pdf)==0 or len(pldf)==0: 
        print('[WARNING] No sites to plot')
        return False

    if highlightSeqs!=[] and seqnameFilter!=None: raise ValueError('Ony "highlightSeqs" or "seqnameFilter" can be specificied')
    
    if seqnameFilter:
        print(f'visualizing sites of {seqnameFilter}...')
        
        highlightSeqs=[seqnameFilter]

        # get order of most impressive site ids
        siteIdList=[]
        siteIdAdded=set()
        
        # get all seqs that...
        #               have a site id             & are of the sequence of interest   & remove any where the sequence is the same as wt
        # rowsOfInterest=(pldf['site-id'].notnull()) & (pldf['enhancer-name']==seqnameFilter) & (pldf['kmer-test-same-as-wt'].notnull())
        rowsOfInterest=pldf['enhancer-name']==seqnameFilter
        
        if sum(rowsOfInterest)==0: 
            print(f'No hits for {seqnameFilter} found.')
            return False
        
        seqname_pldf=pldf.loc[rowsOfInterest,:]
        
        # rank by gof or lof
        if seqnameFilter_hypothesis=='gof':
            ascending=False
        elif seqnameFilter_hypothesis=='lof':
            ascending=True
        seqname_pldf=seqname_pldf.sort_values(f'tf-strength-value',ascending=ascending)
        
        for siteid in seqname_pldf['site-id']:
            if siteid not in siteIdAdded:
                siteIdList.append(siteid)
                siteIdAdded.add(siteid)

        # sort pdf by the filter ids you want
        pdf['site-id']=pd.Categorical(pdf['site-id'], categories=siteIdList, ordered=True)
        pdf=pdf.sort_values('site-id')
                
    else:
        print('visualizing all sites in order of the pdf...')
    
    if seqnameFilter:
        print(f'# sites detected for {seqnameFilter}:',len(siteIdList))
    
    with PdfPages(out_pdf_fn) as pdfn:
        
        for siteid in pdf['site-id']:#zipdf(pdf['site-id']:#,['site-id']):
            # print(siteid)
            if seqnameFilter:
                if siteid not in siteIdAdded: continue
            
            title=f'{siteid}'
            
            # print(siteid,highlightSeqs)
            ax=visualize_site(siteid, pldf=pldf, plotType=plotType, title=title,highlightSeqs=highlightSeqs)
                
            if ax:
                pdfn.savefig()
                plt.close(fig)
                # plt.show() 
                
            else:
                plt.close(fig)
                print('SKIPPPEEEDD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                pass
                
            
    print(f'pdf outputted to {out_pdf_fn}')
    return True
        

def plot_sat_mut(element,mpradf,trims,xstep,plot_width=40,plot_height=5, plotNsVariants=True,gridlinexspace=None,gridlineyspace=None,dpi=150):
    
    plotdf=mpradf.loc[mpradf.Element==element,:]
    plotdf['p-adj']=fdr_correction(plotdf['P-Value'])
    
    realPvalList=[i for i in plotdf['p-adj'] if i!=0]
    if realPvalList!=[]:
        pass
    else:
        minp=min(realPvalList)
    
    fig,ax=quickfig(plot_width,plot_height,dpi)
    maxEffect=-np.inf
    minEffect=np.inf
    posList=[]
    effectList=[]
    for i,(pos,padj,effect) in enumerate(zipdf(plotdf,['pos-rel-0idx','p-adj','Value'])):
        
        posList.append(pos)
        effectList.append(effect)
        
        i=i%4
        
        # print(i)
        if effect>maxEffect: maxEffect=effect
        if effect<minEffect: minEffect=effect
        
        x=pos
        y=effect        
        
        if padj>.05:
            color='gainsboro'
            zorder=1
        else:
            color='black'
            zorder=10000
        
        # plt.scatter(x,y,color='black')
        plt.plot([x+i*.25,x+i*.25],[0,y],color=color,zorder=zorder)
        
    # add trims
    for start,end,label in trims:
        xy=(start,minEffect)
        width=end-start
        height=maxEffect-minEffect
        
        rectangle = patches.Rectangle(xy, width, height, linewidth=1, edgecolor=None, facecolor='blue', alpha=0.1)
        ax.add_patch(rectangle)
        
        x=xy[0]
        ax.text(start,minEffect,label,ha='left',va='bottom',rotation=90)
    
    # aes
    ax.set_xticks(range(0,pos+2*xstep,xstep))
    plt.ylabel('log2(RNA/DNA)')
    plt.title(element)
    clearspines(ax,sides=['top','right','bottom'])
    ax.axhline(0,lw=.5)
    ax.set_xlim(-5,max(posList)+5)
    
    if gridlinexspace:
        for xi in range(0,max(posList),gridlinexspace):
            plt.axvline(xi,ls='--',color='grey',lw=.1)
    if gridlineyspace:
        for yi in range(0,int(max(effectList))+gridlineyspace,gridlineyspace):
            plt.axhline(yi,ls='--',color='grey',lw=.1)
            
    ax.set_xticks(range(0,max(posList),10))

            
    #  gridlines
    return fig,ax
        
        
def plot_msa(enhancer_fn,trim=None,sortByType=None,
             group2color=None,
             sortList=None,xlimbuffer=15,
             nt2color={'A':'green','T':'blue','G':'red','C':'orange','-':'white','N':'white'},
             plotText=False,groupcol=None,textSize=10,plotTrimBox=True,dpi=150,
             xtickinterval=5,gridlinexspace=None,
             onlyColorNtDifferentThanWildType=False,
             outdir=None):
    
    '''if onlyColorNtDifferentThanWildType is provided, it should be the name of the wild-type sequence'''

    # load data
    namecol=0
    alncol=1
    if not groupcol:
        groupcol=2
    
    endf=pd.read_csv(enhancer_fn,sep='\t')
    endf.columns=['name','seq','group']
    
    # double check that names are unique
    if len(endf['name'].unique())!=len(endf['name']): raise ValueError('Names of enhancers must be unique.')
    
    #######################
    # prep seqs
    #######################
    
    # make sure all alignments same legnth
    allSeqs=endf.iloc[:,alncol].tolist()
    lenSet=set([len(aln) for aln in allSeqs])
    alnLength=list(lenSet)[0]
    if len(lenSet)>1: raise ValueError(f'not all alignments are the same length. Lengths are {lenSet}')
    
    # sort datafrmae as it will be plotted
    # if sortByType:
    #     if sortByType=='name':    
    #         endf=endf.loc[endf.iloc[:,namecol],:].set_index(endf.columns[namecol],drop=False)
    #         endf=endf.loc[sortList,:]
    #     elif sortByType=='group':
    #         raise ValueError('not supported yet')
    #     elif sortByType=='label':
    #         raise ValueError('not supported yet')
    #     else:
    #         raise ValueError('sortByType must be either "name", "group" or "label"')
    
    #######################
    # plot all seqs  
    #######################
    
    # get the wildtype sequence
    if onlyColorNtDifferentThanWildType:
        wtseq=''
        for seq,name in zipdf(endf,['seq','name']):
            if name==onlyColorNtDifferentThanWildType:
                wtseq=seq
        if wtseq=='': raise ValueError(f"The name provided by onlyColorNtDifferentThanWildType ({onlyColorNtDifferentThanWildType}) is not included as a name in the enhancer input file")
                
    if trim:
        trimstart,trimstop=trim
    else: 
        trimstart,trimstop=0,alnLength
        
    plot_width=(trimstop-trimstart)
    plot_height=len(endf)/2

    fig,ax=quickfig(plot_width,plot_height,dpi)

    colnames=[endf.columns[i] for i in [namecol,alncol,groupcol]]
    for yi,(name,seq,group) in enumerate(zipdf(endf,colnames)):
        if pd.isnull(group): continue
        if trim:
            startAdj=trimstart-xlimbuffer
            endAdj=trimstop+xlimbuffer+1
        else:
            startAdj=trimstart
            endAdj=trimstop
        seq=seq[max(0,startAdj):max(0,endAdj)]

        # print(seq)

        # plot group color and text
        groupAnnoColor=group2color[group]
        ax.text(startAdj-1,yi,name,va='center',ha='right',fontfamily='monospace',color=groupAnnoColor,size=textSize)
        # rectangle = patches.Rectangle(xygroup, alnLength, .75, linewidth=3, edgecolor=groupAnnoColor, facecolor=None, alpha=1)
        # ax.add_patch(rectangle)

        # print(seq)
        for xi,nt in enumerate(seq):

            xi+=startAdj
            nt=nt.upper()

            # plot text
            ax.text(xi,yi,nt,va='center',ha='center',fontfamily='monospace',size=textSize)

            # plot squares
            if onlyColorNtDifferentThanWildType:
                if wtseq[xi]==nt:
                    color='white'
                else:
                    if nt!='-': color=nt2color[nt]
                    else:       color='grey'
            else:
                color=nt2color[nt]
            xynt=(xi-.5,yi-.5)
            rectangle = patches.Rectangle(xynt, 1, 1, linewidth=1, edgecolor=None, facecolor=color, alpha=.6)
            ax.add_patch(rectangle)


    # plot trim length
    if plotTrimBox:
        if trim:
            xyTrim=(startAdj+xlimbuffer-.5,-0.5)
            rectangle = patches.Rectangle(xyTrim, len(seq)-2*xlimbuffer, len(endf), linewidth=3, edgecolor='red', facecolor='none', alpha=1)
        else:
            xyTrim=(startAdj-.5,-0.5)
            rectangle = patches.Rectangle(xyTrim, len(seq), len(endf), linewidth=3, edgecolor='red', facecolor='none', alpha=1)
            
        ax.add_patch(rectangle)

    ax.set_ylim(-.5,len(endf))
    ax.set_xlim(-1+startAdj,endAdj)

    clearspines(ax,sides=['top','right','bottom','left'])
    clearticks(ax,sides=['y'])
    
    if gridlinexspace:
        for xi in range(0,alnLength,gridlinexspace):
            plt.axvline(xi,ls='--',color='grey',lw=.1)
            
    ax.set_xticks(range(0,alnLength,xtickinterval))

    if outdir:
        plt.savefig(f'{outdir}/enhancer-alignment.svg',format='svg')
        plt.close(fig)
    else:
        plt.show()

################################################################
# html report
################################################################
    
background_Freq= [0.25, 0.25, 0.25, 0.25]


def pwm_to_pfm_approximation(pwm):
    pfm = 2**(pwm)*0.25
    return pfm

def calculate_information_content(jaspar_file, isAlreadyPwm, isFraction, background=None, reverse_complement=False, pwm_file_format = 'jaspar'):
    """
    Calculate the information content for each position in a PWM.
    
    Parameters:
        pwm (numpy.ndarray): A 2D array representing the PWM. Each row corresponds to a nucleotide (A, C, G, T) and each column to a position.
        background (list or numpy.ndarray): Background frequencies for each nucleotide. 
                                             If None, a uniform distribution is assumed.
    
    Returns:
        numpy.ndarray: A 1D array with the information content for each position.
    """
        
    if isAlreadyPwm:
        pwm=get_pwm_python_object(jaspar_file, isAlreadyPwm, isFraction, pwm_file_format=pwm_file_format)
        pfm=pwm_to_pfm_approximation(pwm)
    
    else:
        input_pfm = list(motif_parser(jaspar_file, pwm_file_format=pwm_file_format).values())[0] 
        pfm = get_FracMatrix_from_Jaspar(input_pfm, isFraction)
   
  
    if reverse_complement==True:
        pfm[[0, 3]] = pfm[[3, 0]]
        pfm[[1, 2]] = pfm[[2, 1]]
        pfm=pfm[:,::-1]
  
    pfm = np.array(pfm)
    if background is None:
        background = np.ones(pfm.shape[0]) / pfm.shape[0]
   
    background = np.array(background)
    
    # print(pfm)
    log2 = np.log2((pfm)/ background[:, None])
    log2 = np.nan_to_num(log2, neginf=0) 
    ic = np.sum(pfm * log2, axis=0) 
    
    return pfm, ic

def sequence_logo_generator(jaspar_file, isAlreadyPwm, isFraction, background=None, reverse_complement=False,dpi=100,outfn=None, pwm_file_format='jaspar'):
    
    pfm, ic = calculate_information_content(jaspar_file,isAlreadyPwm, isFraction, background, reverse_complement, pwm_file_format=pwm_file_format)
    
    ic=[i/max(ic) for i in ic]
    # print(ic)
    
    num_positions = pfm.shape[1]
    
    fig, ax = plt.subplots(figsize=(num_positions, 4),dpi=dpi)
    adj_pfm = pfm*(ic)
    # ax.set_ylim(0,max([max(i) for i in adj_pfm]))
    ax.set_ylim(0,1)
    colors = ['#00BB00', '#0000EE', '#F9A500', '#DD0000']
    nuc_labels = ["A", "C", "G", "T"]

    # get directory of python script relative to image files
    script_file_path = os.path.abspath(os.path.dirname(__file__))
    imgA = mpimg.imread(os.path.join(script_file_path, '05-compareTfSitesAcrossSequencesFromDnaSequences/pwmletters/a.png'))
    imgC = mpimg.imread(os.path.join(script_file_path, '05-compareTfSitesAcrossSequencesFromDnaSequences/pwmletters/c.png'))
    imgG = mpimg.imread(os.path.join(script_file_path, '05-compareTfSitesAcrossSequencesFromDnaSequences/pwmletters/g.png'))
    imgT = mpimg.imread(os.path.join(script_file_path, '05-compareTfSitesAcrossSequencesFromDnaSequences/pwmletters/t.png'))
    
    images = [imgA,imgC,imgG,imgT]
    
    for j in range(num_positions):
        sorted_idx = np.argsort(adj_pfm[:,j])
        bottoms = 0
        for idx in sorted_idx:
            height = adj_pfm[idx, j]
            if height> 0:
                bar = ax.bar(j, adj_pfm[idx,j], bottom=bottoms, width=0.99, color='white', alpha=0)
                rect = bar[0]
                x = rect.get_x()
                y = rect.get_y()
                h = rect.get_height()
                w = rect.get_width()
                ax.imshow(images[idx], extent=[x, x+w, y, y+h])
            bottoms += adj_pfm[idx,j]

    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.6, num_positions-0.4)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect(aspect='auto')
    ax.grid(False)
    
    if outfn:
        plt.savefig(outfn,format='png')
        plt.close(fig)
    else:
        return fig,ax

def get_kmer_with_dashes(dna,start,strand,length):
    
    seqLen=len(dna)
    # if dna[start]=='-': raise ValueError('cant start on a - ')
    
    if strand=='+':   end=start+length
    elif strand=='-': end=start-length

    # if not full kmers, report that it is invalid
    if end<0 or end>seqLen: return False

    # get kmer
    if strand=='+': kmer=dna[start:end]
    else:           kmer=revcomp(dna[end+1:start+1])

    # print(kmer)
    # keep progressing forward til you get all nucleotides and no dashes
    if '-' in kmer:

        numRealNt=len(kmer)-kmer.count('-')

        while numRealNt!=length:
            if strand=='+': 
                end+=1
                kmer=dna[start:end]
            else:           
                end-=1
                kmer=revcomp(dna[end+1:start+1])

            numRealNt=len(kmer)-kmer.count('-')

            # give up if you reach the end 
            if end<0 or end>seqLen: break

    # Determine number of total dashes for adjustting possible start positions
    insertionCount=kmer.count('-')
    originalKmer=kmer

    # get final kmer wihtout dashes, skip if all seqs not A/T/G/C
    kmer=kmer.replace('-','')
    
    return kmer

def generate_pwm_kmer_report_fig(lmdf,
                                 cmdf,
                                 enhancer_tsv,
                                 group_tsv,
                                 siteid,
                                 jasparMotifFile,
                                 dpi,hypothesis,containsWt,pseudocounts,isAlreadyPwm,isFraction,outfn=None,
                                 pwm_file_format='jaspar'):
        
    # print('loading files...')
    
    # load pfm file for calculating score
    pwm=get_pwm_python_object(jasparMotifFile,isAlreadyPwm,isFraction, pseudocounts=pseudocounts, pwm_file_format=pwm_file_format)
    maxPwmScore=all_funcs_get_max_score_of_pwm(pwm)
    minPwmScore=all_funcs_get_min_score_of_pwm(pwm)
    
    pwmPC=get_pwm_python_object(jasparMotifFile,isAlreadyPwm,isFraction, pseudocounts=pseudocounts, pwm_file_format=pwm_file_format)
    maxPwmScorePC=all_funcs_get_max_score_of_pwm(pwmPC)
    minPwmScorePC=all_funcs_get_min_score_of_pwm(pwmPC)
    
    ##################################################################
    # get group name to label mapping
    ##################################################################
    gdf=pd.read_csv(group_tsv,sep='\t')
    gdf.columns=['group','label']
    
    groupname2label={}
    for group,label in zipdf(gdf,['group','label']):
        groupname2label[group]=label
        
    ##################################################################
    # determine all possible starts this site can occur at
    ##################################################################
    
    thiscmdf=cmdf.loc[cmdf['site-id']==siteid,:].reset_index(drop=True)
    if len(thiscmdf)>1: raise ValueError('multiple site ids with same number')
    if len(thiscmdf)==0: raise ValueError('this site id is not in the dataframe')
    
    possibleStarts=thiscmdf.at[0,'start-set']
    edf=pd.read_csv(enhancer_tsv,sep='\t')
    edf.columns=['enhancer-name','enhancer-sequence','group']
    
    ##################################################################
    # get data for this pfm
    ##################################################################
    
    thisdf=lmdf.loc[lmdf['site-id']==siteid,:].reset_index(drop=True)

    # get  pfm id
    if thisdf['tf-definition'].nunique()!=1: raise ValueError
    pwmid=thisdf.at[0,'tf-definition']
    
    # get strand
    if thisdf['strand'].nunique()!=1: raise ValueError
    strand=thisdf.at[0,'strand']
    
    # get this kmer length
    if thisdf['kmer-length'].nunique()!=1: raise ValueError
    kmerLength=thisdf.at[0,'kmer-length']
    
    ##################################################################
    # determine reference kmer for plotting
    ##################################################################
    
    # if gain of function, you want the reference kmer to be the highest test scoring kmer. 
    # when you plot all other sequences, the best match to this sequence will be plotted (out of the possible kmers adjusting for indels)
    if hypothesis=='gof':
        tempdf=thisdf.sort_values('tf-strength-value',ascending=False).reset_index(drop=True)
        refkmer=tempdf.at[0,'kmer']
        
    elif hypothesis=='lof':
        tempdf=thisdf.sort_values('tf-strength-value',ascending=True).reset_index(drop=True)
        refkmer=tempdf.at[0,'kmer']
        
    else: raise ValueError('hypothesis must be either gof or lof')
    
    # plot the pfm
    # print('plotting pwm...')
    fig,ax=sequence_logo_generator(jasparMotifFile,isAlreadyPwm=isAlreadyPwm,isFraction=isFraction, dpi=dpi, pwm_file_format=pwm_file_format)
    

    # get kmers
    y=-.1
    grouplabel2color={'wild-type':'black','test':'purple','control':'grey'}
        
    groupyposadj=1.25
    
    for datakey,grouplabel in [('dw','wild-type'),('dt','test'),('dc','control')]:
        
        # print('analyzing',grouplabel,'...')
        textcolor=grouplabel2color[grouplabel]
        thissite=lmdf['site-id']==siteid
        thisgroup=lmdf['group-label']==grouplabel
        
        if   hypothesis=='gof': ascending=False
        elif hypothesis=='lof' :ascending=True
        else: raise ValueError("hypothesis must be either gof or lof")
        
        thisdf=lmdf.loc[thissite&thisgroup,:].sort_values('tf-strength-value',ascending=ascending)

        # if there are hits, simply plot these
        if len(thisdf)>0:

            for kmer,name,group,strength in zipdf(thisdf,['kmer','enhancer-name','group','tf-strength-value']):
                
                # save wt kmer
                if grouplabel=='wild-type':
                    wtKmer=kmer
                    
                # skip if kmer is same as wt
                if containsWt and grouplabel!='wild-type' and kmer==wtKmer:
                    continue
                
                # plot kmer
                for x in range(len(kmer)):
                    ax.text(x,y,kmer[x],ha='center',va='center',size=15,color=textcolor)
                    
                scorePC=all_funcs_get_score_of_kmer(pwmPC,kmer,maxPwmScorePC,minPwmScorePC)
                
                # plot meta data
                ax.text(-1,y,grouplabel,color=textcolor,va='center',ha='right')
                ax.text(kmerLength,y,f'{strength:.2f} ({scorePC:.2f})',color=textcolor,va='center',ha='left')
                ax.text(kmerLength+groupyposadj,y,name,color=textcolor,va='center',ha='left')
                    
                y-=.1

                
        # if there are no hits, check all possible starts and report the kmer that has the best match to the test
        else:
            for name,group,seq in zipdf(edf,['enhancer-name','group','enhancer-sequence']):
                if groupname2label[group]!=grouplabel:continue
                
                possibleKmers=[]
                
                for si in possibleStarts:

                    kmer=get_kmer_with_dashes(seq,start=si,strand=strand,length=kmerLength)
                    if not kmer: continue

                    lDist=levenshtein_distance(refkmer,kmer)
                    possibleKmers.append((lDist,kmer))

                possibleKmers=sorted(possibleKmers)
                
                bestKmer=possibleKmers[0][1]
                bestKmerScore=   all_funcs_get_score_of_kmer(pwm,bestKmer,maxPwmScore,minPwmScore)
                bestKmerScorePC= all_funcs_get_score_of_kmer(pwmPC,bestKmer,maxPwmScorePC,minPwmScorePC)
                
                # save wt kmer
                if grouplabel=='wild-type':
                    wtKmer=bestKmer
                
                # skip if kmer is same as wt
                if containsWt and grouplabel!='wild-type' and bestKmer==wtKmer:
                    continue
                    
                for x in range(len(bestKmer)):
                    ax.text(x,y,bestKmer[x],ha='center',va='center',size=15,color=textcolor)
                    
                # meta data
                ax.text(-1,y,grouplabel,color=textcolor,va='center',ha='right')
                ax.text(kmerLength,y,f'{bestKmerScore:.2f} ({bestKmerScorePC:.2f})',color=textcolor,va='center',ha='left')
                ax.text(kmerLength+groupyposadj,y,name,color=textcolor,va='center',ha='left')
                y-=.1
                
        # make some space between the group labels    
        y-=.05        
    
    # other aes
    ax.text(-1,0,'label',va='center',ha='right')
    ax.text(kmerLength,0,'score',va='center',ha='left')
    ax.text(kmerLength+groupyposadj,0,'name',va='center',ha='left')

    if outfn:
        # plt.savefig(outfn+'.svg',format='svg',bbox_inches='tight')
        plt.savefig(outfn+'.png',format='png',bbox_inches='tight')
        plt.close(fig)
        
        for name in list(locals().keys()):
            del locals()[name]
            
    else:
        
        for name in list(locals().keys()):
            del locals()[name]
            
        return fig,ax
        
def generate_core_report_fig(lmdf,
                             cmdf,
                             enhancer_tsv,
                             group_tsv,
                             siteid,
                             dpi,
                             hypothesis,
                             containsWt,
                             seq2aff=None,outfn=None):
    
    ##################################################################
    # get group name to label mapping
    ##################################################################
    gdf=pd.read_csv(group_tsv,sep='\t')
    gdf.columns=['group','label']
    
    groupname2label={}
    for group,label in zipdf(gdf,['group','label']):
        groupname2label[group]=label
        
    ##################################################################
    # determine all possible starts this site can occur at
    ##################################################################
    
    thiscmdf=cmdf.loc[cmdf['site-id']==siteid,:].reset_index(drop=True)
    if len(thiscmdf)>1: raise ValueError('multiple site ids with same number')
    if len(thiscmdf)==0: raise ValueError('this site id is not in the dataframe')
    
    possibleStarts=thiscmdf.at[0,'start-set']
    edf=pd.read_csv(enhancer_tsv,sep='\t')
    edf.columns=['enhancer-name','enhancer-sequence','group']
    
    ##################################################################
    # get data for this site
    ##################################################################
    
    thisdf=lmdf.loc[lmdf['site-id']==siteid,:].reset_index(drop=True)
    
    # get strand
    if thisdf['strand'].nunique()!=1: raise ValueError
    strand=thisdf.at[0,'strand']
    
    # get this kmer length
    if thisdf['kmer-length'].nunique()!=1: raise ValueError
    kmerLength=thisdf.at[0,'kmer-length']
    
    # get this core site
    if thisdf['tf-definition'].nunique()!=1: raise ValueError
    coreSite=thisdf.at[0,'tf-definition']
    
    ##################################################################
    # determine reference kmer for plotting
    ##################################################################
    
    # if gain of function, you want the reference kmer to be the highest test scoring kmer. 
    # when you plot all other sequences, the best match to this sequence will be plotted (out of the possible kmers adjusting for indels)
    if hypothesis=='gof':
        tempdf=thisdf.sort_values('tf-strength-value',ascending=False).reset_index(drop=True)
        refkmer=tempdf.at[0,'kmer']
        
    elif hypothesis=='lof':
        tempdf=thisdf.sort_values('tf-strength-value',ascending=True).reset_index(drop=True)
        refkmer=tempdf.at[0,'kmer']
        
    else: raise ValueError('hypothesis must be either gof or lof')
    
    ##################################################################
    # plot the core
    ##################################################################
    
    fig,ax=plt.subplots(1,figsize=(kmerLength/10,4),dpi=dpi)
    
    for xi,nt in enumerate(coreSite):
        ax.text(xi,0,nt,ha='center',va='bottom',size=30, fontfamily='Courier New')
        
    # make re object out of the core site
    reCoreSite=IupacToRegexPattern(coreSite)

    ##################################################################
    # plot the other kmers
    ##################################################################
    
    y=-.1
    grouplabel2color={'wild-type':'black','test':'purple','control':'grey'}
        
    groupyposadj=1.25
    
    for datakey,grouplabel in [('dw','wild-type'),('dt','test'),('dc','control')]:
        
        # print('analyzing',grouplabel,'...')
        textcolor=grouplabel2color[grouplabel]
        thissite=lmdf['site-id']==siteid
        thisgroup=lmdf['group-label']==grouplabel
        
        if   hypothesis=='gof': ascending=False
        elif hypothesis=='lof' :ascending=True
        else: raise ValueError("hypothesis must be either gof or lof")
        
        thisdf=lmdf.loc[thissite&thisgroup,:].sort_values('tf-strength-value',ascending=ascending)

        # if there are hits, simply plot these
        if len(thisdf)>0:

            for kmer,name,group,strength in zipdf(thisdf,['kmer','enhancer-name','group','tf-strength-value']):
                
                # save wt kmer
                if grouplabel=='wild-type':
                    wtKmer=kmer
                    
                # skip if kmer is same as wt
                if containsWt and grouplabel!='wild-type' and kmer==wtKmer:
                    continue
                
                # plot kmer
                for x in range(len(kmer)):
                    ax.text(x,y,kmer[x],ha='center',va='center',size=15,color=textcolor)
                    
                # plot meta data
                ax.text(-1,y,grouplabel,color=textcolor,va='center',ha='right')
                ax.text(kmerLength,y,f'{strength:.2f}',color=textcolor,va='center',ha='left')
                ax.text(kmerLength+groupyposadj,y,name,color=textcolor,va='center',ha='left')
                    
                y-=.1

                
        # if there are no hits, check all possible starts and report the kmer that has the best match to the test
        else:
            for name,group,seq in zipdf(edf,['enhancer-name','group','enhancer-sequence']):
                
                # skip enhancers which are not in this group lable
                if groupname2label[group]!=grouplabel:continue
                
                # check all possible kmers given by possible starts to see which is the best match without any indels
                possibleKmers=[]
                for si in possibleStarts:

                    kmer=get_kmer_with_dashes(seq,start=si,strand=strand,length=kmerLength)
                    if not kmer: continue
                    
                    lDist=levenshtein_distance(refkmer,kmer)
                    possibleKmers.append((lDist,kmer))
                possibleKmers=sorted(possibleKmers)
                
                bestKmer=possibleKmers[0][1]
                
                # get score if the kmer abides by the iupac
                if seq2aff:
                    if re.search(reCoreSite,bestKmer):
                        bestKmerAff = seq2aff[bestKmer]
                        bestKmerAff = f'{bestKmerAff:.2f}'
                    else:
                        bestKmerAff = 'N.C.'
                else:
                    bestKmerAff = '-'
                
                # save wt kmer
                if grouplabel=='wild-type':
                    wtKmer=bestKmer
                
                # skip if kmer is same as wt
                if containsWt and grouplabel!='wild-type' and bestKmer==wtKmer:
                    continue
                    
                for x in range(len(bestKmer)):
                    ax.text(x,y,bestKmer[x],ha='center',va='center',size=15,color=textcolor)
                    
                # meta data
                ax.text(-1,y,grouplabel,color=textcolor,va='center',ha='right')
                ax.text(kmerLength,y,bestKmerAff,color=textcolor,va='center',ha='left')
                ax.text(kmerLength+groupyposadj,y,name,color=textcolor,va='center',ha='left')
                y-=.1
                
        # make some space between the group labels    
        y-=.05        
    
    # other aes
    ax.text(-1,0,'label',va='center',ha='right')
    ax.text(kmerLength,0,'affinity',va='center',ha='left')
    ax.text(kmerLength+groupyposadj,0,'name',va='center',ha='left')
    
    clearspines(ax,sides=['top','bottom','left','right',])
    clearticks(ax,sides=['x','y'])
    
    plt.tight_layout()
    
    # ax.set_ylim(y,1)

    if outfn:
        plt.savefig(outfn+'.png',format='png',bbox_inches='tight')
        # plt.savefig(outfn,format='svg',bbox_inches='tight')
        plt.close(fig)
    else:
        return fig,ax
    
def whitespace(n):
    return '&nbsp;'*n

def encode_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_section(cmdf,lmdf,siteid,hypothesis,datadir='./'):
    
    if hypothesis not in ['gof','lof']: raise ValueError('hypothesis must be either gof or lof')
    
    pfmid=cmdf.at[siteid,'tf-definition']
    tfname=cmdf.at[siteid,'tf-name']
    dettype=cmdf.at[siteid,'tf-definition']
    kmerLength=cmdf.at[siteid,'kmer-length']
    
    # print(pfmid)
    
    ###########################################
    # define title
    ###########################################
    
    title=f'''
    <div>
    TF Name:   {tfname}<br>
    '''
    
    # add pfm id if appropriate
    if dettype=='score-pwm':
        title+=f'PFM ID:    {pfmid}<br>'
    
    title+='<br></div>'
    title=textwrap.dedent(title)
    
    ###########################################
    # get title
    ###########################################
    
    whiteSpaceBuffer=2
    fontsize=20

    tab='&nbsp;'*(whiteSpaceBuffer)
        
    data = {
        'siteid':siteid,
        'title':title,
        'kmerplot':encode_image(f'{datadir}/kmer-images-{hypothesis}/{siteid}.png'),
        'siteplot':encode_image(f'{datadir}/score-images/site-{siteid}.png')}
    
    grouplabel2stringWithHtmlWhitespace={
        'control':'control'+whitespace(2+whiteSpaceBuffer),
        'test':'test'+whitespace(5+whiteSpaceBuffer),
        'wild-type':'wild-type'+whitespace(whiteSpaceBuffer)
    }
    
    ###########################################
    # Create a Markdown template
    ###########################################
    
    md_template = """
    #Site ID:   {{ siteid }} <br>
    {{ title }}
    <br>
    <img src="{{ siteplot }}" alt="Report Image" style="width: 400px;"><br>
    <br>
    <img src="{{ kmerplot }}" alt="Report Image" style="width: 600px;"><br>
    <br>
    <div>
    {% for item in dw %}
    {{ item.grouplabel }} {{ item.kmer }}[[ ]]{{ item.strength }}[[ ]]{{ item.enhancername }}<br>
    {% endfor %}
    <br>
    {% for item in dt %}
    {{ item.grouplabel }} {{ item.kmer }}[[ ]]{{ item.strength }}[[ ]]{{ item.enhancername }}<br>
    {% endfor %}
    <br>
    {% for item in dc %}
    {{ item.grouplabel }} {{ item.kmer }}[[ ]]{{ item.strength }}[[ ]]{{ item.enhancername }}<br>
    {% endfor %}
    </div>
    <br>
    <br>
    <hr>
    <br>
    <br>
    """.replace('[[fontsize]]',str(fontsize)).replace('[[width]]',str(kmerLength*fontsize*.686)).replace('[[ ]]','&nbsp;&nbsp;&nbsp;')
    
    md_template=textwrap.dedent(md_template)

    # Render the template with data
    template = Template(md_template)
    md_text = template.render(data)
    
    html = markdown.markdown(md_text,extensions=['nl2br'])
    
    # with open('test.html','w') as f: f.write(html)
    
    return html

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")
    
def generate_html_report(cmdf,lmdf,
                         enhancer_tsv,
                         group_tsv,
                         input_tf_sheet_df,
                         input_pwm_jaspar_formatted_batch_file,
                         hypothesis,
                         isAlreadyPwm,
                         isFraction,
                         pseudocounts,
                         minPwmThreshold=.6,
                         outdir='html',
                         siteidList=None,outputFirst=None,scoreVisType='swarm',
                         ignorePythonWarnings=False,
                         subsetTfDetectionType='',
                         lineprofiler=None,
                         pwm_file_format='jaspar'):
    
    '''
    Generates HTML report for results of compare seqs
    
    Inputs
      cmdf - pandas DataFrame - cmdf output of compare_seqs_pipeline()
      lmdf - pandas DataFrame - lmdf output of compare_seqs_pipeline()
      enhancer_tsv - str - same input used for compare_seqs_pipeline()
      group_tsv -str - same input used for compare_seqs_pipeline()
      input_tf_sheet_df - str - same input used for compare_seqs_pipeline()
      input_pwm_jaspar_formatted_batch_file - str - same input used for compare_seqs_pipeline()
      hypothesis - str - lof or gof. If you want to find LOF binding sites, choose "lof". If you want to find gain of binding sites, use "gof".
      minPwmThreshold - float - the minimum y axis value that will be used for plotting scores of binding sites
      outdir - str - output directory for all output files
      siteidList - list - site ids to print results for
      outputFirst - int - only output the first X hits inthe HTML report
      scoreVisType - str - visualization style for plots of binding scores/affinities. Can be "swarm" "violin" or "strip"
      ignorePythonWarnings - bool - sometimes these kind of plots generate too many python warnings. If this option is set to True, no python warnings will print.
      subsetTfDetectionType - str - will only print "pwm" or "core" hits. Can only be set to "pwm" or "core".
    '''
    
    if ignorePythonWarnings:
        warnings.filterwarnings("ignore")

    
    mkdir_if_dir_not_exists(f'{outdir}')
    mkdir_if_dir_not_exists(f'{outdir}/pfm-files/')
    mkdir_if_dir_not_exists(f'{outdir}/score-images/')
    
    
    cmdf=cmdf.set_index('site-id',drop=False)
    
    if subsetTfDetectionType:
            
        if subsetTfDetectionType=='pwm':
            rowsPlottedcmdf = cmdf['tf-strength-type']=='score-pwm'
            rowsPlottedlmdf = lmdf['tf-strength-type']=='score-pwm'
            
        elif subsetTfDetectionType=='core':
            rowsPlottedcmdf = cmdf['tf-strength-type'].isin(['core-only','core-affinity'])
            rowsPlottedlmdf = lmdf['tf-strength-type'].isin(['core-only','core-affinity'])
            
        else:
            raise ValueError('subsetTfDetectionType must be either pwm or core')
            
        # cmdf=cmdf.copy(deep=True)
        thiscmdf=cmdf.loc[rowsPlottedcmdf,:]

        # lmdf=lmdf.copy(deep=True)
        thislmdf=lmdf.loc[rowsPlottedlmdf,:]
        
    else:
        thiscmdf=cmdf
        thislmdf=lmdf
        
    
    ############################################################
    # read core ref info
    ############################################################
    
    if input_tf_sheet_df is not None:
        # tfdf=pd.read_csv(input_tf_sheet_tsv,sep='\t')
        # tfdf.columns=['tfname','core','aff-ref']
    
        # tf2affref={}
        # for tfname,affref in zipdf(tfdf,['tfname','aff-ref']):
        #     tf2affref[tfname]=affref
        # # print(tf2affref)
    
        input_tf_sheet_df=input_tf_sheet_df.set_index('tfname')
    
    
    #######################################################################################
    print('loading pwm and affinity reference files... and plotting site id scores...')
    #######################################################################################
    
    if hypothesis=='gof' or hypothesis=='both':
        mkdir_if_dir_not_exists(f'{outdir}/kmer-images-gof/')
    if hypothesis=='lof' or hypothesis=='both':
        mkdir_if_dir_not_exists(f'{outdir}/kmer-images-lof/')
    if hypothesis not in ['lof','gof']: raise ValueError('hypothesis must be gof or lof')
    
    siteid2pfmid={}
    pfmid2siteidset={}
    siteid2tfname={}
    tfname2seq2aff={}
    siteid2tfbstype={}
        
    for detectiontype,siteid,definition,tfname,gofrank,lofrank in zipdf(thiscmdf,['tf-strength-type','site-id','tf-definition','tf-name','gof-rank','lof-rank']):
        
        if hypothesis=='gof' and pd.isnull(gofrank): continue
        if hypothesis=='lof' and pd.isnull(lofrank): continue
        
        # skip siteid if its not included in the custom print lit
        if siteidList:
            if siteid not in siteidList:
                continue
                
        siteid2tfbstype[siteid]=detectiontype
        
        if detectiontype=='score-pwm':
            
            pfmid=definition
            
            if pfmid not in pfmid2siteidset:
                pfmid2siteidset[pfmid]=set()
                
            pfmid2siteidset[pfmid].add(siteid)
            siteid2pfmid[siteid]=pfmid
            
            if lineprofiler=='visualize_site':
                lp = LineProfiler()
                lp_wrapper = lp(visualize_site)    
                lp_wrapper(siteid,thislmdf,plotType=scoreVisType,ylim=(minPwmThreshold,1.05),title=siteid,highlightSeqs=[],outfn=f'{outdir}/score-images/site-{siteid}.png')
            else:
                visualize_site(siteid,thislmdf,plotType=scoreVisType,ylim=(minPwmThreshold,1.05),title=siteid,highlightSeqs=[],outfn=f'{outdir}/score-images/site-{siteid}.png')
            
        elif detectiontype in ['core-affinity','core-only']:
            
            siteid2tfname[siteid]=tfname
            if input_tf_sheet_df is not None:
                affrefFn=input_tf_sheet_df.at[tfname,'aff-ref']
            if pd.isna(affrefFn): 
                seq2aff=None
            else:      
                if tfname in tfname2seq2aff:
                    seq2aff=tfname2seq2aff[tfname]
                else:
                    seq2aff=loadNormalizedFile(affrefFn)
                    tfname2seq2aff[tfname] = seq2aff
                
                
            
            visualize_site(siteid,thislmdf,plotType=scoreVisType,ylim=(0,1.05),title=siteid,highlightSeqs=[],outfn=f'{outdir}/score-images/site-{siteid}.png')
            
        

        
    ############################################################
    print('extracting pfms...')
    ############################################################
    
    if subsetTfDetectionType == 'pwm':
        if pwm_file_format == 'jaspar':
            for idx, entry in enumerate(read_in_chunks(input_pwm_jaspar_formatted_batch_file,5)):
                #info = entry[0].split()
                pfmid = None  # Maggie

                
                header = entry[0].split('\t')[0].replace('>', '')
                match = re.search(r'MA.*?\.\d+', header)
                if match:
                    pfmid = match.group()

                elif re.search(r'([^.\s]+\.){2,}[^.\s]+',  entry[0]): 
                    pfmid = entry[0].lstrip('>').strip()
                    #print(pfmid)
                else:
                    raise ValueError(f'{entry[0].strip()}')
                if pfmid in pfmid2siteidset:

                    entry='\n'.join(entry)
                    with open(f'{outdir}/pfm-files/{pfmid}.jaspar.txt','w') as f: f.write(entry)
               
        
        elif pwm_file_format== 'uniprobe':
            filename = os.path.basename(input_pwm_jaspar_formatted_batch_file)
            motif_name = os.path.splitext(filename)[0].split('_')[0]
            with open (input_pwm_jaspar_formatted_batch_file, 'r') as f:
                motif_id = []
                is_number = 0
                to_print = []
                for line in f:
                    if line.strip() == "":
                        continue
                    numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf', line)
                    if len(numbers) >= 5:
                        if is_number == 0:
                            motif_id = motif_name+' '+' '.join(motif_id)
                        
                        if motif_id in pfmid2siteidset:
                         
                            if motif_id not in to_print:
                                to_print.append(motif_id)
                            to_print.append(list(map(float, numbers)))
                        else:
                            to_print = []
                        is_number+=1
                        if is_number == 4:
                            is_number = 0
                            if len(to_print)>=4:
                                with open(f'{outdir}/pfm-files/{motif_id}.jaspar.txt','w') as f: 
                                    for row in to_print:
                                        if isinstance(row, str):
                                            f.write(row + '\n')
                                        else:
                                            f.write('\t'.join(str(x) for x in row) + '\n')
                            motif_id = []
                    else: 
                        motif_id.append(line.strip())
        else:
            raise ValueError('Only file format from jaspar and uniprobe are accepted')

    ############################################################
    print('visualize scores for site ids...')
    ############################################################
    
    for pfmid in pfmid2siteidset:
        for siteid in pfmid2siteidset[pfmid]:
            
            # visualize sites
            # visualize_site(siteid,lmdf,plotType=scoreVisType,ylim=(.6,1.05),title=siteid,highlightSeqs=[],outfn=f'{outdir}/score-images/site-{siteid}.png')
            pass
            

    ############################################################
    print('generate kmer images...')
    ############################################################
        
    # if you are subsetting the detection type, create a suffix to add to the file
    if subsetTfDetectionType:
        subsetTfDetectionType=f'_{subsetTfDetectionType}'
        
    # determine sites to include in the report
    if siteidList:
        jobName=f'customlist{subsetTfDetectionType}.html'
        jobList=[]
        if hypothesis=='gof' or hypothesis=='both':
            jobList+=[('gof',jobName,siteidList)]
        if hypothesis=='lof' or hypothesis=='both':
            jobList+=[('lof',jobName,siteidList)]
        
    # if sites not provided, analyze all GOF and LOF
    else:

        sortCols=[]
        if hypothesis=='both' or hypothesis=='lof': 
            sortCols.append(('lof',f'lof{subsetTfDetectionType}.html','lof-rank'))
            
        if hypothesis=='both' or hypothesis=='gof': 
            sortCols.append(('gof',f'gof{subsetTfDetectionType}.html','gof-rank'))
            
        jobList=[]
        for hi,jobname,col in sortCols:
            thisdf=thiscmdf.sort_values(col).set_index('site-id')
            thisdf=thisdf.loc[~thisdf[col].isna(),:]
            theseSiteIds=thisdf.index
            if outputFirst: theseSiteIds=theseSiteIds[:outputFirst]
            jobList.append((hi,jobname,theseSiteIds))
            # print(theseSiteIds)
            
    njobs=0
    for hi,_,siteidList in jobList:
        for siteid in siteidList:
            njobs+=1
            
    jobsdone=0
    for hi,_,siteidList in jobList:
        for siteid in siteidList:
            
            # print_memory_usage()
            
            if jobsdone % 2  ==0: 
                percentdone=round(100*jobsdone/njobs,2)
                print(f'{percentdone}%, ',end='')
                
            jobsdone+=1
                
            detectiontype=siteid2tfbstype[siteid]
            
            if detectiontype=='score-pwm':
                pfmid=siteid2pfmid[siteid]
                jasparMotifFile=f'{outdir}/pfm-files/{pfmid}.jaspar.txt'
                if lineprofiler=='generate_pwm_kmer_report_fig':
                    lp = LineProfiler()
                    lp_wrapper = lp(generate_pwm_kmer_report_fig)    
                    lp_wrapper(lmdf=thislmdf,
                                                 cmdf=thiscmdf,
                                                 enhancer_tsv=enhancer_tsv,
                                                 group_tsv=group_tsv,
                                                 siteid=siteid,
                                                 jasparMotifFile=jasparMotifFile,
                                                 dpi=100,isAlreadyPwm=isAlreadyPwm,
                                                 hypothesis=hi,pseudocounts=pseudocounts,
                                                 containsWt=True,outfn=f'{outdir}/kmer-images-{hi}/{siteid}.png',
                                                 pwm_file_format=pwm_file_format)
                else:
                    generate_pwm_kmer_report_fig(lmdf=thislmdf,
                                                 cmdf=thiscmdf,
                                                 enhancer_tsv=enhancer_tsv,
                                                 group_tsv=group_tsv,
                                                 siteid=siteid,
                                                 jasparMotifFile=jasparMotifFile,
                                                 dpi=100,isAlreadyPwm=isAlreadyPwm,
                                                 isFraction=isFraction,
                                                 hypothesis=hi,pseudocounts=pseudocounts,
                                                 containsWt=True,outfn=f'{outdir}/kmer-images-{hi}/{siteid}',
                                                 pwm_file_format=pwm_file_format)
                
            elif detectiontype in ['core-only','core-affinity']:
            
                generate_core_report_fig(lmdf=thislmdf,
                                         cmdf=thiscmdf,
                                         enhancer_tsv=enhancer_tsv,
                                         group_tsv=group_tsv,
                                         siteid=siteid,
                                         dpi=100,
                                         hypothesis=hi,
                                         containsWt=True,
                                         seq2aff=seq2aff,outfn=f'{outdir}/kmer-images-{hi}/{siteid}')
                
            else: raise ValueError('detectiontype must be either score-pwm, core-only or core-affinity')
                        
    ############################################################
    print('generate html...')
    ############################################################
    
    
    for hi,htmlout,siteidList in jobList:        
        
        # begin html
        html='''    
        <style>
            body {
                font-family: Courier, monospace;
                font-size: [[fontsize]]px;
            }
        </style>
        
        Notes: <br>
         - Scores for PWM are given without pseudocounts and with pseudocounts within parenthesis (e.g. 0.88 (0.91)) <br>
         - Kmers are not reported in the kmer plots which have the same sequence as wild-type <br>
         - Affinities reported as N.C. indicate that the dna sequence has no TF binding core sequence as specified by the user <br>
         - When binding site cores are provided without affinities, the affinity will be plotted in this report as "-" <br>
        '''

        for siteid in siteidList:

            html+=generate_section(thiscmdf,thislmdf,siteid,datadir=outdir,hypothesis=hi)

        # print(f'{outdir}/{htmlout}')
        with open(f'{outdir}/{htmlout}','w') as f: f.write(html)
        
    if lineprofiler: lp.print_stats()


# wrapper function
def compare_seqs_wrapper(enhancer_dna_alignment_table, 
                         enhancer_functional_group_table, 
                         tf_affinity_information=None, #converted to df to account for the gene pattern tf aff files parameter
                         pwm_input=None, 
                         isAlreadyPwm=False,
                         isFraction=False,
                         pseudocounts=False,
                         minimum_binding_change=0.1, 
                         minimum_pwm_score=0.8, 
                         hypothesis='both', 
                         betaPwm = False,
                         output_name=None, 
                         outdir='./',
                         pwm_file_format='jaspar'):
    
    # make dir if it doesn't exist
    mkdir_if_dir_not_exists(outdir)
    
    # check boolean inputs
    isAlreadyPwm = check_bool(isAlreadyPwm)
    isFraction = check_bool(isFraction)
    pseudocounts = check_bool(pseudocounts)
    
    # make sure either aff or pfm data is provided
    if (tf_affinity_information is None) and (pwm_input is None):
        raise ValueError('At least one of affinity or pwm data must be provided.')
    
    # compare seqs function
    enhancer_fn_containsWt=False
    for line in read_tsv(enhancer_functional_group_table,header=True,pc=False):
        group=line[1]
        if group=='wild-type': enhancer_fn_containsWt=True

    ldf,mdf,lmdf,cmdf=\
        compare_seqs_pipeline(enhancer_tsv=enhancer_dna_alignment_table, 
                              containsWildType=enhancer_fn_containsWt,
                              group_tsv=enhancer_functional_group_table,
                              isAlreadyPwm=isAlreadyPwm,
                              isFraction=isFraction,
                              pseudocounts=pseudocounts,
                              input_tf_sheet_df=tf_affinity_information,
                              input_pwm_batch_fn=pwm_input,
                              deltamin=minimum_binding_change,
                              plotMsa=False,
                              trimAlignment=(None,None),
                              ifTrim_removeSameAsWt=True,
                              outdir=outdir,
                              scoreRound=3,
                              pwm_min_score=minimum_pwm_score,
                              betaPwm=betaPwm,
                              pwmPrintProgress=100,
                              ignorePythonWarnings=True,
                              pwm_file_format = pwm_file_format)
    
    if type(lmdf)==bool: 
        return #if no hits are found then the length of the df will be zero

    # generate html report function
    searchfor=[]
    if hypothesis=='lof' or hypothesis=='both': searchfor+=[('lof','abl'),('lof','dec')]
    if hypothesis=='gof' or hypothesis=='both': searchfor+=[('gof','dnv'),('gof','inc')]

    # set value of subsetTfDetectionType based whether we are using pwm or aff data
    if tf_affinity_information is None:
        subsetTfDetectionType = 'pwm'
    else:
        subsetTfDetectionType = 'core'

    for hi,outputType in searchfor:
        
        noutputs=len(cmdf.loc[cmdf['variant-effect']==outputType,:])
        
        if noutputs==0:  print(f'[HTML] no {outputType} are detected... skipping...')
        else:            print(f'[HTML] {noutputs:,} hits for {outputType} are detected... starting now...')
            
        generate_html_report(cmdf.loc[cmdf['variant-effect']==outputType,:],
                             lmdf,
                             enhancer_tsv=enhancer_dna_alignment_table,
                             group_tsv=enhancer_functional_group_table,
                             input_tf_sheet_df=tf_affinity_information,
                             input_pwm_jaspar_formatted_batch_file=pwm_input,
                             hypothesis=hi,
                             isAlreadyPwm=isAlreadyPwm,
                             isFraction=isFraction,
                             pseudocounts=pseudocounts,
                             minPwmThreshold=minimum_pwm_score,
                             outdir=f'{outdir}/html-{outputType}/',
                             siteidList=None,  
                             outputFirst=None,
                             scoreVisType='swarm',
                             ignorePythonWarnings=True,
                             subsetTfDetectionType=subsetTfDetectionType,
                             pwm_file_format=pwm_file_format)
def standardize_chr(val):
    val = str(val).replace('chr', '')
    return 'chr' + val
    
def standardize_col_names(gdf):
    gdf.columns=['chrom','pos','ref','alt','hypothesis']+list(gdf.columns)[5:]
    gdf['pos']=gdf['pos'].apply(lambda i: int(i))
    gdf['chrom'] = gdf['chrom'].apply(standardize_chr)
    
    return gdf

def check_gdf_format(gdf,chr2seq,posIdxType):

    gdf=standardize_col_names(gdf)
    
    # check data is as expected
    if not is_string_dtype(gdf['chrom']): raise ValueError('The first column must be the chromosome (e.g. chr1) and therefore should be able to be converted to python type "string".')
    if not is_integer_dtype(gdf['pos']): raise ValueError('The second column must be the position (e.g. 17356) and therefore should be able to be converted to python type "integer".')
    if not gdf['ref'].isin({'A', 'T', 'G', 'C','-'}).all(): raise ValueError('The third column must be the reference nucleotide (e.g. A/T/G/C) and therefore should be able to be converted to python type "string".  Currently TF Sites only allows for A/T/G/C/-, it does not allow for multiple bp deletions yet.')
    if not gdf['alt'].isin({'A', 'T', 'G', 'C','-'}).all(): raise ValueError('The fourth column must be the alternate nucleotide (e.g. A/T/G/C) and therefore should be able to be converted to python type "string". Currently TF Sites only allows for A/T/G/C/-, it does not allow for multiple bp deletions yet.')
    if not gdf['hypothesis'].str.lower().isin({'gof', 'lof', 'both', 'na'}).all(): raise ValueError('The fifth column must be the hypothesis (e.g. lof/gof/both) .')
                            
    # check if refs/pos are in the geonome
    for chrom,pos,ref,alt in zipdf(gdf,['chrom','pos','ref','alt']):
        
        if posIdxType==1: pos-=1
        
        if chrom not in chr2seq:
            chromList=list(chr2seq.keys())
            raise ValueError(f'Chromosome',chrom,f'not found in the reference genome. These are the chromosomes within the reference genome you provided:\n{chromList}')
            
        if pos > len(chr2seq[chrom]):
            maxPos=len(chr2seq[chrom])
            raise ValueError(f'Position {pos} not found in chromosome {chrom}. The length of the chromosome is {maxPos}.')
            
    return True

def percent(number,rounding_digit=1):
    '''Get percent of fraction'''
    if rounding_digit==0:
        return str(int(100*number))+'%'
    else:
        return str(round(100*number,rounding_digit))+'%'                        
    
def check_refs_against_genome(chr2seq,gdf,posIdxType):
    
    gdf=standardize_col_names(gdf)
    print('GDF has proper format:',check_gdf_format(gdf,chr2seq,posIdxType))
    
    # keep track if you are using the wrong indexing
    adjust2correct={-1:0,0:0,1:0}
    
    refFoundList=[]
    for chrom,pos,ref,alt in zipdf(gdf,['chrom','pos','ref','alt']):
        
        if posIdxType==1: pos-=1
        
        refFound=chr2seq[chrom][pos]
        refFoundList.append(refFoundList)
        
        if refFound==ref:                adjust2correct[0] +=1    
        elif chr2seq[chrom][pos-1]==ref: adjust2correct[-1]+=1
        elif chr2seq[chrom][pos+1]==ref: adjust2correct[1] +=1
        #else: raise ValueError('The ref is not found in the genome')

    n=len(gdf)
    print(f'% Accurate With {posIdxType}-indexing:       ', percent(adjust2correct[0]/n))
    print(f'% Accurate 1-bp left  of intended coordinate:', percent(adjust2correct[-1]/n))
    print(f'% Accurate 1-bp right of intended coordinate:', percent(adjust2correct[1]/n))
    
    if adjust2correct[0]/n < 1: print('[[WARNING]] The reference value of positions are off. The generated sequences will assume your table is correct, but you will want to double check these sequences are generated as expected.')

def seq_extraction(window, chr2seq, chrom, pos, ref, alt, posIdxType, returnRef=False):
   
    pos = pos
    if posIdxType==1:
        pos = pos-1
    r_len = len(ref)
    a_len = len(alt)
    fasta_seq = chr2seq[chrom]

    if returnRef:
        return fasta_seq[pos-window:pos+r_len+window]
        
    seq1 = str(fasta_seq[pos-window:pos])
    seq2 = str(fasta_seq[pos+r_len:pos+window+r_len])
    
    seq = ''
    if str(fasta_seq[pos:pos+r_len])!=ref:
        raise ValueError(f'The REF does not match. REF: {ref}. Genome: {fasta_seq[pos:pos+r_len]}')
                
    # if r_len == a_len:
    #     if alt != '.':
    #         seq = seq1+alt+seq2
    #     else:
    #         seq = seq1+('-'*r_len)+seq2
    
    # elif r_len < a_len: # insertion
    #     seq = seq1+alt+seq2
           
    # else: # deletion, r_len > a_len
    #     dif = r_len - a_len
    #     alt = alt+dif*'-'
    #     seq = seq1+alt+seq2
    if r_len == a_len:
        alt = alt if alt != '.' else '-'*r_len
    elif r_len > a_len:
        alt += '-'*(r_len-a_len)
    seq = seq1 + alt + seq2
    return seq

def write_row(rowList,delim='\t'):
    '''Write a single row of a tsv file.'''
    return delim.join([str(i) for i in rowList])+'\n'

def genotype_compare_seqs_wrapper(genome_file,
                                  variant_file,
                                  posIdxType,
                                  windowSize,
                                  tf_affinity_information=None,
                                  pwm_input=None,
                                  isAlreadyPwm=False,
                                  isFraction=False,
                                  minimum_binding_change=.1,
                                  minimum_pwm_score=.8,
                                  pseudocounts=False,
                                  outname='differential-binding-sites.tsv',
                                  outdir='results',
                                  pwm_file_format='jaspar'):
    '''
    genome_file: chromosome and corresponding sequence, must be a pickle file
    variant_file: table with at least five columns for: 'chrom','pos','ref','alt','hypothesis'
    posIdxType: 1 if the position index is 1-based, or 0 if it is 0-based
    windowSize: Number of nucleotides to include on each side of a variant when extracting the surrounding sequence.
    '''
    gdf = pd.read_csv(variant_file, sep='\t')
    gdf = standardize_col_names(gdf)
    
    # check if genome file is pickled by loading 
    chr2seq=None
    try:
        with open(genome_file, 'rb') as f:
            chr2seq = pickle.load(f)
    except Exception:
        raise ValueError('Input genome file must be a .pickle file')
    
    check_refs_against_genome(chr2seq,gdf,posIdxType)
    
    mkdir_if_dir_not_exists(outdir)
    refSeqList = []
    altSeqList = []
    
    # make sure column names are as follows
    gdf.columns = ['chrom', 'pos', 'ref', 'alt', 'hypothesis']

    for _, row in gdf.iterrows():
        #'chrom','pos','ref','alt','hypothesis'
        #print(row)
        chrom = row['chrom']
        pos = row['pos']
        ref = row['ref']
        alt = row['alt']
        hyp = row['hypothesis']
        if hyp=='na': 
             refSeqList.append('')
             altSeqList.append('')
             continue
        refsequence = seq_extraction(windowSize, chr2seq, chrom, pos, ref, alt, posIdxType, returnRef=True)
        altsequence = seq_extraction(windowSize, chr2seq, chrom, pos, ref, alt, posIdxType)
        refSeqList.append(refsequence)
        altSeqList.append(altsequence)

          ##################################################################
#         # analysis pipeline
#         ##################################################################
        ID=f'{chrom}_{pos}_{ref}_{alt}'
        thisAnalysisOutDir=f'{outdir}/{ID}'
        mkdir_if_dir_not_exists(thisAnalysisOutDir)
        
        line_out= write_row(['name','alignment','group'])
        line_out+=write_row([f'ref_{chrom}_{pos}_{ref}',refsequence,'ref'])
        line_out+=write_row([f'ref_{chrom}_{pos}_{alt}',altsequence,'alt'])
        enhancer_fn=f'{outdir}/{ID}/in1-enhancers.tsv'
        with open(enhancer_fn,'w') as f: f.write(line_out)
        
        line_out= write_row(['group','label'])
        line_out+= write_row(['ref','wild-type'])
        line_out+= write_row(['alt','test'])
        group_definition_fn=f'{outdir}/{ID}/in2-groups.tsv'
        with open(group_definition_fn,'w') as f: f.write(line_out)

        hypothesis = hyp
        #thisoutdir=thisAnalysisOutDir
        compare_seqs_wrapper(enhancer_dna_alignment_table=enhancer_fn, 
                              enhancer_functional_group_table=group_definition_fn, 
                              tf_affinity_information=tf_affinity_information, #converted to df to account for the gene pattern tf aff files parameter
                              pwm_input=pwm_input, 
                              isAlreadyPwm=isAlreadyPwm,
                              isFraction=isFraction,
                              pseudocounts=pseudocounts,
                              minimum_binding_change=minimum_binding_change, 
                              minimum_pwm_score=minimum_pwm_score, 
                              hypothesis=hyp, 
                              output_name=ID, 
                              outdir=thisAnalysisOutDir,
                              pwm_file_format = pwm_file_format)
        
    gdf['ref-window-seq']=refSeqList
    gdf['alt-window-seq']=altSeqList
    
    gdf.to_csv(f'{outdir}/{outname}', sep='\t', index=None)
        
    
################################################################
# Graveyard
################################################################
