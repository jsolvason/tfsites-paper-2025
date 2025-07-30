import matplotlib.pyplot as plt
import pickle 

def clearspines(ax,sides=['top','right']):
    for s in sides:
        ax.spines[s].set_visible(False)

def clearticks(ax,sides=['x','y']):
    if 'x' in sides: ax.set_xticks([])
    if 'y' in sides: ax.set_yticks([])
        
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

def revcomp(dna): 
	'''Takes DNA sequence as input and returns reverse complement'''
	inv={'A':'T','T':'A','G':'C','C':'G', 'N':'N','W':'W'}
	revcomp_dna=[]
	for nt in dna:
		revcomp_dna.append(inv[nt])
	return ''.join(revcomp_dna[::-1])

def zipdf(df,cols):
    return zip(*[df[c] for c in cols])

def dprint(d,n=0):
    '''Print a dictionary'''
    for i,(k,v) in enumerate(d.items()):
        if i<=n:
            print(k,v)

cb={}
cb['lightblue']= [i/255 for  i in [86,180,233]]
cb['green']    = [i/255 for  i in [0,158,115]]
cb['red']      = [i/255 for  i in [213,94,0]]
cb['yellow']   = [i/255 for  i in [240,228,66]]
cb['orange']   = [i/255 for  i in [230,159,0]]
cb['blue']     = [i/255 for  i in [0,114,178]]
cb['pink']     = [i/255 for  i in [204,121,167]]
cb['black']    = [i/255 for  i in [0,0,0]]
cb['orangenature'] = [i/255 for  i in [210,58,40]]

def loadAff(ref):
        '''Load an arbitrary affinity dataset. First column should be 8mer DNA 
sequence and second column should be the normalized affinity (between 0-1).'''
        Seq2EtsAff  = {line.split('\t')[0]:float(line.split('\t')[1]) for line in open(ref,'r').readlines()}
        return Seq2EtsAff

def quickfig(x=5,y=5,dpi=150):
    return plt.subplots(1,figsize=(x,y),dpi=dpi,facecolor='white')

def load_pickle_dict(fn):
    with open(fn,'rb') as f:
        d=pickle.load(f)
    return d

def percent(number,rounding_digit=1):
    '''Get percent of fraction'''
    if rounding_digit==0:
        return str(int(100*number))+'%'
    else:
        return str(round(100*number,rounding_digit))+'%'  

from random import choice
import itertools
import re

import random

esp3i='CGTCTC'
bbsi='GAAGAC'
sbfi='CCTGCAGG'
kpni='GGTACC'
saci='GAGCTC'
psti='CTGCAG'
bseri='GAGGAG'
nhei='GCTAGC'

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

def dnaIn(dna,iterable):
	if dna in iterable: return True
	elif revcomp(dna) in iterable: return True
	else: return False

def count_with_revcomp(pattern,seq):    
	c=0
	for s in [seq,revcomp(seq)]:
		for kmer in get_kmers(s,len(pattern)):
			if kmer==pattern:
				c+=1
	# if pattern is palindrom, div by 2 so you dont count same twice
	if pattern==revcomp(pattern):
		return int(c/2)
	else:
		return c

def count_with_revcomp_re(re_pattern,length,seqFwd,returnList=False):
    '''A method to search dna sequence using re expression allowing for overlaps.'''
    returnListObj=[]
    for direction,seq in [('fwd',seqFwd),('rev',revcomp(seqFwd))]:
        for i,kmer in enumerate(get_kmers(seq,length)):
            if re.search(re_pattern,kmer): 
                returnListObj.append((direction,i,kmer))
    if returnList:
        return returnListObj
    else:
        return len(returnListObj)

def get_kmers(string,k):
	'''Takes DNA sequence as input and a kmer length and YIELDS all kmers of length K in the sequence.'''
	for i in range(len(string)):
		kmer8=string[i:i+k]
		if len(kmer8)==k:
			yield kmer8
			
def GenerateRandomDNA(length,gcContent=.5):
    '''Takes length as input and returns a random DNA string of that length.'''

    atContent=1-gcContent
    
    cp,gp=gcContent/2,gcContent/2
    tp,ap=atContent/2,atContent/2
    
    DNA=''.join(random.choices( ['C','G','T','A'], weights=(cp,gp,tp,ap),k=length))
        
    return DNA

def GenerateSingleRandomSequence(template):
        '''Takes a string with letters A,T,C,G,IUPAC and returns a DNA string with rnadom AGTC nucleotides at position(s) with N.'''
        dna=''
        for i,nt in enumerate(template):
                if nt not in ATGC:
                        dna+=choice(Iupac2AllNt[nt])
                else:
                        dna+=nt
        return dna

def GenerateAllPossibleSequences(template):
	'''Takes a string with letters A,T,C,G,N and returns all possible DNA strings converting N=>A,T,G,C'''
	randomSeqList=list(itertools.product('ATGC',repeat=template.count('N')))
	seqList=set()
	for seqPossibility in randomSeqList:
		seq=seqPossibility
		dnaOut=''
		for i,nt in enumerate(template):
			if nt == 'N':
				dnaOut+=seq[0]
				seq=seq[1:]
			else:
				dnaOut+=nt
		seqList.add(dnaOut)
	return seqList

def hamming(str1, str2):
	'''Takes 2 strings as input and returns hamming distance'''
	if len(str1) != len(str2):
		raise ValueError("Strand lengths are not equal!")
	else:
		return sum(1 for (a, b) in zip(str1, str2) if a != b)

def IupacToAllPossibleSequences(dna):
	'''Takes DNA with IUPAC letters and returns all possible DNA strings with only A/G/T/C.'''
	Round2Seqs={}
	Round2Seqs[0]=[]
	for nt in Iupac2AllNt[dna[0]]:
		Round2Seqs[0].append(nt)
		
#     print(Round2Seqs)
	for i,iupac in enumerate(dna):
		if i==0: continue
			
		Round2Seqs[i]=[]
		
		for seq in Round2Seqs[i-1]:
			for nt in Iupac2AllNt[iupac]:
#                 print(nt)
#                 print(seq)
				thisSeq=seq+nt
				Round2Seqs[i].append(thisSeq)
				
	lastRound=max(Round2Seqs.keys())
	return Round2Seqs[lastRound]

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

def revcomp_regex(pattern):
	'''Takes a DNA regex object and returns the reverse complement of that regex object.'''
	trans={'A':'T','G':'C','T':'A','C':'G','.':'.','(':')',')':'(','|':'|'}
	return ''.join([trans[i] for i in pattern])[::-1]

def gc_content(seq):
	'''Takes sequence as input and returns GC content from 0-1.0.'''
	return (seq.count('G')+seq.count('C'))/len(seq)

def find_all_matches_multiple_IUPAC_patterns(dna,iupac_pattern_list):
    '''Returns starts and strands of all iupacs. The start position is the left most 0-indexed position of the hit for + and - matches.'''
    
    iupac2len={iupac:len(iupac) for iupac in iupac_pattern_list}
    iupac2reo={iupac:IupacToRegexPattern(iupac) for iupac in iupac_pattern_list}
    
    positionsReported=set()
    for i in range(len(dna)):
        
        for iupac,l in iupac2len.items():
            
            pattern=iupac2reo[iupac]
            query=dna[i:i+l]
            
            for query_ori,strand in [(query,'+'),(revcomp(query),'-')]:
                if re.search(pattern,query_ori): 
                    yield (i,strand,iupac)
    
