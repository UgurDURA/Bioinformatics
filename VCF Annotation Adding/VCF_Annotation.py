
import allel
from attr import field

 
callset = allel.read_vcf('24sample.vcf')
# print(callset)
print(sorted(callset.keys()))

print(callset['samples'])

print(callset['variants/CHROM'])

print(callset['variants/QUAL'])

print(callset['calldata/GT'])


gt = allel.GenotypeArray(callset['calldata/GT'])
print(gt)


df = allel.vcf_to_dataframe('24sample.vcf',fields='*', alt_number=2)
print(df)

# allel.vcf_to_csv('24sample.vcf','Extracted_CHROM.csv', fields =['CHROM'])

# with open('Extracted_CHROM.csv', mode='r') as f:
#     print(f.read())


