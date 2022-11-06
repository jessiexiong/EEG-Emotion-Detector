import pandas as pd

'''
Takes txt file and changes it into a dataframe that we can work with
Expected output looks like this:
       Sample Index  EXG Channel 0  EXG Channel 1  EXG Channel 2  EXG Channel 3
0               0.0     727.623716     909.005124    1075.183838     858.761438
1               1.0     704.948702     902.737051    1061.907193     875.048702
2               1.0     743.432273     908.316982    1064.753257     856.207086
3               2.0     751.158906     898.866255    1073.934712     873.474204
4               2.0     724.467240     897.852742    1068.912026     893.497628
'''

# columns we're looking for
fields = ['Sample Index',' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2', ' EXG Channel 3']

# convert raw data frame
df = pd.read_csv('files/blink.txt', delimiter=',', skiprows=4, usecols = fields, header=0)

# original column names has space in front, remove the space
df.columns = df.columns.str.replace(r'^ ', '', regex=True)

print(df)

