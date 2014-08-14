""" 
Use price, class and gender to create a statistical model

Author : Abdulazeez Otori
Date : 12th August 2014

"""

import pandas as pd
import numpy as np
import csv as csv
from scipy.stats import bernoulli

# read training file
df_train = pd.read_csv('train.csv', header=0)

# In order to analyse the price column I need to bin up that data
# here are my binning parameters, the problem we face is some of the fares are very large
# So we can either have a lot of bins with nothing in them or we can just lose some
# information by just considering that anythng over 39 is simply in the last bin.
# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
df_train['CappedFare'] = df_train['Fare'].map(lambda x: min(x, fare_ceiling-1))

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size
number_of_classes = len(df_train.Pclass.unique())       # But it's better practice to calculate this from the Pclass directly:
                                                  # just take the length of an array of UNIQUE values in column index 2

# This reference matrix will show the proportion of survivors as a sorted table of
# gender, class and ticket fare.
# First initialize it with all zeros
survival_table = np.zeros([2,number_of_classes,number_of_price_brackets],float)

# I can now find the stats of all the women and men on board
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        
        women_only_stats =df_train[(df_train.Sex == "female") & (df_train.Pclass == i+1) & (df_train.CappedFare >= j*fare_bracket_size) & (df_train.CappedFare < (j+1)*fare_bracket_size)].Survived
        men_only_stats = df_train[(df_train.Sex != "female") & (df_train.Pclass == i+1) & (df_train.CappedFare >= j*fare_bracket_size) & (df_train.CappedFare < (j+1)*fare_bracket_size)].Survived

        survival_table[0,i,j] = women_only_stats.astype(float).mean()  # Female stats
        survival_table[1,i,j] = men_only_stats.astype(float).mean()    # Male stats
 
# Since in python if it tries to find the mean of an array with nothing in it
# (such that the denominator is 0), then it returns nan, we can convert these to 0
# by just saying where does the array not equal the array, and set these to 0.
survival_table[ survival_table != survival_table ] = 0.

# Now I have my indicator I can read in the test file and write out
# if a women then survived(1) if a man then did not survived (0)
# First read in test
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Also open the a new file so I can write to it. 
predictions_file = open("genderClassProbabilityModel.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])

# First thing to do is bin up the price file
for row in test_file_object:
    for j in xrange(number_of_price_brackets):
        # If there is no fare then place the price of the ticket according to class
        try:
            row[8] = float(row[8])    # No fare recorded will come up as a string so
                                      # try to make it a float
        except:                       # If fails then just bin the fare according to the class
            bin_fare = 3 - float(row[1])
            break                     # Break from the loop and move to the next row
        if row[8] > fare_ceiling:     # Otherwise now test to see if it is higher
                                      # than the fare ceiling we set earlier
            bin_fare = number_of_price_brackets - 1
            break                     # And then break to the next row

        if row[8] >= j*fare_bracket_size\
            and row[8] < (j+1)*fare_bracket_size:     # If passed these tests then loop through
                                                      # each bin until you find the right one
                                                      # append it to the bin_fare
                                                      # and move to the next loop
            bin_fare = j
            break
    # Now I have the binned fare, passenger class, and whether female or male, we can
    # just cross ref their details with our survival table
    if row[3] == 'female':
        p = survival_table[ 0, float(row[1]) - 1, bin_fare ]
        predictions_file_object.writerow([row[0], "%d" % bernoulli.rvs(p, size=1)[0] ])
    else:
        p = survival_table[ 1, float(row[1]) - 1, bin_fare]
        predictions_file_object.writerow([row[0], "%d" % 0])

# Close out the files
test_file.close()
predictions_file.close()