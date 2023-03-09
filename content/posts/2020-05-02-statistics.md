---
title: Statistics for Machine Learning
date: 2020-05-02
draft: false
tags:
  - stats
  - ml
categories:
  - ml
math:
  enable: true
---

![](https://user-images.githubusercontent.com/8268939/81514537-90566000-92e4-11ea-977c-5b2884f04a93.png)

# Data types in stats

![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcblx0QVtUeXBlcyBvZiBkYXRhXSAtLT4gQihOdW1lcmljYWwpXG5cdEIgLS0-IERpc2NyZXRlXG4gICAgQiAtLT4gQ29udGludW91c1xuICAgIEEgLS0-IEQoQ2F0ZWdvcmljYWwpXG4gICAgRCAtLT4gT3JkaW5hbFxuICAgIEQgLS0-IE5vbWluYWxcbiAgICBcblxuIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)

### Examples of Numerical

```
# continous
mu = 20
sigma=2
data_continous = numpy.random.normal(mu, sigma, 1000) # generate from 100 to 150 with 0.1 difference
sns.distplot(data_continous, color="blue")
plt.show()

# discrete
import numpy as np
dice_rolls = [np.random.randint(1, 7) for _ in range(10)]
plt.hist(dice_rolls)
plt.show()
```
    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_3_0.png)
    
    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_3_1.png)
    

### Nominal Data

Data you can't order. 
- Gender
- religion
- hair color

```
data = {'Name': ['Jim','Jake','Jessy'],
        'Gender': ['Male','Male','Female']
        }

data = pd.DataFrame(data, columns=['Name', 'Gender'])
data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jim</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jessy</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>

### Ordinal Data

Data you can order, but can't do arthematic.

- customer ratings
- economic status

```
data = {'Movie': ['Superman','Heman','Spiderman'],
        'Rating': [4.0,4.7,4.9]
        }

data = pd.DataFrame(data, columns=['Movie', 'Rating'])
data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Superman</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Heman</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spiderman</td>
      <td>4.9</td>
    </tr>
  </tbody>
</table>
</div>


# Central Tendancy

## Generate data

```
import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randint
from numpy import mean
import seaborn as sns
import matplotlib.pyplot as plt

# lets generate some weights of population

# seed the random number generator
seed(1)
# generate a sample of weights of population
weights = randint(low=120, high=200, size=10000)
```

Lets plot the histogram of weights of population and analyse it.


```
import matplotlib.pyplot as plt
sns.distplot(weights, color="blue")
plt.xlabel("weights")
plt.ylabel("frequency")
plt.show()
```
    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_12_0.png)

## Mean

$\mu=\frac{\sum_{i=1}^{N} x_{i}}{N}$

- one of the several values to descrie central tendancy of the data

```
import numpy
numpy.mean(weights)
```

    159.6552

## Median

**if n is odd**

$Median =\left(\frac{n+1}{2}\right)^{t h} term$

**if n is even**

$\text { Median }=\frac{\left(\frac{n}{2}\right)^{t h} \text {term}+\left(\frac{n}{2}+1\right)^{t h} \text { term }}{2}$


```
numpy.median(weights)
```

    159.0

## Mode

Mode is the most frequently occured value in our distribution. 

```
from scipy import stats
stats.mode(weights)
```
    ModeResult(mode=array([152]), count=array([152]))

## How they change

![FD0B573E-2D62-4100-B9F9-68FC36677A87](https://user-images.githubusercontent.com/8268939/81034613-95329400-8e4c-11ea-9e44-3930cf0e0619.jpeg)

# Measure of spread

## Range 

Range is the difference between min and max value. It shows how much our data is spread.

```
np.max(weights) - np.min(weights)
```

    79

## Quartiles

![](https://user-images.githubusercontent.com/8268939/81514170-6734d000-92e2-11ea-84ae-982368fe0ce4.png)

```
from numpy import percentile

# calculate quartiles
quartiles = percentile(weights, [25, 50, 75])

print('Q1: %.3f' % quartiles[0])
print('Q2 or Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Q4 or Max: %.3f' % np.max(weights))
```

    Q1: 140.000
    Q2 or Median: 159.000
    Q3: 180.000
    Q4 or Max: 199.000


## Variance

Measures how much data is spread from the mean. For technical reasons, we use (n-1) in denominator.

$\text { Variance }=s^{2}=\frac{\sum(x_i-\bar{x})^{2}}{n-1}$

$(x_i -\bar{x})$ is deviation from mean for every value of sample, so variance is mean squared deviation

![](https://user-images.githubusercontent.com/8268939/81039743-b603e500-8e5e-11ea-8205-702d2f818ee4.jpeg)

```
np.var(weights)
```
    537.16871296

## Std Deviation 

Measures the spread from the mean. You can think of it like average distance of data from mean. To negate the squares applied earlier, we do square root here.

$s=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}$

![D3AC658E-085B-4565-95BD-412A2CD262AF](https://user-images.githubusercontent.com/8268939/81039810-d2a01d00-8e5e-11ea-9b2e-a775a602b2ac.jpeg)


For a normal distribution let's see how standard deviation varies from the mean

The below percentages are for most plots which are normally distributed.

![](https://user-images.githubusercontent.com/8268939/81082544-43b7f280-8ea8-11ea-988c-8ade97f1c843.jpeg)


```
np.std(weights)
```

    23.1769004174415

# Covariance & Co-relation

![](https://user-images.githubusercontent.com/8268939/81138760-fb381d80-8f17-11ea-8a85-e35ba3d45d29.jpeg)

## Covariance

covariance measures how two variables are dependent on each other. For a positive covariance 2nd variable increases if 1st increases. For negetive one decreases while other increases. 

$\operatorname{cov}(X, Y)=\frac{\sum_{i=1}^{N}\left(x_{i}-\mu_{x}\right)\left(y_{i}-\mu_{y}\right)}{N}$

## Corelation 

**corelation co-efficient** : value lies always between -1 and 1

$\rho_{x, y}=\frac{\operatorname{cov}(X, Y)}{\sigma_{x}, \sigma_{y}}$

![](https://user-images.githubusercontent.com/8268939/81138794-173bbf00-8f18-11ea-893e-2fcbdbfb90a1.jpeg)


```
# Import pandas library 
import pandas as pd 
  
# initialize list of lists 
data = [[180, 160], [160, 175], [155, 125], [158, 148]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Height', 'Weight']) 
  
# print dataframe. 
df 
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>180</td>
      <td>160</td>
    </tr>
    <tr>
      <th>1</th>
      <td>160</td>
      <td>175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>155</td>
      <td>125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>158</td>
      <td>148</td>
    </tr>
  </tbody>
</table>
</div>




```
numpy.corrcoef(df['Height'], df['Weight'])
```




    array([[1.        , 0.42121072],
           [0.42121072, 1.        ]])



# Random Variable 

Assigns a numerical value to the outcome of random experiment.

![](https://user-images.githubusercontent.com/8268939/81478819-880eff80-91d4-11ea-949a-d990fa5af45b.jpeg)



# Distributions

## Histogram 

Plots frequency of values against values. A histogram can tell following in data 

![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcblx0QVtIaXN0b2dyYW1dIC0tPiBCKENlbnRyYWwgVGVuZGFuY3kpXG5cdEEgLS0-IFNwcmVhZFxuICAgIEEgLS0-IE91dGxpZXJzXG4gICAgQSAtLT4gTW9kZXNcblxuIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)

## Probability Density Functions

- Continuous
- Discrete

![](https://user-images.githubusercontent.com/8268939/81479176-a37b0a00-91d6-11ea-90f2-b8a5dbbde264.jpeg)


## Cumulative Density Function

The max on y axis for CDF will be 1 as all the probabilities will add upto 1.

![](https://user-images.githubusercontent.com/8268939/81479581-efc74980-91d8-11ea-9d3d-0be4301c51f8.jpeg)



# Conditional Probability 

Measure of probability of an event occurring given that another event has occurred. 

**For dependent Events**

$P(\mathrm{A} | \mathrm{B})=\frac{\mathrm{P}(\mathrm{A} \cap \mathrm{B})}{P(\mathrm{B})}$

$\mathrm{P}(\mathrm{A} | \mathrm{B})=$ Probability of $\mathrm{A}$, given $\mathrm{B}$ occurs 

$\mathrm{P}(\mathrm{A} \cap \mathrm{B})=$ Probability of 
$\mathrm{A}$ and $\mathrm{B}$ occurring 

$\mathrm{P}(\mathrm{B})=$ Probability of $\mathrm{B}$

**For Independent Events**

$P(A | B)=P(A) \quad$ (if $A$ and $B$ independent)

Lets see some example of coin tossings. 

![](https://user-images.githubusercontent.com/8268939/81362809-95be6b00-9096-11ea-9b73-2a9c8e96f0b3.jpeg)


Here is the tree diagram for combinations. 

Finally if you sum up, all the combinations will result in sum of 1. 

(1/4) + (1/4) + (1/4) + (1/4) = 1






```
# Lets see all possible combinations of coin tossings
from itertools import product 
tossings = set(product(['H', 'T'], repeat=3))
print("All possible combinations of coin 3 tossings")
tossings
```

    All possible combinations of coin 3 tossings





    {('H', 'H', 'H'),
     ('H', 'H', 'T'),
     ('H', 'T', 'H'),
     ('H', 'T', 'T'),
     ('T', 'H', 'H'),
     ('T', 'H', 'T'),
     ('T', 'T', 'H'),
     ('T', 'T', 'T')}




```
# filter by 1st trail is Head
first_head = {item for item in tossings if item[0] == 'H'}
first_head
```




    {('H', 'H', 'H'), ('H', 'H', 'T'), ('H', 'T', 'H'), ('H', 'T', 'T')}




```
two_head = {item for item in tossings if item.count('H') == 2}
two_head
```




    {('H', 'H', 'T'), ('H', 'T', 'H'), ('T', 'H', 'H')}




```
# p(first_head / two_head) : probability of first one being head given there are 2 heads

def probability(items):
  return len(items) / len(tossings)
```


```
def conditional_probability(A, B):
  return len(A & B) / len(B)
```


```
probability(first_head)
```




    0.5




```
probability(two_head)
```




    0.375




```
conditional_probability(first_head, two_head)
```




    0.6666666666666666



# Central Limit Theorem

> The distribution of mean of all the samples will be normal distribution even if actual population is not normal.



```
import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randint
from numpy import mean
import seaborn as sns
import matplotlib.pyplot as plt

# seed the random number generator
seed(1)
# generate a sample of weights of population
weights = randint(low=120, high=200, size=10000)
print('The average weight is {} pounds'.format(mean(weights)))

weight_df = pd.DataFrame(data={'weight_in_pounds': weights})

weight_df.head()
```

    The average weight is 159.6552 pounds


    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight_in_pounds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132</td>
    </tr>
    <tr>
      <th>2</th>
      <td>192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>195</td>
    </tr>
  </tbody>
</table>
</div>




```
# Lets visualize the population weight frequency graph

sns.distplot(weight_df['weight_in_pounds'], color="blue")
plt.xlabel("random variable of weights")
plt.ylabel("probability of occurence")
plt.title("Distribution of weight of people");
```


    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_52_0.png)
    


Lets collect the mean of all the samples taken from the population


```
no_of_samples_list = [20, 100, 1000] # total samples n
total_mean_list = [] # to store mean of each sample caclulated
mean_of_mean_list = [] 

for n in no_of_samples_list:
  mean_list_given_sample_num = []
  for sample_no in range(n):
    curr_sample = np.random.choice(weight_df['weight_in_pounds'], size = 100) # each sample size k
    mean = np.mean(curr_sample)
    mean_list_given_sample_num.append(mean)
  total_mean_list.append(mean_list_given_sample_num)
  mean_of_mean_list.append(np.mean(mean_list_given_sample_num))
  
# Lets view the distribution and frequency of mean of this samples 

# Make the graph 40 inches by 40 inches
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5), sharex=True)

# plot numbering starts at 1
plot_number=1
for mean_list in total_mean_list:
    ax = sns.distplot(mean_list, color="blue")
    print("plt number {} and mean of mean {}".format(plot_number, np.mean(mean_list)))
    ax.set_title("no of samples {}".format(len(mean_list)))
    # Go to the next plot for the next loop
    plot_number = plot_number + 1
    plt.show()
```

    plt number 1 and mean of mean 158.977



    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_54_1.png)
    


    plt number 2 and mean of mean 159.8567



    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_54_3.png)
    


    plt number 3 and mean of mean 159.63302000000002



    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_54_5.png)
    


> when the number of samples increased, the distribution of mean of samples tends to become normal distribution function.

Lets see the mean of means for different no of samples values.


```
mean_of_mean_list
```




    [160.49999999999997, 159.65560000000002, 159.65964000000002]



Lets visualize all the 3 in single plot to compare.


```
sns.distplot(total_mean_list[0], label="mean of samples for $n={}$".format(no_of_samples_list[0]))
sns.distplot(total_mean_list[1], label="mean of samples for $n={}$".format(no_of_samples_list[1]))
sns.distplot(total_mean_list[2], label="mean of samples for $n={}$".format(no_of_samples_list[2]))
plt.title("Distribution of Sample Means of People's Mass in Pounds", y=1.015, fontsize=20)
plt.xlabel("sample mean mass [pounds]")
plt.ylabel("frequency of occurence")
plt.legend();
```


    
![png](2020_05_02_Statistics_files/2020_05_02_Statistics_58_0.png)
    


# References

https://dfrieds.com/math/central-limit-theorem.html
