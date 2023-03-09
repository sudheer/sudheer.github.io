---
title: Logistic Regression
date: 2020-04-28
draft: false
tags:
  - ml
categories:
  - ml
math:
  enable: true
---

# Introduction

Logistic regression uses the logistic sigmoid function to return a probability value from feature variables.

How logistic regression works ?

![](https://kroki.io/blockdiag/svg/eNpNjUEKAjEMRfdzipC9Jxh0Iy4F9-Ki08YSGhtpK0MR727VGenuh_fy_yRqg2Pj4TkAHNWRwGYHh1z4Zgp9Mp4SObYF29GcvZic-VrH4f9wtiqaYAvoE1GsJKIzXsYm4NqEnXXnGBa8tvVYk4mefgIs49DxHOokj6_wegN_eTvD)

# Examples

- A person is obese or not ?
- Does Mr A has cancer ?
- Will this team win the match today ?
- Email is spam or not ?

![5A828DBD-A9A7-41E3-B3A6-D2CFFFDE98DC](https://user-images.githubusercontent.com/8268939/80929921-b9b34100-8d64-11ea-9f91-166187ab360c.jpeg)


# Why not linear regression

- Linear regression predicts output as continuous range from $-\infty$ to $+\infty$. But we are predicting discrete values like 0 and 1 in case of logistic regression.
- Moreover we can't map all the output values onto a straight line as in case of linear function. There is huge chance that we miss predictions as shown in figure below

![A1B249F9-F2A6-40F0-80C5-8AD36E738439](https://user-images.githubusercontent.com/8268939/80929411-a30aeb00-8d60-11ea-917b-29bef2fa6c07.jpeg)

In logistic regression, the output of linear regression is passed to a sigmoid funtion to convert the predicted continuous to discrete categorical values.

# Linear Regression 

![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgeDEoKHhfMSkpIC0tPiB8dzF8IEIoKFN1bSkpXG4gIHgyKCh4MikpIC0tPiB8dzJ8IEJcbiAgeDMoKHgzKSkgLS0-IHx3M3wgQlxuICBCIC0tPiBDW3gxdzEgKyB4MncyICsgeDN3MyArIC4uLl1cblxuICBcbiAgXG4gIFxuICBcbiAgXG5cdFx0IiwibWVybWFpZCI6eyJ0aGVtZSI6Im5ldXRyYWwifX0)


# Logistic Regression 

![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgeDEoKHhfMSkpIC0tPiB8dzF8IEIoKFN1bSkpXG4gIHgyKCh4MikpIC0tPiB8dzJ8IEJcbiAgeDMoKHgzKSkgLS0-IHx3M3wgQlxuICBCIC0tPiBDW3gxdzEgKyB4MncyICsgeDN3MyArIC4uLl1cbiAgQyAtLT4gRCgoU2lnbW9pZCkpXG5cbiAgXG4gIFxuICBcbiAgXG4gIFxuXHRcdCIsIm1lcm1haWQiOnsidGhlbWUiOiJuZXV0cmFsIn0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9)


Let's see the differences between Linear and Logistic 

$\begin{array}{l|l|l}
\hline & \text { Linear } & \text { Logistic } \\
\hline \text { Target Variables } & \text { Continuous } & \text { Categorical } \\
\hline \text { Problem Type } & \text { Regression } & \text { Classification } \\
\hline \text { Hypothesis } & \theta^{T} x & sigmoid\left(\theta^{T} x\right) \\
\hline \text { Loss } & \text { Mean Squared } & \text { Logistic } \\
\end{array}$

# Types

- **Binary:** Output dependent variabels mapped to 2 categorical values
- **Multinomial:** Three or more categorical values for classification
- **Ordinal:** Three or more categorical values with ordering

# Math intro

![3F6FE2A8-A53B-476F-9E9C-CBB4B6217102](https://user-images.githubusercontent.com/8268939/80930525-ee28fc00-8d68-11ea-9195-2420b7fb3813.jpeg)


### Odds and Log Odds

Since the goal of logistic function is to map linear combination of input variabels into a probability, we need a link to map linear combination to probability, and that link is logit function. Before knowing about logit functions, let's see what odds, log odds and odds ratio mean.

#### Odds

$\begin{aligned}
\operatorname{odds}(Y=1) &=\frac{P(Y=1)}{P(Y=0)}=\frac{P(Y=1)}{1-P(Y=1)} \\
&=\frac{p}{1-p} = \frac{Probability of event happening}{Probability of event not happening}
\end{aligned}$

Lets check the odds for a sample data


```
import pandas as pd

data = [['CS', 'Dropout'], ['EE', 'Graduated'], ['CS', 'Dropout'], ['CS', 'Graduated'], ['EE', 'Dropout'], ['CS', 'Dropout'], ['CS', 'Dropout'],['EE','Graduated']] 
df = pd.DataFrame(data, columns = ['Branch', 'Status'])

pd.crosstab(index=df['Branch'], columns= df['Status'], margins=True)
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
      <th>Status</th>
      <th>Dropout</th>
      <th>Graduated</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Branch</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CS</th>
      <td>4</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>EE</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>All</th>
      <td>5</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```
# odds of cs graduated
odds_cs_grad = (1/5)/(4/5)  # p/(1-p)
print("odds of cs graduated {}".format(odds_cs))
```

    odds of cs graduated 0.25



```
# odds of EE graduated
odds_ee = (2/3)/(1/3) # p/(1-p)
print("odds of ee graduated {}".format(odds_ee))
```

    odds of ee graduated 2.0



```
# Odds ratio 

odds_ratio = odds_ee/odds_cs
print("odds ratio of ee to cs is {}".format(odds_ratio))

print("A EE student is {} times likely to graduate than CS".format(odds_ratio))
```

    odds ratio of ee to cs is 8.0
    A EE student is 8.0 times likely to graduate than CS


### Lets plot log and log odds functions


```
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def odds(p):
    return p / (1 - p)

def log_odds(p):
    return np.log(p / (1 - p))

x = np.arange(0.01, 1, 0.05)
odds_x = odds(x)

log_odds_x = log_odds(x)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
plt.axvline(0)
plt.axhline(0)
axes[0].plot(x, odds_x)
axes[0].set_title("odds function")
axes[0].set(xlabel="x", ylabel="odds")
axes[1].plot(x, log_odds_x)
axes[1].set_title("log odds function")
axes[1].set(xlabel="x", ylabel="log_odds")
fig.tight_layout()
```


    
![png](assets/images/28_04_2020_logistic_regression_files/28_04_2020_logistic_regression_13_0.png)
    


### Logit function

$\operatorname{logit}(p)=\log \left(\frac{P}{1-P}\right), \text { for } 0 \leq p \leq 1$

This logit function is what we are trying to equate it to our linear combination of input variables. 

$\log(\frac{P}{1-P}) = \theta_1  x_i + \theta_0$

$P = \frac{1}{1+e^(\theta_1 x_i + \theta_0)}$ This exactly looks like sigmoid function which we will study below.

$P$ = probability of success

$-\infty \leq x_i \leq \infty$; 


## Sigmoid function

Sigmoid function is used in the logistic regression to map infinite values into a finite discrete target values.

Equation of sigmoid function is $g(z)=\frac{1}{1+e^{-z}}$

The function is plotted below 

$\begin{aligned}
&\lim _{x \rightarrow \infty} g(z)=1\\
&\lim _{x \rightarrow-\infty} g(z)=0
\end{aligned}$

Interesting thing about sigmoid function is, even the derivative of it can be expressed as the function itself. The first order derivate of sigmoid function is $\frac{d g(z)}{d z}=g(z)[1-g(z)]$


```
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline

z = np.linspace (-10,10,100)

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

plt.figure(figsize=(10,6))
plt.plot(z,sigmoid(z))
plt.xlim([-10,10])
plt.ylim([-0.1,1.1])
plt.axvline(0)
plt.axhline(0)
plt.xlabel('z');
plt.ylabel('g(z)')
plt.title('Sigmoid function');
plt. show ( )
```


    
![png](assets/images/28_04_2020_logistic_regression_files/28_04_2020_logistic_regression_16_0.png)
    


### Bernoulli Distribution 

We need to get some basics of Bernoulli Distribution here. Bernoulli says 

$f_{\text {Bernoulli}}=\left\{\begin{array}{ll}
1-P ; & \text { for } n=0 \\
P ; \quad \text   { for } n=1 
\end{array}\right.$

where $n = 0$ is failure event and $n = 1$ is a successful event.

# Hypothesis

This equation takes the featurs (x) and parameters ($\theta$) as input and predicts the output dependent variable.

The weighted combination of input variables is ...
$\theta_{1} \cdot x_{1}+\theta_{2} \cdot x_{2}+\ldots+\theta_{n} \cdot x_{n}$

Writing the above function in linear algebra from ...

$\sum_{i=1}^{m} \theta_{i} x_{i}=\theta^{T} x$

Lets write this in matrix form 

$\left[\begin{array}{c}
\theta_{1} \\
\theta_{2} \\
\cdot \\
\cdot \\
\theta_{n}
\end{array}\right]^{T} \cdot\left[\begin{array}{c}
x_{1} \\
x_{2} \\
\cdot \\
\cdot \\
x_{n}
\end{array}\right]=\left[\begin{array}{cccc}
\theta_{1} & \theta_{2} \ldots \theta_{n}
\end{array}\right] \cdot\left[\begin{array}{c}
x_{1} \\
x_{2} \\
\cdot \\
\cdot \\
\cdot \\
x_{n}
\end{array}\right]=\theta_{1} x_{1}+\theta_{2} x_{2}+\ldots+\theta_{n} x_{n}$

If we pass this equation to sigmoid function ....

$P\left(\theta^{T} x\right)=g\left(\theta^{T} x\right) = \frac{1}{1+e^{-\theta^{T} x}}$

where $P\left(\theta^{T} x\right) = h_{\theta}(x)$ and $g()$ is called sigmoid function.

Now the hypothesis can be written as $h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}$

where $h_{\Theta}(x)=P(Y=1 | X ; \theta )$

In words Probability that $Y=1$ for features $X$ with co-efficients $\theta$

# Cost Function

we can't use the cost function Sum of Squared errors (SSE) in logistic regression as it would give convex graph and we will get lot of local minima and makes it very difficult to reach to a point of global minima.

![017B613F-8ACD-4337-997D-4599CA4F7122](https://user-images.githubusercontent.com/8268939/80727106-2820a680-8aba-11ea-9188-6ef7a650e79d.jpeg)

In linear regression, we have used Sum of squared errors (SSE) for calculating cost. In logistic regression we use slightly different approach. Suppose if a function predicts sucess % of 90 and seem to be a failure, we penalize it heavily than 30% probability prediction.

So for logistic regression, we go for a logarithemic cost function as below. The log cost function penalizes confident and wrong predictions heavily

$\operatorname{cost}\left(h_{\theta}(x), y\right)=\left\{\begin{array}{ll}
-\log \left(h_{\theta}(x)\right) & \text { if } y=1 \\
-\log \left(1-h_{\theta}(x)\right) & \text { if } y=0
\end{array}\right.$

if we convert the above to one liner ...

$\operatorname{cost}\left(h_{\theta}(x), y\right)=-y \log \left(h_{\theta}(x)\right)-(1-y) \log \left(1-h_{\theta}(x)\right)$

Finally the cost function for all the values will be

$\begin{aligned}
J(\theta) &=\frac{1}{m} \sum_{i=1}^{m} \operatorname{cost}\left(h_{\theta}\left(x^{(i)}\right), y^{(i)}\right) \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
\end{aligned}$

### Minimize cost function

![](https://kroki.io/blockdiag/svg/eNp1jkEKwkAMRfc9RZi9Jyi66coDuBIXcSaOoUNS0mip4t2lVCiUun083v_XorFNjBneFUA4Cjtj4RfBQJzv3gfYHSB0Romjg6NlcniizbzBAo32DreHRGeVGZ-6hL5KbKTratrcaJyjFjXYQ8hGJCOVokO41P8uLn7H0v7E9YlFUkPJNGmfL0pNU8w=)

use gradient descent to minimize the cost function 

$\frac{\partial J(\theta)}{\partial \theta_{j}}=\frac{1}{m} \sum_{i=1}^{m}\left(h\left(x^{i}\right)-y^{i}\right) x_{j}^{i}$


```
def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression. 
    
    Parameters
    -------------------------------------
    theta : weight vector of shape (n+1, )
    X : input of shape (m x n+1) m: no of training ex, n:no of features
    y : predicted y (m, ).
    
    Returns
    -------------------------------------
    cost : value of cost function
    grad : vector of shape (n+1, ) -> gradient of the cost fun wrt weights
    """
    m = X.shape[0]  # number of training examples

    # initialize Returns
    cost = 0
    grads = np.zeros(theta.shape)
    
    #Prediction
    sigmoid_result = sigmoid(x.dot(theta))
    Y_T = y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(sigmoid_result)) + ((1-Y_T)*(np.log(1-sigmoid_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (sigmoid_result-Y.T).T))
    db = (1/m)*(np.sum(sigmoid_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return cost, grads
```
