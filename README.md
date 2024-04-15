# /dev/log for ft_linear_regression

# Regression
Regression is when we havea dataset which consist of a dependant variable (the outcome) and an independant variable (the things we can change / control) which we are trying to find the relationship between them.

After the relationship is found, we can use that to predict the outcome for a given change. There are multiple regression types

<table>
<tr>
<td>

![image](https://hackmd.io/_uploads/BJC8geFxA.png)


</td>

<td>

Linear regression is regression using a straight line

</td>

</tr>

<tr>
<td>

![image](https://hackmd.io/_uploads/BkYFleFxR.png)

</td>

<td>

logistic regression is regression using a logistic curve

</td>

</tr>

</table>

# Least square method

To generate the line that represents a linear regression, we can use this method. The idea is that we will use our datapoints and determine the minimum value of the residuals using a formula

<table>
<tr>
    <th> change </th>
    <th> outcome </th>
</tr>

<tr>
    <td>1</td>
    <td>2</td>
</tr>
    
<tr>
    <td>3</td>
    <td>4</td>
</tr>
    
<tr>
    <td>3</td>
    <td>5</td>
</tr>
    
<tr>
    <td>4</td>
    <td>4</td>
</tr>
    
<tr>
    <td>5</td>
    <td>5</td>
</tr>
</table>

> change mean = 3, outcome mean = 4

So we start with a sample dataset; we get the mean of both the x and y values (independant and dependant variables) which is (3, 4) and it turns out the regression line has to go through (3,4)

<table>
<tr>
    <td>
        TODO insert graph pic here
    </td>
    <td>
        * notice the linear regression is a line and can be
         y = mx + c
    </td>
</tr>
</table>

And then we will take the difference between all x values and the means x; and same goes for y

<table>
<tr>
    <th> x </th>
    <th> y </th>
    <th> x - mean_x </th>
    <th> y - mean_y </th>
    <th> (x - mean_x) ^ 2 </th>
    <th> (x - mean_x)(y - mean_y) </th>
</tr>

<tr>
    <td> 1 </td>
    <td> 2 </td>
    <td> -2 </td>
    <td> -2 </td>
    <td> 4 </td>
    <td> 4 </td>
</tr>
    
<tr>
    <td> 2 </td>
    <td> 4 </td>
    <td> -1 </td>
    <td> 0 </td>
    <td> 1 </td>
    <td> 0 </td>
</tr>
    
<tr>
    <td> 3 </td>
    <td> 5 </td>
    <td> 0 </td>
    <td> 1 </td>
    <td> 0 </td>
    <td> 0 </td>
</tr>
    
<tr>
    <td> 4 </td>
    <td> 4 </td>
    <td> 1 </td>
    <td> 0 </td>
    <td> 1 </td>
    <td> 0 </td>
</tr>
    
<tr>
    <td> 5 </td>
    <td> 5 </td>
    <td> 2 </td>
    <td> 1 </td>
    <td> 4 </td>
    <td> 2 </td>
</tr>
    
</table>

We also need to calculate `(x - mean_x)^2` and `(x - mean_x)(y - mean_y)` because the are used in the following formula for least squares, this is derived using the minimum value of the residuals function.

$$
b_1=\frac{\sum(x-\bar{x})(y-\bar{y})}{\sum(x-\bar{x})^2}
$$


This would give us the gradient of the r\regression line, we can simple calculate the intercept of the regression line by subsituting the gradient in the `y = mx + c` equation using (3, 4) as our `x` and `y` respectively

# R2 - Correleation
R2 is used to represent the accuracy of the relationship. A completely accurate mapping will have an R2 of 1 (mean / mean), while relationships which are less accurate will apporach 0. It takes the sum of the predicted outcomes - mean outcomes over the sum of actual outcomes - mean outcomes squared.

$$
R_2=\frac{\sum(\hat{y}-\bar{y})^2}{\sum(y-\bar{y})^2}
$$

Basically saying how far is the mean of the predicted values is from the mean of the actual values? While R2 is comparing the means of predicted and mean of actual values, the standard error or residue is the comparison between the predicted values and the actual values themselves. 

# Least square derivation
Residuals are differences between the predicted values and the actual values themselves and not the mean (like R2). The least square functions objective is to miuimize this as much as possible. Hence, it is derived from finding the minimum value for the sum of all residuals squared

We will start off with the sum of all residuals squared.

<table>   
<tr>
<td>  

$\sum \left( y-\widehat{y}\right) ^{2}$

</td>

<td>
    why squared? because to remove negatives and it has properties that makes it a good estimator
</td>
</tr>
</table>

The $\widehat{y}$ above can be subsituted, giving us 

$$
\sum \left[ y-\left( b_{1}x+b_{0}\right) \right] ^{2}
$$

We are trying to find values of $ b_{1}$ and $b_{0}$ so that the sum of squared residuals is minimum, we can use partial derivative.

$$
\begin{aligned}\dfrac{d}{db_{0}}\sum \left[ y-\left( b_{1}x+b_{0}\right) \right] ^{2}=0\\ \Rightarrow \sum \dfrac{d}{db_{0}}\left( y-b_{1}x-b_{0}\right) ^{2}=0\\ \Rightarrow \sum 2\left( Y-b_{1}x-b_{0}\right) \left( -1\right) =0\end{aligned}
$$

Note that the last line there is like so because of the chain rule, we need to also calculate the derivative inside the parenthesis since the derivative here isnt $\dfrac{d}{d\left( y-b_{1}x-b_{0}\right) }$, its $\dfrac{d}{db_{0}}$ instead. The first equation can be simplified to 

$$
-2\sum \left( y-b,x-b_{0}\right) =0
$$

The second equation can be derived like so 

$$
\begin{aligned}\dfrac{d}{db_{1}}\sum \left( Y-b_{1}x-b_{0}\right) ^{2}=0\\ \Rightarrow \sum 2\left( Y-b_{1}x-b_{0}\right) \left( -x\right) =0\\ \Rightarrow -2\sum x\left( Y-b_{1}x-b_{0}\right) =0\end{aligned}
$$

We can solve the first equation by simplifying the equation in terms if $b_{0}$

$$
\begin{aligned}\sum \left( y-b_{1}x-b_{0}\right) =0\\ \Rightarrow \sum Y-\sum b,x-\sum b_{0}=0\\ \Rightarrow \sum Y-b_{1}\sum x-nb_{0}=0\\ \Rightarrow nb_{0}=\sum y-b_{1}\sum x\\ \Rightarrow b_{0}=\overline{y}-b_{1}\overline{x}\end{aligned}
$$

The third and fourth line is done by using the constant factor rule for summation:

$$
\sum^{n}_{i=0}x=x_{1}+x_{2}+x_{3}\ldots x_{n}
$$


If $x$ is a constant, this can be represented by $n(x)$. the $\overline{x}$ and $\overline{y}$ are means.

We can use the $b_{0}$ above and solve for the second equation like so 

$$
\begin{aligned}\sum x\left( Y-b_{1}x-b_{0}\right) =0\\ \Rightarrow \sum x\left[ Y-b_{1}x-\left( \overline{Y}-b_{1}\overline{x}\right) \right] =0\\ \Rightarrow \sum x\left[ Y-\overline{Y}-b_{1}\left( x-\overline{x}\right) \right] =0\\ \Rightarrow \sum x\left( Y-\overline{Y}\right) -\sum b_{1}x\left( x-\overline{x}\right) =0\\ \Rightarrow b_{1}=\dfrac{\sum x\left( Y-\overline{Y}\right) }{\sum x\left( x-\overline{x}\right) }\end{aligned}
$$

Which is equivilant to 

$$
b_1=\frac{\sum(x-\bar{x})(Y-\bar{Y})}{\sum(x-\bar{x})^2}
$$

# Gradient Descent

Gradient Descent is also another way to find the minimum for squared residuals using approximation. We can start with a simple example to demonstrate how it works:

<table>
<tr>
<td>

![image](https://hackmd.io/_uploads/r1RR4xKxC.png)


</td>

<td>

The regression line should follow the formula `y = b1x + b0` and lets say `b0` is already solved with value 0.64 and we need to solve for `b1`

</td>

</tr>
</table>

We can slowly guess `b1` until it reaches its minumum, according to this graph 

![image](https://hackmd.io/_uploads/BkjuBgtgA.png)


However, doing this is slow if we have a large dataset, as we need to make many adjustments. Gradient descent helps us to do less calculations by incresing the length of steps at the beginning and slowing down when it approaches the minumum

![image](https://hackmd.io/_uploads/H1OjBxteR.png)


The first thing that we need is the loss function (a function we want to minimize the value of). In this case, the loss function will be the sum of squared residuals. We then take the following steps

1. Take the derivative for the loss function and pick a random value for the output (intercept)
2. Get the result of the derivative - this result will be used to calculate the output (intercept) for the next iteration
3. Calculate the step size using the `resullt of derivative * learning rate (another arbitrary value)`
4. Get the new output (intercept) by having the `old output - step size`
5. Rinse and repeat, until step size is close to 0 or we have iterated and arbitrary number of steps

We can also use multiple parameters with multiple loss functions, say we are trying to minimize both `b1` and `b0`, we can do

$$
\begin{aligned}
& \frac{d}{d b_0}=\sum(0+1 x)=\alpha \\
& \frac{d}{d b_1}=\sum(0+1 x)=\beta
\end{aligned}
$$

The 0 and 1 coefficients are random initial values for the first iteration. It will be change to the following after the iteration ends

$$
\begin{aligned}
\frac{d}{d b_0}=\sum\left(\alpha+\beta_\lambda\right) \\
\frac{d}{d b_1}=\sum\left(\alpha+\beta_2\right)
\end{aligned}
$$


# Z-score and distribution
The Z-score is a metric that measures how far a data point is away from the mean, and the standard deviation is a unit of measurement for the z-score. E.g. the Z-score of value 69 is 0.2 standard deviations away from the mean.

As such, the magnitude of 1 standard deviation needs to be derived from the dataset itself, so that the z-score calculations remain relative. Naturally, a dataset which has more spread out will have a bigger standard deviation becase when we want to determine the z-score, we will have to take bigger steps to reach the relative distance and vice-versa.

<table>
<tr>
<td>
    
$\sigma=\sqrt{\frac{\sum(x-\mu)^2}{N}}$

</td>
<td>
Standard deviation
</td>
</tr>
    
<tr>
<td>
    
$Z=\frac{x-\mu}{\sigma}$

</td>
<td>
    Z score
</td>
</tr>
</table>


Z score visualization:
<table>
<tr>
<td>
61.3 
</td>

<td>
69.2
</td>

<td>
77.1
</td>

<td>
85
</td>

<td>
92.9
</td>

<td>
100.8
</td>

<td>
108.7
</td>
</tr>


<tr>
<td>
x - 3(sd)
</td>

<td>
x - 2(sd)
</td>

<td>
x - sd
</td>

<td>
x
</td>

<td>
x + sd
</td>

<td>
x - 2(sd)
</td>

<td>
x + 3(sd)
</td>
</tr>




<tr>
<td>
z = -3
</td>

<td>
z = -2
</td>

<td>
z = -1
</td>

<td>
z = 0
</td>

<td>
z = 1
</td>

<td>
z = 2
</td>

<td>
z = 3
</td>
</tr>


</table>

# Normalization and standardization (feature scaling)
The problem: When it comes to approaching values by step size like in gradient descent, there are cases where one feature (axis) might overpower or have greater change than the other. This causes more uneccasary updates for the not-so-changey feature and it will make it oscillate or even worse, diverge before the optimum minima is reached for both of these features.

Feature scaling mitigates this by manipulating the data so that they fit certain constraints which will resilt is a faster learning / approximation process.

**Normalization** aka min-max scaling is a feature scaling method which scales the data so that the minimum of the data is 0 and the maxumim is 1. This decreases the magnitude of the changes to ease the approximation process. This feature scaling retains the relative ratio between data points but it is sensitive to outliers.  E.g.

```
1, 2, 3, 10 -> 0, 0.11, 0.22, 1
1, 2, 3, 100 -> 0, 0.0101, 0.0202, 1
1, 2, 3, 1000 -> 0, 0.00011, 0.00022, 1
```

As we can see, the more drastic the outliers, the more close the values 1, 2, and 3 get scaled together. This will cause them to be assumed as equal in come cases and will cause data loss.

$$
\hat{x}=\frac{x-x_{\text {min }}}{x_{\text {max }}-x_{\text {min }}}
$$
> min max formula

**Standardization** is a feature scaling which shifts the data so that the mean is 0 and the standard deviation is 1. This does not retain the relative scale of the data.

```
1, 3, 4, 5, 6 -> -1.6, -0.42, 0.11, 0.7, 1.28 *the ratio of difference between 1 and 3 vs 5 and 6 is different with -1.6 and -0.42 vs 1.28 and 0.7

1, 2, 3, 4, 5 -> -1.4, -0.7, 0, 0.7, 1.4
```
$$
x=\frac{x-\bar{x}}{\sigma}
$$
> standardization formula

Althought z-score scaling might look like eliminating the relative scales between data, it is not since :
1. It is reversible to obtain the coefficients for the actual data
2. It is not manipulating the scale of the actual data; we are just plotting the z-score of each data point and if we run regression on that, we are actually getting the relationship of the predicted data's difference with the mean and not the predicted datas difference between the actual data which SHOULD be the same scales **even for multiple features** because they have been scaled by their own respective standard deviations

# Installation
Activate env
`souce bin/env/activate`

Install deps
`pip install -r requirements.txt`

Run training program 
`python train.py`

Run Prediction program
`python predict.py`

![image](https://hackmd.io/_uploads/HJrkw3clA.png)
