# Car Price Prediction Multiple Linear Regression
This notebook will introduce some foundation machine learning and data science concepts by exploring the problem of Car price prediction.<br />
It is intended to be an end-to-end example of what a data science and machine learning proof of concept might look like.<br />
## 1. Problem Definition
A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.<br />
They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market.<br />
**The company wants to know:**<br />
Which variables are significant in predicting the price of a car. <br />
How well those variables describe the price of a car. <br />
Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market. <br />
### Brief about Linear Regression: <br />
**Linear regression** is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope. It's used to predict values within a continuous range, (e.g. sales, price).<br />
The simplest form of the regression equation with one dependent and one independent variable is defined by the formula y = c + b*x, where y = estimated dependent variable score, c = constant, b = regression coefficient, and x = score on the independent variable.<br />
There are two main types:<br />
* Simple regression <br />
Simple linear regression uses traditional slope-intercept form, where m and b are the variables our algorithm will try to “learn” to produce the most accurate predictions. x represents our input data and y represents our prediction.<br />
y=mx+b <br />
* Multivariable regression <br />
A more complex, multi-variable linear equation might look like this, where w represents the coefficients, or weights, our model will try to learn.<br />
f(x,y,z)=w1x+w2y+w3z <br />
The variables x,y,z represent the attributes, or distinct pieces of information, we have about each observation. For sales predictions, these attributes might include a company’s advertising spend on radio, TV, and newspapers. <br />
Sales=w1Radio+w2TV+w3News <br />
**Ideal Rregression model graphical representation for reference. Which is our target to build a model like this** <br />
![Ideal Rregression model graphical representation for reference](https://github.com/sayanbiswasgithub/car-price-prediction-multiple-linear-regression/blob/main/LRPic.PNG)
