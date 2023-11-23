# Covid-19-Cases-Prediction

## Project Description

In 2020, the emergence of an unknown pneumonia, later identified as COVID-19, led to a global pandemic affecting over 200 countries. Governments implemented various measures like travel restrictions, quarantines, and lockdowns to curb the virus's spread. However, a perceived lack of effective efforts and a need for automated tracking and prediction systems prompted scientists to propose the use of deep learning models. Specifically, they suggested employing an LSTM neural network to predict daily COVID-19 cases in Malaysia based on the past 30 days' data. The goal was to inform decisions on whether to impose or rescind travel bans, emphasizing the potential of AI in mitigating the impact of the pandemic.

## Problem Statement:
-  The 2020 global COVID-19 pandemic revealed weaknesses in our ability to respond quickly to health crises.
-  Government measures were hindered by delays and a lack of automated systems.
-  The specific issue is the absence of a reliable method to predict daily COVID-19 cases in Malaysia, impacting interventions like travel bans.
-  To tackle this, there's a critical need for an advanced deep learning model, specifically an LSTM neural network. This model aims to forecast new cases based on the past 30 days' data, offering actionable insights for policymakers to make informed decisions and manage the virus spread effectively.

<p align="center">
<img src="https://github.com/SalmanSuhaimi/Covid-19-Cases-Prediction/assets/148429853/39d09187-f9f2-4426-ad7b-f7ad78c2b372" width="500" height="500"/>
</p>

## Objective:
-  The objective is to create and implement an advanced deep learning model, particularly utilizing an LSTM neural network, with a primary focus on accurately predicting daily COVID-19 cases in Malaysia based on the past 30 days' data.
-  This model aims to overcome the vulnerabilities exposed during the 2020 global pandemic, providing policymakers with timely and proactive insights for informed decisions regarding the imposition or rescinding of travel restrictions.
-  By automating tracking and prediction processes, the goal is to enable swift responses to emerging health crises, safeguard public health, and contribute to the global effort in managing and controlling the spread of COVID-19.
-  Continuous evaluation and refinement ensure the model's adaptability to evolving trends and emerging data, reinforcing its effectiveness over time.

### Why do We Use Long Short-Term Memory (LSTM)?
-  LSTM neural networks are preferred for tasks like predicting daily COVID-19 cases due to their specialized architecture for handling sequential data and time series.
-  LSTMs excel in capturing long-term dependencies, avoiding issues like the vanishing gradient problem.
-  They are effective in handling time lags, automatically learning relevant features, and offering flexibility for various sequential data applications.
-  In the context of predicting COVID-19 cases, LSTMs prove valuable for modeling the temporal dependencies and intricate patterns inherent in the progression of the pandemic over time, contributing to more accurate predictions.

### Flow LSTM
1. Setup:
  -  Import necessary libraries and modules, including TensorFlow and visualization tools.
  -  Set up configurations for plotting.

2. Load Dataset:
  -  Load a dataset ('cases_malaysia_covid.csv') containing COVID-19 cases in Malaysia with columns like 'cases_new', 'cases_import', 'cases_recovered', and 'cases_active'.
  -  Convert the 'date' column to a datetime format.

3. Data Inspection:
  -  Check the data information, including column data types.
  -  Convert 'cases_new' column from object to integer type, handling potential formatting issues.
  -  Inspect basic statistics and visualize the time series data.

4. Data Cleaning:
  -  Identify and handle null values in the 'cases_new' column.
  -  Visualize data distribution using box plots and histograms.
  -  Fill null values with the median and drop duplicate rows.

5. Train-Validation-Test Split:
  -  Split the dataset into training, validation, and test sets for time series data.

6. Data Normalization:
  -  Normalize the data using mean and standard deviation from the training set.

7. Data Inspection After Normalization:
  -  Visualize the distribution of normalized features using violin plots.
    
![violin](https://github.com/SalmanSuhaimi/Covid-19-Cases-Prediction/assets/148429853/0ac5977b-e411-4ad0-a37a-ae53e0c00456)

8. WindowGenerator:
  - Initialization:
      -  The class is set up with input parameters and dataframes for training, validation, and testing.
      -  It keeps track of column indices for labels and inputs.

  - Indices and Slicing:
      -  It defines slices and indices for input and label windows based on specified widths and shifts.

  - 'split_window' Method:
      -  Splits features into inputs and labels, optionally stacking labels for multiple columns.

  - '__repr__' Method:
      -  Returns a string with window details.

  - 'plot' Method:
      -  Plots input features, labels, and model predictions for a specified column.

  - 'make_dataset' Method:
      -  Converts input data into a TensorFlow dataset suitable for training models.

  - 'Properties' (train, val, test, example):
      -  Provide easy access to training, validation, and test datasets.
      -  The example property returns a batch of inputs and labels for plotting.

9. Model Training - Single-Step Prediction:
  -  Create an LSTM model for single-step prediction (predicting the next day's cases).
  -  Train the model using the defined WindowGenerator.

10. Model Evaluation - Single-Step Prediction:
  -  Evaluate the trained model on the validation and test sets.
  -  Plot the predicted values against the actual values.
    
![plot single_step](https://github.com/SalmanSuhaimi/Covid-19-Cases-Prediction/assets/148429853/488cb6e5-f928-450c-856f-bb9f48b19753)

Graph shows plot the predicted values against the actual values for Single-Step Prediction

11.  Model Training - Multi-Step Prediction:
  -  Define a multi-step prediction scenario (predicting the next 25 days' cases).
  -  Create and train a new LSTM model for multi-step prediction.

12.  Model Evaluation - Multi-Step Prediction:
  -  Evaluate the multi-step prediction model on the validation and test sets.
  -  Plot the predicted values against the actual values.

![plot multi_step](https://github.com/SalmanSuhaimi/Covid-19-Cases-Prediction/assets/148429853/5e028fd5-9f7c-413f-a18e-a8e9920b35e9)

Graph shows plot the predicted values against the actual value for multi step prediction

13.  Display Model Summary and Structure:
  -  Display the summary and structure of both the single-step and multi-step prediction models.

## Conclusion
In conclusion, the undertaken objective of developing and implementing an advanced deep learning model, specifically utilizing an LSTM neural network, to predict daily COVID-19 cases in Malaysia is a crucial step towards enhancing our capacity to respond effectively to health crises. By addressing the shortcomings exposed during the 2020 global pandemic, this model serves as a proactive tool for policymakers, offering timely insights into the necessity of travel restrictions. The focus on automation and continuous evaluation reflects a commitment to adaptability and resilience in the face of evolving trends. Ultimately, the implementation of this advanced predictive model contributes to global efforts in managing and controlling the spread of COVID-19, prioritizing public health and fostering a more responsive and informed approach to crisis management.

## Data source:
https://github.com/MoH-Malaysia/covid19-public
