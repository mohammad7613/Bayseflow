
## General description of the pipeline


Each evaluation pipeline needs three modules:

Generating validation data 

Sampling posterior given the valiation data

Calculating metrics of evaluation given the validation data and the posterior samples

Reporting the evaluation results. 


Therefore, we should have three modules (.py files):

Create Evaluation Batch-> It corresponds to Generating Validation Data -> It should be generalized as it can vary among different 
validations


Metric Calculation -> It should be about quality checking and it can vary among different evaluations.


Reporting  -> It should vary among saving array to saving plots and visualizing


The sampling may not need a separate module as it can not be implemented in various ways. It can be generated in approximate.sample at this stage.



In the following, I have filled each section with functions and data structures we have in each module.


## Reporting

Here, we have to list the basic modular functions:

plot recovery

save the figure 

save the metric data of evaluation in a data structure 

save the metadata of the evaluation pipeline including posterior samples and validata size and conditions, evaluation name, model specs 



## Metric Calculation


The metric calculations

posterior mean and std

posterior coverage and confidence interval

Absolute Mean Error and Mean Squared Error 





## Evaluation Generation 

I should define a basic structure of Evaluation data generated and the pipeline of generation. 


