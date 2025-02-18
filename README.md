# Deep mixture of linear mixed models for complex longitudinal data
## Abstract
Mixtures of linear mixed models are widely used for modelling longitudinal data for which observation times differ between subjects. In typical applications, temporal trends are described using a basis expansion, with basis coefficients treated as random effects varying by subject. Additional random effects can describe variation between mixture components, or other known sources of variation in complex designs. A key advantage of these models is that they provide a natural mechanism for clustering. Current versions of mixtures of linear mixed models are not specifically designed for the case where there are many observations per subject and  complex temporal trends, which require a large number of basis functions to capture. In this case, the subject-specific basis coefficients are a high-dimensional random effects vector, for which the covariance matrix is hard to specify and estimate, especially if it varies between mixture components. To address this issue, we consider the use of deep mixture of factor analyzers models as a prior for the random effects. The resulting deep mixture of linear mixed models is well-suited for high-dimensional settings, and we describe an efficient variational inference approach to posterior computation. The efficacy of the method is demonstrated in biomedical applications and on simulated data.

## The DMLMM Code
This repository contains code for the DMLMM as introduced in the paper "Deep mixture of linear mixed models for complex longitudinal data" by Lucas Kock, Nadja Klein and David J. Nott. 

minimal_working_example.py gives a minimal working example and should be the starting point to familiarize yourself with the code.

DMLMM/dmlmm.py is the code to generate and train the DMLMM. It takes a list of observations y, a list of design matrices x, as well as the number of knots per layer K and the dimensions in each layer D as inputs. The function train() fits the model to the data. Afterwards predictions for unobserved time points can be made with predict_y(...). 

The folder Simulations contains code to reproduce the simulation study presented in the paper. 


