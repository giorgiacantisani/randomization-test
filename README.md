# randomization-test
Code for running an adaptation of the computationally-intensive randomization test [1], 
a non-parametric hypothesis test. This code was used in the paper:

Cantisani et al. "EEG-based decoding of auditory attention to a target instrument in 
polyphonic music." 2019 IEEE Workshop on Applications of Signal Processing to Audio 
and Acoustics (WASPAA).

Considering a random classifier, the function computes its performances n_iter times, 
leading to an empirical distribution of the performances. This empirical distribution 
is then approximated with a theoretical distribution which could be a normal or a 
t-distribution (the one that fits better). At this point, the function evaluates how 
likely the input performances (given by y_pred and y_true) were to be produced by 
this artificial distribution of performances obtaining the P-value.

[1] E. W. Noreen, "Computer-intensive methods for testing hypotheses". Wiley New York, 1989.

