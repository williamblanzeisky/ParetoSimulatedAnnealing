import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Pareto_Simulated_Annealing:

    def __init__(self, random_state=0, verbosity = 0):
        self.random_state = random_state
        self.verbosity = verbosity

    def initialize_weight(self,n_coef):
        w = np.random.normal(0, 1, size=(1, n_coef))
        b = np.random.normal(0, 1)
        return w,b

    def logistic_regression_predict(self,w,b,x,threshold=0.5):
        return np.where((1/(1+np.exp(-(x@w.T+b))))>threshold,1,0).ravel()
    
    def logistic_regression_predict_proba(self,w,b,x,threshold=0.5):
        return (1/(1+np.exp(-(x@w.T+b)))).ravel()

    def underestimation_score(self,y_true,y_pred,SA):
    
        mydict = {}
        mydict['actual'] = y_true
        mydict['predicted'] = y_pred
        mydict['SA'] = SA
        us = pd.DataFrame(mydict)

        P_dash_FX0 = self.df_count_feat_val_match(us, 'predicted', 1, 'SA',0)
        P_FX0 = self.df_count_feat_val_match(us, 'actual', 1, 'SA',0)
        Bias_FX0 = P_dash_FX0/P_FX0
        
        if P_FX0 == 0:
            print("Divsion by zero detected!")
                   
        return Bias_FX0

    def df_count_feat_val_match(self,df1, f1, v1, f2, v2):
        return len (df1[(df1[f1]==int(v1)) & (df1[f2]==int(v2))])

    def evaluate(self,y_true,y_pred,sa):
        ba = balanced_accuracy_score(y_true,y_pred)
        us_s = self.underestimation_score(y_true,y_pred,sa)
        return ba,us_s

    def identify_pareto(self,scores):
        # Count number of items
        population_size = scores.shape[0]
        # Create a NumPy index for scores on the pareto front (zero indexed)
        population_ids = np.arange(population_size)
        # Create a starting list of items on the Pareto front
        # All items start off as being labelled as on the Parteo front
        pareto_front = np.ones(population_size, dtype=bool)
        # Loop through each item. This will then be compared with all other items
        for i in range(population_size):
            # Loop through all other items
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    def fit(self, x_train, y_train):

        ## initialize PSA parameters
        Set_pareto = []
        pareto_param = []
        T_ba = 0.01
        T_us = 0.1
        beta = 0.05
        alpha = 0.95
    

        ## Randomly generate a set of initial solutions Set_theta
        Set_theta = []
        for i in range (1):
            w,b = self.initialize_weight(x_train.shape[1])
            Set_theta.append((w,b))

        for (w,b) in Set_theta:
            y_hat = self.logistic_regression_predict(w,b,x_train)
            best_ba,best_us = self.evaluate(y_train,y_hat,x_train[:,-1])

            while T_ba > 0 and T_us > 0:
                N_success = 0
                N_trials = 0
                while N_success < 1000 and N_trials < 10000:

                    ## generate candidate solutions
                    updates = np.zeros(x_train.shape[1]+1)
                    updates[np.random.choice(np.arange(len(updates)))] = np.random.normal(0, 1, 1)
                    w_candidate = w + updates[:x_train.shape[1]] * beta
                    b_candidate = b + updates[-1] * beta
                    y_hat = self.logistic_regression_predict(w_candidate,b_candidate,x_train)
                    candidate_ba,candidate_us = self.evaluate(y_train,y_hat,x_train[:,-1])

                    delta_E_ba = (best_ba - candidate_ba)
                    delta_E_us = (abs(1-best_us) - abs(1-candidate_us))

                    if (delta_E_ba < 0 and delta_E_us > 0):
                        w = w_candidate
                        b = b_candidate
                        best_ba = candidate_ba
                        best_us = candidate_us
                        N_success += 1
                        Set_pareto.append((best_ba,best_us))
                        pareto_param.append((w,b))
                    else:
                        prob_us = abs(delta_E_us) / T_us
                        prob_ba = delta_E_ba/T_ba
                        delta_E = prob_ba + prob_us
                        accept_prob = min(1,np.exp(-delta_E))
                        if accept_prob > np.random.rand():
                            w = w_candidate
                            b = b_candidate
                            best_ba = candidate_ba
                            best_us = candidate_us
                            Set_pareto.append((best_ba,best_us))
                            pareto_param.append((w,b))
                            N_success += 1
                        else:
                            N_trials += 1

                ## Decrease temperatures by Alpha
                T_ba = T_ba * alpha
                T_us = T_us * alpha
            
                if self.verbosity != 0:
                    print("Temp_ba: {}, Temp_US: {}".format(T_ba,T_us))
                    print("Best ba  :   {}".format(best_ba))
                    print("Best us_s:   {}".format(best_us))
                    print("Success #:   {}".format(N_success))
                    print("Trials  #:   {}".format(N_trials))
                    print("Coef     :   {}".format(w))
                    print("Intercept:   {}".format(b))
                    print("convergence: {}".format(convergence))
                    print("\n")
            

            # Convergence
            if N_success == 0:
                print("PSA has converged!")
                break
        return Set_pareto,pareto_param

        def filter_Pareto(Set_pareto,pareto_param):
            pareto_array = np.array(Set_pareto)
            inverser = lambda t: 1/(abs(1-t)+0.000000001)
            pareto_array[:,1] = [inverser(x) for x in pareto_array[:,1]]
            pareto_index = self.identify_pareto(pareto_array)

            #pareto filtered
            pareto_ba = [Set_pareto[idx][0] for idx in pareto_index]
            pareto_us = [Set_pareto[idx][1] for idx in pareto_index]
            pareto_w = [pareto_param[idx][0] for idx in pareto_index]
            pareto_b = [pareto_param[idx][1] for idx in pareto_index]

            return pareto_ba,pareto_us,pareto_w,pareto_b
