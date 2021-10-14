import tensorflow as tf
import numpy as np
import pandas as pd
import pickle as pkl



class PIBB(object):
    """
    Implementation of Black Box Path Integrals from paper ...
    
    
    Params:
        - tensorflow/keras neural network model
        - revenue function
        - saving path
        - max iters
        - dim: number of params
        - n_calls:
        - K: number of exploratory points
        . perturbation_intensity: sttdev in case of gaussian perturbation/bound in case of uniform distribution
        - alpha:
        - decay:
        - decay2:
        - perturb: kind of perturbation
    """
    
    def __init__(
        self, 
        model: tf.keras.Model = None,  
        revenue_function: function = None,
        saving_path: str = None,
        
        ) -> None:
        
        self.model = model
        self.n_calls = 0
        self.revf = revenue_function
        self.saving_path = saving_path
        self.dim = self.agent.model.count_params()
        self.w = 0
        self.losses = []    
    
    def run(self, K, stddev, max_iters, alpha, decay, decay2, perturb):
        alphas = []
        all_dims = np.arange(self.dim)
        y = np.zeros(self.dim) 
        self.w = y
        loss = self.loss_function(y)
        self.losses.append(loss)
        initial_loss = loss
        for j in range(max_iters):
            noises = []
            
            for k in range(K):
                if perturb == 'gaussian':
                    noise = np.random.normal(0, stddev, self.dim)
                elif perturb == 'uniform':
                    noise = np.random.uniform(-stddev, stddev, self.dim)
                z = y + noise
                noises.append((self.loss_function(z)-loss, noise))
                self.losses.append(loss)
            ordered_noises = sorted(noises, key=itemgetter(0))
            costs = np.array([c[0]*0.05 for c in ordered_noises])
            noises = np.array([c[1] for c in ordered_noises])
            
            p = np.exp(-(1/alpha)*costs)
            print(sum(p))
            probs = p/sum(p)
            print(probs)
            final_noise = np.dot(probs, noises)
            y = y + final_noise
            new_loss = self.loss_function(z)
            
            print(new_loss)
            c = 0
            if new_loss < loss:
                self.w = y
                loss = new_loss

            self.losses.append(loss)
            stddev *= decay2
            if sum(p) < 1e-100 or sum(p) > 1e+100:
                alpha = alphas[-1]
                decay = 1
            else:
                alpha *= decay
                alphas.append(alpha)
        
        return self.w, self.losses