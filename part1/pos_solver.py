###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
import operator

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    POS = ['noun', 'verb', 'adj', 'adv', 'adp', 'conj', 'det', 'num', 'pron', 'prt', 'x', '.']
    prob = []

    def __init__(self):
        self.Prior_probabilities = {}
        self.Transition_probabilities = {}
        self.denominator = {}
        self.Emission_probabilities = {}
        self.prob_S = []
        self.POS = ['noun', 'verb', 'adj', 'adv', 'adp', 'conj', 'det', 'num', 'pron', 'prt', 'x', '.']
        self.prob_VE=[]
        self.prob_MAP=[]
        
    # Calculate the log of the posterior self.probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, algo, label):
        log_vals = 0
        
        if algo == "1. Simplified":
            for i in range(len(label)):
                # Use the list created for simplified algorithm
                log_vals += math.log(self.prob_S[i])
            return log_vals
        
        elif algo == "2. HMM VE":
            for i in range(len(label)):
                # Use the list created for VE algorithm
                log_vals += math.log(self.prob_VE[i])
            return log_vals
        
        elif algo == "3. HMM MAP":
            for i in range(len(label)):
                # Use the list created for MAP algorithm
                log_vals += math.log(self.prob_MAP[i])
            return log_vals
        
        elif algo == "0. Ground truth":
            return 100
    
    # Do the training!
    #
    def train(self, data):
##        self.Prior_probabilities = {}
##        self.Transition_probabilities = {}
##        self.denominator = {}
##        self.Emission_probabilities = {}
        
        for sentence_with_pos in data:
            ## Calulates the Transition count
            for i, state_of_i in enumerate(sentence_with_pos[1]):
                if state_of_i not in self.Transition_probabilities:
                    self.Transition_probabilities[state_of_i] = {}
                    
                if (i+1) < (len(sentence_with_pos[1])):
                    state_of_i_plus_1 = sentence_with_pos[1][i+1]
                    for j in self.POS:
                        for k in self.POS:
                            if state_of_i == j and state_of_i_plus_1 == k:
                                if state_of_i in self.denominator:
                                    self.denominator[state_of_i] += 1
                                else:
                                    self.denominator[state_of_i] = 1
                                
                                if state_of_i_plus_1 in self.Transition_probabilities[state_of_i]:
                                    self.Transition_probabilities[state_of_i][state_of_i_plus_1] += 1
                                else:
                                    self.Transition_probabilities[state_of_i][state_of_i_plus_1] = 1

            for i, word in enumerate(sentence_with_pos[0]):
                if word not in self.Emission_probabilities:
                    self.Emission_probabilities[word] = {}
                    
                for j in self.POS:
                    ## Calculates Prior count
                    if i == 0 and sentence_with_pos[1][0] == j:
                        if j in self.Prior_probabilities:
                            self.Prior_probabilities[j] += 1
                        else:
                            self.Prior_probabilities[j] = 1
                            
                    ## Calculates Emission count
                    word_tag = sentence_with_pos[1][i]
                    if word_tag == j:
                        if word_tag in self.Emission_probabilities[word]:
                            self.Emission_probabilities[word][word_tag] += 1
                        else:
                            self.Emission_probabilities[word][word_tag] = 1


        #Check if the label is present in the dictionary if not add it and give the value of it as 1
        #only for prior and transition.
        for key in self.POS:
            if key not in self.Prior_probabilities:
                self.Prior_probabilities[key] = 1
                
            if key not in self.denominator:
                self.denominator[key] = 1
                
            if key not in self.Transition_probabilities:
                self.Transition_probabilities[key] = {}
                for tag in self.POS:
                    if tag not in self.Transition_probabilities[key]:
                        self.Transition_probabilities[key][tag] = 1
            else:
                for tag in self.POS:
                    if tag not in self.Transition_probabilities[key]:
                        self.Transition_probabilities[key][tag] = 1


        #Calculate the actual values of probabilities
        for tag in self.Prior_probabilities:
            self.Prior_probabilities[tag] = self.Prior_probabilities[tag] / float(len(data))          

        for key in self.Transition_probabilities:
            for tag in self.Transition_probabilities[key]:
                self.Transition_probabilities[key][tag] = self.Transition_probabilities[key][tag] / float(self.denominator[key])
            
        #For emission assigning the default value for labels not present in the dictionary as well as
        #calculating the final probabilities.
        for word in self.Emission_probabilities:
            for tag in self.POS:
                if tag not in self.Emission_probabilities[word]:
                    self.Emission_probabilities[word][tag] = 1   
            for word_tag in self.Emission_probabilities[word]:
                self.Emission_probabilities[word][word_tag] = self.Emission_probabilities[word][word_tag] / float(self.denominator[word_tag])    

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        prediction = []
        self.prob = []
    
        for i, word in enumerate(sentence):
            Simplified = {}
            Simplified[word] = {}
            for word_tag in self.POS:
                if word in self.Emission_probabilities:
                    Simplified[word][word_tag] = (self.Emission_probabilities[word][word_tag]) * (self.Prior_probabilities[word_tag])
                else:
                    Simplified[word][word_tag] = self.Prior_probabilities[word_tag]
                        
            self.prob.append(max(Simplified[word].iteritems(), key=operator.itemgetter(1))[1])
            prediction.append(max(Simplified[word].iteritems(), key=operator.itemgetter(1))[0])                                          
        return prediction


    def hmm_ve(self, sentence):

            prediction = []
            alpha = {}

            for i, word in enumerate(sentence):
                alpha_index = str(i)
                alpha[alpha_index] = {}
                for word_tag in self.POS:
                    if alpha_index == "0":
                        alpha[alpha_index][word_tag] = (self.Emission_probabilities[word][word_tag]) * (
                            self.Prior_probabilities[word_tag])
                    else:
                        for k in self.POS:
                            if word_tag in alpha[alpha_index]:
                                alpha[alpha_index][word_tag] += alpha[str(i - 1)][k] * self.Transition_probabilities[k][
                                    word_tag]
                            else:
                                alpha[alpha_index][word_tag] = alpha[str(i - 1)][k] * self.Transition_probabilities[k][
                                    word_tag]

                        alpha[alpha_index][word_tag] *= self.Emission_probabilities[word][word_tag]

                self.prob.append(max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[1])
                prediction.append(max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[0])
                # print self.prob[i]
            return prediction

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

