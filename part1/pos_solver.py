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
    #POS = ['noun', 'verb', 'adj', 'adv', 'adp', 'conj', 'det', 'num', 'pron', 'prt', 'x', '.']
    #prob = []

    def __init__(self):
        self.Prior_probabilities = {}
        self.Transition_probabilities = {}
        self.denominator = {}
        self.Emission_probabilities = {}
        self.POS = ['noun', 'verb', 'adj', 'adv', 'adp', 'conj', 'det', 'num', 'pron', 'prt', 'x', '.']
        self.prob_S = []
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
                self.Prior_probabilities[key] = 0.000001   ## Changed from 1 to 0.000001
                
            if key not in self.denominator:
                self.denominator[key] = 1
                
            if key not in self.Transition_probabilities:
                self.Transition_probabilities[key] = {}
                for tag in self.POS:
                    if tag not in self.Transition_probabilities[key]:
                        self.Transition_probabilities[key][tag] = 0.000001   ## Changed from 1 to 0.000001
            else:
                for tag in self.POS:
                    if tag not in self.Transition_probabilities[key]:
                        self.Transition_probabilities[key][tag] = 0.000001   ## Changed from 1 to 0.000001


        #Calculate the actual values of Prior probabilities
        for tag in self.Prior_probabilities:
            self.Prior_probabilities[tag] = self.Prior_probabilities[tag] / float(len(data))          

        #Calculate the actual values of Transition probabilities
        for key in self.Transition_probabilities:
            for tag in self.Transition_probabilities[key]:
                self.Transition_probabilities[key][tag] = self.Transition_probabilities[key][tag] / float(self.denominator[key])
            
        #For emission assigning the default value for labels not present in the dictionary as well as
        #calculating the Emission probabilities.
        for word in self.Emission_probabilities:
            for tag in self.POS:
                if tag not in self.Emission_probabilities[word]:
                    self.Emission_probabilities[word][tag] = 0.000001   ## Changed from 1 to 0.000001
            for word_tag in self.Emission_probabilities[word]:
                self.Emission_probabilities[word][word_tag] = self.Emission_probabilities[word][word_tag] / float(self.denominator[word_tag])    

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        prediction = []
        del self.prob_S[:]
    
        for i, word in enumerate(sentence):
            Simplified = {}
            Simplified[word] = {}
            for word_tag in self.POS:
                if word in self.Emission_probabilities:
                    Simplified[word][word_tag] = (self.Emission_probabilities[word][word_tag]) * (self.Prior_probabilities[word_tag])
                else:
                    Simplified[word][word_tag] = self.Prior_probabilities[word_tag]
                        
            self.prob_S.append(max(Simplified[word].iteritems(), key=operator.itemgetter(1))[1])
            prediction.append(max(Simplified[word].iteritems(), key=operator.itemgetter(1))[0])                                          
        return prediction


    def hmm_ve(self, sentence):
        prediction = []
        del self.prob_VE[:]
        alpha = {}

        for i, word in enumerate(sentence):
            alpha_index = str(i)
            alpha[alpha_index] = {}
            for word_tag in self.POS:
                if alpha_index == "0":
                    if word in self.Emission_probabilities:
                        alpha[alpha_index][word_tag] = (self.Emission_probabilities[word][word_tag]) * (self.Prior_probabilities[word_tag])
                    else:
                        alpha[alpha_index][word_tag] = self.Prior_probabilities[word_tag]
                else:
                    for k in self.POS:
                        if word_tag in alpha[alpha_index]:
                            alpha[alpha_index][word_tag] += alpha[str(i - 1)][k] * self.Transition_probabilities[k][word_tag]
                        else:
                            alpha[alpha_index][word_tag] = alpha[str(i - 1)][k] * self.Transition_probabilities[k][word_tag]

                    if word in self.Emission_probabilities:
                        alpha[alpha_index][word_tag] *= self.Emission_probabilities[word][word_tag]

            self.prob_VE.append(max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[1])
            prediction.append(max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[0])
            #print self.prob[i]
        return prediction

    def hmm_viterbi(self, sentence):
        prediction = []
        del self.prob_MAP[:]
        alpha = {}
        backtrack = {}
        last_prediction = ""
        last_prob = 0
        
        for i, word in enumerate(sentence):
            if i not in backtrack:
                backtrack[i] = {}
                
            #print "i is :", i
            alpha_index = i
            alpha[alpha_index] = {}
            for word_tag in self.POS:                    
                if alpha_index == 0:
                    if word in self.Emission_probabilities:
                        alpha[alpha_index][word_tag] = (self.Emission_probabilities[word][word_tag]) * (self.Prior_probabilities[word_tag])
                        #print "If word {0} exists in self.Emission_probabilities then the output for word_tag {1} is: {2}" .format(word, word_tag, str(alpha[alpha_index][word_tag]))
                    else:
                        alpha[alpha_index][word_tag] = self.Prior_probabilities[word_tag]
                        #print "If word {0} does not exist in self.Emission_probabilities then the output for word_tag {1} is: {2}" .format(word, word_tag, str(alpha[alpha_index][word_tag]))
                else:
                    next_list = []
                    for j, k in enumerate(self.POS):
                        #if word_tag in alpha[alpha_index]:
                            #alpha[alpha_index][word_tag] += alpha[str(i - 1)][k] * self.Transition_probabilities[k][word_tag]
                        #else:
                            #alpha[alpha_index][word_tag] = alpha[str(i - 1)][k] * self.Transition_probabilities[k][word_tag]
                        #print "Current Word Tag is: ", word_tag
                        #print "K is: ", k
                        #print "Alpha dict is: ", alpha
                        #print "Alpha value is: ", alpha[str(i - 1)][k]
                        #print "Transition values are: ", self.Transition_probabilities[k] 
                        #next_list.append(alpha[str(i - 1)][k] * self.Transition_probabilities[k][word_tag])
                        next_list +=[(k,(alpha[i - 1][k] * self.Transition_probabilities[k][word_tag]))]
                        #print "Next value is: ", next_list[j];
            
                    #print "Next list is: ", next_list
                    #print "Max of list of {0} is: {1}" .format(word_tag, max(next_list))
                    value = max(next_list, key=operator.itemgetter(1))
                    backtrack[i][word_tag] = value
                    #print "Max value is: ", value
                    alpha[alpha_index][word_tag] = value[1]
                    
                    if word in self.Emission_probabilities:
                        if word_tag in alpha[alpha_index]:
                            alpha[alpha_index][word_tag] *= self.Emission_probabilities[word][word_tag]
                            
        #print "Value of alpha index is :", alpha_index
        last_prediction = max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[0]                    
        last_prob = max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[1]
        prediction.append(last_prediction)
        self.prob_MAP.append(last_prob)
            #self.prob_MAP.append(max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[1])
            #prediction.append(max(alpha[alpha_index].iteritems(), key=operator.itemgetter(1))[0])
            #print self.prob[i]
        #return prediction
        for j in range(len(sentence)-1, 0, -1):
            previous = backtrack[j][last_prediction]
            prediction.append(previous[0])
            self.prob_MAP.append(previous[1])
            last_prediction = previous[0]
            
        prediction = prediction[::-1]
        self.prob_MAP = self.prob_MAP[::-1]
        #print "Prediction for sentence {0} is: {1}" .format(sentence, prediction)
        #return ['noun'] * len(sentence)
        return prediction

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

