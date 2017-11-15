#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import math
import io
import copy

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print im.size
    #print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print "\n".join([ r for r in train_letters['.']])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[0]])


#code
#calculate prior probability for characters using training text files P(pixel-value|Letter)
lp=0.00000001
blanks=[[0 for i in range(14)] for j in range(25)]
stars=[[0 for i in range(14)] for j in range(25)]
for i in range(25) :
    for j in range (14) :
        counts=0
        countb=0
        for key,value in train_letters.iteritems() :
            if value[i][j]=="*" :
                counts+=1
            else :
                countb+=1
        blanks[i][j]=countb
        stars[i][j]=counts
#print blanks
#print stars
#count stars in train letters
# stars_train={}
# for key,value in train_letters.iteritems():
#     count=0
#     for i in range (25) :
#         for j in range (14) :
#             if value[i][j]=="*":
#                 count+=1
#     stars_train[key]=count
# print "here"
# print stars_train

#using fig b to estimate letters
def simpleocr(test_letters) :
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    esti=""
    prob={}
    for i in range(len(test_letters)):
        for c in TRAIN_LETTERS :
            prob[c]=math.log(((float)(1)/(float)(72)))
        for j in range(len(test_letters[i])) :
            for k in range(len(test_letters[i][j])) :
                if test_letters[i][j][k]=="*" :
                    for key,value in train_letters.iteritems() :
                        if value[j][k]=="*" :
                            prob[key]+=math.log((float)(1)/(float)(stars[j][k]))
                        else :
                            prob[key]+=math.log((float)(lp))
                elif test_letters[i][j][k]==" " :
                    for key,value in train_letters.iteritems() :
                        if value[j][k]==" " :
                            prob[key]+=math.log((float)(1)/(float)(blanks[j][k]))
                            break
                        else :
                            prob[key]+=math.log((float)(lp))
        esti=esti+prob.keys()[prob.values().index(max(prob.values()))]
    return esti

#print simpleocr(test_letters)

#calculating transition probabilities and initial Probability
f=open(train_txt_fname,"rb")
l=f.read().splitlines()
trans_occur={}
initial_occur={}
char_occur={}
line_count=len(l)
for line in l :
    for index in range(len(line)-1) :
        if line[index] not in char_occur.keys() :
            char_occur[line[index]]=1
        else :
            char_occur[line[index]]+=1
        if index==0 :
            if line[index] not in initial_occur.keys() :
                initial_occur[line[index]]=1
            else :
                initial_occur[line[index]]+=1
        if (line[index],line[index+1]) not in trans_occur.keys() :
            trans_occur[(line[index],line[index+1])]=1
        else :
            trans_occur[(line[index],line[index+1])]+=1

#print "trans prob of space and cap T :"
#print trans_occur[(' ','T')]
#print ((float)(trans_occur[(' ','T')])/(float)(char_occur[' ']))
#print ((float)(intial_occur['T'])/(float)(line_count))

#using fig a to estimate letters
def viterbi(test_letters) :
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    esti=""
    prob={}
    previous_prob={}
    masterlist=[]
    back_traverse={}
    for i in range(len(test_letters)):
        if i==0 :
            for c in TRAIN_LETTERS :
                if c in initial_occur.keys() :
                    prob[c]=math.log((float)(initial_occur[c])/(float)(line_count))
                else :
                    prob[c]=math.log((float)(lp))
            for j in range(len(test_letters[i])) :
                for k in range(len(test_letters[i][j])) :
                    if test_letters[i][j][k]=="*" :
                        for key,value in train_letters.iteritems() :
                            if value[j][k]=="*" :
                                prob[key]+=math.log((float)(1))
                            else :
                                prob[key]+=math.log((float)(lp))
                    # elif test_letters[i][j][k]==" " :
                    #     if j > 5 and j <=20 :
                    #         for key,value in train_letters.iteritems() :
                    #             if value[j][k]==" " :
                    #                 prob[key]+=math.log((float)(0.1))
                    #             else :
                    #                 prob[key]+=math.log((float)(lp))
            previous_prob=copy.deepcopy(prob)
        if i>0 :
            maxdic={}
            for c in TRAIN_LETTERS :
                    prob[c]=0
            for j in range(len(test_letters[i])) :
                for k in range(len(test_letters[i][j])) :
                    if test_letters[i][j][k]=="*" :
                        for key,value in train_letters.iteritems() :
                            if value[j][k]=="*" :
                                prob[key]+=math.log((float)(1))
                            else :
                                prob[key]+=math.log((float)(lp))
                    elif test_letters[i][j][k]==" " :
                        if j>5 and j<=20 :
                            for key,value in train_letters.iteritems() :
                                if value[j][k]==" " :
                                    prob[key]+=math.log((float)(0.1))
                                else :
                                    prob[key]+=math.log((float)(lp))
            for ch in TRAIN_LETTERS :
                mmax=-1000000000000
                mchar=""
                for key,value in previous_prob.iteritems() :
                    if (key,ch) in trans_occur.keys() :
                        temp=previous_prob[key]+math.log((float)(trans_occur[(key,ch)])/(float)(char_occur[key]))
                    else :
                        temp=previous_prob[key]+math.log(lp)
                    if temp > mmax :
                        mchar=key
                        mmax=temp
                maxdic[ch]=mchar
                prob[ch]+=mmax
            masterlist.append(maxdic)
            previous_prob=copy.deepcopy(prob)
    last=prob.keys()[prob.values().index(max(prob.values()))]
    for i in range (len(masterlist)) :
        esti+=masterlist[i][last]
        last=masterlist[i][last]
    esti+=prob.keys()[prob.values().index(max(prob.values()))]
    return esti

def ve(test_letters) :
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    esti=""
    prob={}
    previous_prob={}
    for i in range(len(test_letters)):
        if i==0 :
            for c in TRAIN_LETTERS :
                if c in initial_occur.keys() :
                    #prob[c]=((float)(initial_occur[c])/(float)(line_count))
                    prob[c]=((float)(initial_occur[c]))
                    #prob[c]=1
                else :
                    prob[c]=((float)(0.01))
            for j in range(len(test_letters[i])) :
                for k in range(len(test_letters[i][j])) :
                    if test_letters[i][j][k]=="*" :
                        for key,value in train_letters.iteritems() :
                            if value[j][k]=="*" :
                                prob[key]=prob[key]*((float)(1))
                                #prob[key]+=math.log((float)(0.9))
                            else :
                                prob[key]=prob[key]*((float)(0.1))
                    # elif test_letters[i][j][k]==" " :
                    #     for key,value in train_letters.iteritems() :
                    #         if value[j][k]==" " :
                    #             prob[key]+=math.log((float)(0.1))
                    #         else :
                    #             prob[key]+=math.log((float)(0.001))
            esti=esti+prob.keys()[prob.values().index(max(prob.values()))]
            previous_prob=copy.deepcopy(prob)
        if i>0 :
            for c in TRAIN_LETTERS :
                    prob[c]=1
            for j in range(len(test_letters[i])) :
                for k in range(len(test_letters[i][j])) :
                    if test_letters[i][j][k]=="*" :
                        for key,value in train_letters.iteritems() :
                            if value[j][k]=="*" :
                                prob[key]=prob[key]*((float)(1))
                            else :
                                prob[key]=prob[key]*((float)(0.1))
                    # elif test_letters[i][j][k]==" " :
                    #     for key,value in train_letters.iteritems() :
                    #         if value[j][k]==" " :
                    #             prob[key]+=math.log((float)(0.1))
                    #         else :
                    #             prob[key]+=math.log((float)(0.001))
            tao={}
            for ch in TRAIN_LETTERS :
                sub_tao={}
                for key,value in previous_prob.iteritems() :
                    if (key,ch) in trans_occur.keys() :
                       #temp=previous_prob[key]*((float)(trans_occur[(key,ch)])/(float)(char_occur[key]))
                         temp=previous_prob[key]*((float)(trans_occur[(key,ch)]))
                    else :
                        temp=previous_prob[key]*((float)(0.1))
                    sub_tao[key]=temp
                tao[ch]=sum(sub_tao.values())*prob[ch]
            for key,value in prob.iteritems() :
                prob[key]=prob[key]#*tao[key]
            previous_prob=copy.deepcopy(prob)
            #print tao.keys()[tao.values().index(max(tao.values()))]
            esti=esti+tao.keys()[tao.values().index(max(tao.values()))]
            #print esti
    return esti

print "Simple: "+simpleocr(test_letters)
print "HMM VE: "+ve(test_letters)
print "HMM MAP: "+viterbi(test_letters)
