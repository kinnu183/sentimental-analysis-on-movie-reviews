import numpy as np
import re
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import KFold
import argparse
import os

vNegative = []
Negative = []
Positive = []
vPositive = []
data_X = ""
data_Y = ""

def generateStopWordList():

    #Fetch the Text File which has all the stopwords from the PATH
    stopWords_dataset = dirPath+"/stopwords.txt"

    #Stopwords List
    stopWords = []

    #Open the stopwords file read the data and store in a list
    try:
        fp = open(stopWords_dataset, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
    except:
        print("ERROR: Opening File")

    return stopWords

def generateAffinityList(datasetLink):

    affin_dataset = datasetLink
    try:
        affin_list = open(affin_dataset).readlines()
    except:
        print("ERROR: Opening File", affin_dataset)
        exit(0)
    #print(affin_list)

    return affin_list

def createDictionaryFromPolarity(affin_list):

    # Create list to store the words and its score i.e. polarity
    words = []
    score = []

    # for every word in AFF-111 list, generate the Words with their scores (polarity)
    for word in affin_list:
        words.append(word.split("\t")[0].lower())
        score.append(int(word.split("\t")[1].split("\n")[0]))

    #Categorize words into different Categories
    for elem in range(len(words)):
        if score[elem] == -4 or score[elem] == -5:
            vNegative.append(words[elem])
        elif score[elem] == -3 or score[elem] == -2 or score[elem] == -1:
            Negative.append(words[elem])
        elif score[elem] == 3 or score[elem] == 2 or score[elem] == 1:
            Positive.append(words[elem])
        elif score[elem] == 4 or score[elem] == 5:
            vPositive.append(words[elem])
    Positive.append('EMO_POS')
    Negative.append('EMO_NEG')

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'( :\s?\) | :-\) | \(\s?: | \(-: | :\'\) )', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'( :\s?D | :-D | x-?D | X-?D )', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'( <3 | :\* )', ' EMO_POS ', tweet)
    
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'( ;-?\) | ;-?D | \(-?; )', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'( :\s?\( | :-\( | \)\s?: | \)-: )', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'( :,\( | :\'\( | :"\( )', ' EMO_NEG ', tweet)
    return tweet

def preprocessing(dataSet):

    processed_data = []

    #Make a list of all the Stopwords to be removed
    stopWords = generateStopWordList()

    #For every TWEET in the dataset do,
    for tweet in dataSet:

        temp_tweet = tweet

        #Convert @username to USER_MENTION
        tweet = re.sub('@[^\s]+','',tweet).lower()
        tweet.replace(temp_tweet, tweet)

        #Remove the unnecessary white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Replace #HASTAG with only the word by removing the HASH (#) symbol
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        #Replace all the numeric terms
        tweet = re.sub('[0-9]+', "",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove all the STOP WORDS
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)

        tweet.replace(temp_tweet, tweet)

        # Replace emojis with either EMO_POS or EMO_NEG
        tweet = handle_emojis(tweet)
        
        #Replace all Punctuations
        tweet = re.sub('[^a-zA-z ]',"",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove additional white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Save the Processed Tweet after data cleansing
        processed_data.append(tweet)

    return processed_data

def FeaturizeTrainingData(dataset, type_class):

    neutral_list = []
    i=0

    data = [tweet.strip().split(" ") for tweet in dataset]

    feature_vector = []

    for sentence in data:
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0

        for word in sentence:
            if word in vPositive:
                vPositive_count = vPositive_count + 1
            elif word in Positive:
                Positive_count = Positive_count + 1
            elif word in vNegative:
                vNegative_count = vNegative_count + 1
            elif word in Negative:
                Negative_count = Negative_count + 1
        i+=1

        if vPositive_count == vNegative_count == Positive_count == Negative_count:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])
            neutral_list.append(i)
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, type_class])
    return feature_vector

def FeatureizeTestData(dataset):

    data = [tweet.strip().split(" ") for tweet in dataset]
    count_Matrix = []
    feature_vector = []

    for sentence in data:
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0
        li=[]
        for word in sentence:
              li.append(word)
        for i in range(len(li)-1):
            if (li[i] in vPositive and li[i+1] in vPositive):
                vPositive_count = vPositive_count + 1
            elif ((li[i] in vPositive and li[i+1] in Positive) or (li[i] in Positive and li[i+1] in vPositive)):
                vPositive_count = vPositive_count + 1
            elif ((li[i] in vPositive and li[i+1] in Negative) or (li[i] in Negative and li[i+1] in vPositive)):
                Positive_count = Positive_count + 1
            elif ((li[i] in vPositive and li[i+1] in vNegative) or (li[i] in vNegative and li[i+1] in vPositive)):
                Negative_count = Negative_count + 1
            elif ((li[i] in Positive and li[i+1] in Negative) or (li[i] in Negative and li[i+1] in Positive)):
                Negative_count = Negative_count + 1
            elif ((li[i] in Positive and li[i+1] in vNegative) or (li[i] in vNegative and li[i+1] in Positive)):
                vNegative_count = vNegative_count + 1
            elif ((li[i] in Negative and li[i+1] in vNegative) or (li[i] in vNegative and li[i+1] in Negative)):
                vNegative_count = vNegative_count + 1
            elif (li[i] in vNegative and li[i+1] in vNegative):
                vNegative_count = vNegative_count + 1
            elif (li[i] in Negative and li[i+1] in Negative):
                Positive_count = Positive_count + 1
            elif (li[i] in Positive and li[i+1] in Positive):
                Positive_count = Positive_count + 1
            elif((li[i] in Positive and li[i+1] not in(vPositive,vNegative,Negative))or(li[i] not in(vPositive,vNegative,Negative) and li[i+1] in Positive)):
                  Positive_count = Positive_count + 1
            elif((li[i] in vPositive and li[i+1] not in(Positive,vNegative,Negative)) or (li[i] not in(Positive,vNegative,Negative) and li[i+1] in vPositive)):
                  vPositive_count = vPositive_count + 1
            elif((li[i] in Negative and li[i+1] not in(vPositive,vNegative,Positive)) or (li[i] not in(vPositive,vNegative,Positive) and li[i+1] in Negative)):
                  Negative_count = Negative_count + 1
            elif((li[i] in vNegative and li[i+1] not in(vPositive,Negative,Positive)) or (li[i] not in(vPositive,Negative,Positive) and li[i+1] in vNegative)):
                  vNegative_count = vNegative_count + 1

        if (vPositive_count + Positive_count) > (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "positive"])
        elif (vPositive_count + Positive_count) < (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "negative"])
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])
    return feature_vector

#########FOR TEST DATA CLASSIFICATION########
def classify_LR_twitter(train_X, train_Y, test_X, test_Y):

    LR = LogisticRegressionCV()
    LR.fit(train_X, train_Y)
    yHat = LR.predict(test_X)
    i=j=k=0
    for word in yHat:
        if word == "positive" :
            i=i+1
        elif word == "neutral" :
            j=j+1
        elif word == "negative":
            k=k+1
	
    print("positive:",(i/(i+j+k)))
    print("neutral:",(j/(i+j+k)))
    print("negative:",(k/(i+j+k)))
    print (i+j+k)
    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuracy: ", Accuracy)
    evaluate_classifier(conf_mat)
    
def classify_twitter_data(file_name):

    test_data = open(dirPath+"/"+file_name, encoding="utf8").readlines()
    test_data = preprocessing(test_data)
    f=open("preprocessed_test_data.txt","w+")
    f.write(str(test_data))
    f.close()
    test_data = FeatureizeTestData(test_data)
    test_data = np.reshape(np.asarray(test_data),newshape=(len(test_data),5))

    #Split Data into Features and Classes
    data_X_test = test_data[:,:4].astype(int)
    data_Y_test = test_data[:,4]

    print("Classifying", args.DataSetName)
    classify_LR_twitter(data_X, data_Y, data_X_test, data_Y_test)

def evaluate_classifier(conf_mat):
    Recall = conf_mat[0,0]/(sum(conf_mat[0]))
    Precision = conf_mat[0,0] / (sum(conf_mat[:,0]))
    F_Measure = (2 * (Precision * Recall))/ (Precision + Recall)

    print("Precision: ",Precision)
    print("Recall: ", Recall)
    print("F-Measure: ", F_Measure)

# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sentimental Analysis of Movie Reviews")
    parser.add_argument("DataSetName", help="Dataset to Classify (rottom batvsuper junglebook zootopia deadpool)", metavar='dataset')
    args = parser.parse_args()

    #fetch th current working dir   
    dirPath = os.getcwd()

    # STEP 1: Generate Affinity List
    print("Please wait while we Classify your data ...")
    affin_list = generateAffinityList(dirPath+"/Affin_Data.txt")

    # STEP 2: Create Dictionary based on Polarities from the Lexicons
    createDictionaryFromPolarity(affin_list)

    # STEP 3: Read Data positive and negative Tweets, and do PREPROCESSING
    print("Reading your data ...")
    positive_data = open(dirPath+"/training-polarity-pos.txt").readlines()
    positive_data = preprocessing(positive_data)
    f1=open("positive.txt","w+")
    f1.write(str(positive_data)) 
    
    negative_data = open(dirPath+"/training-polarity-neg.txt").readlines()
    negative_data = preprocessing(negative_data)
    f2=open("negative.txt","w+")
    f2.write(str(negative_data))

    f1.close()
    f2.close()

    # STEP 4: Create Feature Vectors and Assign Class Label for Training Data
    print("Generating the Feature Vectors ...")
    positive_sentiment = FeaturizeTrainingData(positive_data, "positive")
    negative_sentiment = FeaturizeTrainingData(negative_data,"negative")
    final_data = positive_sentiment + negative_sentiment
    final_data = np.reshape(np.asarray(final_data),newshape=(len(final_data),5))

    #Split Data into Features and Classes
    data_X = final_data[:,:4].astype(int)
    data_Y = final_data[:,4]

    # Classifying Entire Dataset
    print("Training the Classifer according to the data provided ...")
    print("Classifying the Test Data ...")
    print("Evaluation Results will be displayed Shortly ...")
    
           
    #Classifying Test Data
    if args.DataSetName == "batvsuper":
        classify_twitter_data(file_name="BatmanvSuperman.txt")
    elif args.DataSetName == "junglebook":
        classify_twitter_data(file_name="junglebook.txt")
    elif args.DataSetName == "zootopia":
        classify_twitter_data(file_name="zootopia.txt")
    elif args.DataSetName == "deadpool":
        classify_twitter_data(file_name="deadpool.txt")
    else:
        print("ERROR while specifying Movie Tweets File, please check the name again")

