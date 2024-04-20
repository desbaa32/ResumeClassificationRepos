import pandas as pd
#1.Drop column 'ID' and 'Resume_html' 
# Rename Resume_str columns to Resume and save the data into 
def drop_column(data):
    data.drop(columns = ['ID', 'Resume_html'], inplace = True)
    data = data.rename(columns={"Resume_str": "Resume"}, errors="raise")
    data.to_csv('../../data/interim/intermediate_Resume.csv',index='false')
    return data
 
def data_exploration(data):
    print(data.describe())
    print("\n \t ",data.info())
    print("\n \t ---___SHAPE___--- : \n ",data.shape)
    print("\n \t ---___Valeur null-___--- : \n",data.isnull().sum())
    print("\n \t ---___Nombre de cvs par rapport aux categorie-___--- : \n",data['Category'].value_counts()) 
def  preprocess(txt):
    txt = txt.lower() # Convert to lowercase
    txt = re.sub('http\S+\s*', ' ', txt)  # remove URLs
    txt = re.sub('RT|cc', ' ', txt)  # remove RT and cc
    txt = re.sub(r'<.*?>', ' ', txt)
    # Remove special characters, digits, continous underscores and 
    txt = re.sub(r'[^\w\s]|_', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    #remove non-english characters, punctuation and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    
    #--------------------NLP technique---------------------
    # Tokenize the cleaned text
    words = word_tokenize(txt)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    # Apply stemming and lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmatizer_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    txt = ' '.join(lemmatizer_words)
    pos_tags = pos_tag(word_tokenize(txt))
    pos_tags_str = ' '.join([f"{word}_{tag}" for word, tag in pos_tags ]) 
    return pos_tags_str


# try to remove extra word which are note important.
def remove_extra_word(text):
    
    extra_word=['company_NN', 'name_NN', 'city_NN', 'state_NN'] # extra words
    words = text.split()  # Split the text into words
    
    # Filter out the extra words
    filter_word = [word for word in words if word not in extra_word]
    
    filter_text = ' '.join(filter_word)
    
    return filter_text