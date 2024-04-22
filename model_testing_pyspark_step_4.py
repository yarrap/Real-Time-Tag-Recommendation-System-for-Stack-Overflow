from pyspark.sql import SparkSession
from flask import Flask, jsonify
from pyspark.sql.functions import explode, udf, concat_ws, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import SparkSession
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from itertools import combinations
import requests
import json

app = Flask(__name__)

# Define an API endpoint '/api/<type>' to handle requests
@app.route('/api/<type>', methods=["GET"])
def stackoverflow_data(type):
    from pyspark.sql import SparkSession

    # Create a SparkSession with necessary configurations
    spark = SparkSession.builder \
        .appName("Spark NLP")\
        .master("local[3]")\
        .config("spark.driver.memory", "16G")\
        .config("spark.kryoserializer.buffer.max", "2000M")\
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.4")\
        .getOrCreate()

    # Set Spark log levels and configurations
    spark.sparkContext.setLogLevel("INFO")
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100m")  # Set the threshold to 100MB
    from pyspark.sql import functions as F
    
    # Read data from 'data.csv' file
    data_path = 'data.csv'
    data = spark.read.csv(data_path, header=True)
    
    # Preprocess data: substring columns 'Body' and 'Title', explode 'Tags' column, index labels
    data = data.withColumn("index", F.monotonically_increasing_id())
    data = data.withColumn('Body', substring(data['Body'], 1, 300))
    data = data.withColumn('Title', substring(data['Title'], 1, 300)) 
    split_data = data.withColumn("Tags_array", split("Tags", ","))
    exploded = split_data.withColumn("exploded_labels", explode("Tags_array"))
    indexer = StringIndexer(inputCol="exploded_labels", outputCol="indexed_label")
    indexed = indexer.fit(exploded).transform(exploded)
    df  = indexed
    
    # Concatenate 'Body' and 'Title' columns, tokenize, compute TF-IDF, and assemble features
    delimiter = " "
    df = df.withColumn("Body", concat_ws(delimiter, "Body"))    
    delimiter = " "
    df = df.withColumn("Title", concat_ws(delimiter, "Title"))
    tokenizer = Tokenizer(inputCol="Body", outputCol="words_body")
    df = tokenizer.transform(df)
    tokenizer = Tokenizer(inputCol="Title", outputCol="words_title")
    df = tokenizer.transform(df)
    hashingTF_title = HashingTF(inputCol="words_title", outputCol="raw_title", numFeatures=100)
    # ... (continued processing steps)

    # Model training and evaluation
    # ... (model training and evaluation steps)

    # Data manipulation and filtering
    # ... (data filtering and manipulation steps)

    # Convert DataFrame to JSON
    appended_df = appended_df.toJSON().collect()
    json_array = '[' + ', '.join(appended_df) + ']'
    formatted_json = json.dumps(json.loads(json_array), indent=2)
    
    # Return JSON response
    return jsonify(data=formatted_json)

if __name__ == '__main__':
    app.run()
