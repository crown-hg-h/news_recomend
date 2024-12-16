from flask import Flask, request,jsonify
from openai import OpenAI
import pandas as pd
import os
import numpy as np
os.environ["https_proxy"] = "http://localhost:7890"

import mysql.connector
# 数据库连接参数
config = {
    'user': 'root',
    'password': 'sB4L4XfTNarkuAyD',
    'host': '192.168.23.248',
    'database': 'article_generation',
    'raise_on_warnings': True
}
# 建立连接
cnx = mysql.connector.connect(**config)
df=pd.read_sql('''SELECT article_id,article_full_text FROM article_info where create_date='2024-10-14'    ''', con=cnx)

client = OpenAI( api_key="") 
def get_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model="text-embedding-3-large").data[0].embedding
embedding=[]
for i in df['article_full_text']:
    embedding.append(get_embedding(str(i)))
    print(i)
df['embedding']=embedding


app = Flask(__name__)
@app.route('/recommend', methods=['POST'])
def read_file():
    # 检查请求中是否有文件路径
    if 'article' not in request.json:
        return jsonify({'error': 'article'}), 400
    article = request.json['article']
    top = request.json['top']
    print(article)

    vector_a=np.array(get_embedding(article))    
    print(article)
    score=[]
    for i in df['embedding']:
        vector_b = np.array(i)
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        cosine_similarity = dot_product / (norm_a * norm_b)
        score.append(cosine_similarity)
    df['score']=score
    index=list(df.nlargest(top, 'score')['article_id'].values)
    
    json_data = {
        "paper_id": str(index),
        "code": 200
    }
    print(json_data)  
    return json_data


if __name__ == '__main__':
    app.run(debug=True,port=2222,host="0.0.0.0")
