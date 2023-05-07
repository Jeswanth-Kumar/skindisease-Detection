from flask import Flask, render_template, request
from PIL import Image
import io
import boto3
from keras.models import load_model
import uuid
import numpy as np
from werkzeug.utils import secure_filename

# Load pre-trained model
#model = load_model('custom.h5')
model_2 = load_model('skincancer1.h5')


s3 = boto3.client('s3',
                  aws_access_key_id='AKIATZCUWCKP7DPNYPXR',
                  aws_secret_access_key='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')

app = Flask(__name__)
data=['BENIGN','MALIGNANT']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    BUCKET_NAME='skindisease69'
    # Get uploaded image
    if request.method == 'POST':
        img_1 = request.files['image']
        if img_1:
                filename = secure_filename(img_1.filename)
                img_1.save(filename)
                s3.upload_file(
                    Bucket = BUCKET_NAME,
                    Filename=filename,
                    Key = filename
                )
                
    file = request.files['image']
    img = Image.open(file)
    
    # Preprocess image
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Use the model to make a prediction on the image
    
    pred_2= model_2.predict(img_array)
    pred_2_prob=max(pred_2[0])*100
    pred_2=np.argmax(pred_2,axis=1)
    pred_2=data[pred_2[0]]
        
    return render_template('result.html',result_2=pred_2,prob_2=pred_2_prob)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
