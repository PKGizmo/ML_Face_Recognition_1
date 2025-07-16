# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sklearn
import pickle
import cv2
# from PIL import Image

# Loading Models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
mean = pickle.load(open('./model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))

# Settings
gender_pre = ['M', 'K']
font = cv2.FONT_HERSHEY_SIMPLEX


def pipeline_model(path, filename, color='bgr'):

    # Step 1: read image in cv2
    img = cv2.imread(path)

    # Step 2: convert into grayscale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Step 3: crop face from image using Haar Cascade Classifier
    faces = haar.detectMultiScale(gray, 1.5, 3)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)  # draw rectangle
        roi = gray[y:y+h, x:x+w]  # crop image

        # Step 4: normalization (0-1)
        roi = roi / 255.0

        # Step 5: resize image into 100 x 100 array
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)

        # Step 6: flattening (1x10,000)
        roi_reshape = roi_resize.reshape(1, 10000)  # 1, -1

        # Step 7: subtract with mean
        roi_mean = roi_reshape - mean

        # Step 8: get eigen image - apply PCA model
        eigen_image = model_pca.transform(roi_mean)

        # Step 9: pass to ML Model (SVM)
        results = model_svm.predict_proba(eigen_image)[0]

        # Step 10:
        predict = results.argmax()  # 0 (male) or 1 (female)
        score = results[predict]

        # Step 11:
        text = f"{gender_pre[predict]} : {score:.2f}"
        cv2.putText(img, text, (x, y-25), font, 1, (0, 255, 0), 2)

    cv2.imwrite(filename=f'static/predict/{filename}', img=img)
