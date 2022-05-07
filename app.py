import torch
import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image


import skimage.filters as filters
import torch.nn as nn
import torch.nn.functional as F

import cv2
import mahotas
import imutils
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
import tensorflow as tf
import cv2 as cv2

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

#     # load model at very first
# model = tf.keras.models.load_model(STATIC_FOLDER + '/' + 'Sandy.h5')
#model_digit = tf.keras.models.load_model(STATIC_FOLDER + '/' +'handwritten_model.h5')
#model_operator = joblib.load(STATIC_FOLDER + '/' + "model_cls_operator.pkl")
#labels_name = ['*', '+', '-', 'div']

#from skimage import feature
"""
class HOG:
    def __init__(self, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), transform=False):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform = transform

    def describe(self, image):
        hist = feature.hog(image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            transform_sqrt=self.transform)
        return hist

def deskew(image, width):
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5*w*skew],
                    [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    image = imutils.resize(image, width=width)

    return image

def center_extent(image, size):
    (eW, eH) = size

    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=eW)
    else:
        image = imutils.resize(image, height=eH)

    extent = np.zeros((eH, eW), dtype='uint8')
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX+image.shape[1]] = image

    CM = mahotas.center_of_mass(extent)
    (cY, cX) = np.round(CM).astype("int32")
    (dX, dY) = ((size[0]//2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)

    return extent

hog_1 = HOG(orientations=18, pixels_per_cell=(10,10), cells_per_block=(1,1), transform=True)

def extract_hog(features):
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')
    return hog_features
"""
# call model to predict an image
#def api(full_path):
    #return full_path
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(45, 32, kernel_size=5, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.ffnn1 = nn.Linear(128*7, 500)
        self.ffnn2 = nn.Linear(500, 250)
        self.ffnn3 = nn.Linear(250, 45)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.reshape(output.size(0), -1)
        output = self.drop_out(output)
        output = self.ffnn1(output)
        output = self.ffnn2(output)
        output = self.ffnn3(output)
        return F.log_softmax(output)
        
net=Model().to(device)
print(net)   
"""def imshow(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))
    ax.imshow(image, cmap=cmap)
    return fig, ax
"""
def define_kernel(kernel_size, std, size):
    half_size = kernel_size // 2
    kernel = np.zeros([kernel_size, kernel_size])
    std_X = std
    std_Y = std * size
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i-half_size
            y = i-half_size
            
            exponent=np.exp(-x**2/(std_X*2)-y**2/(std_Y*2))
            x_ = (x**2-std_X**2)/(2*np.pi*std_X**5*std_Y)
            y_ = (y**2-std_Y**2)/(2*np.pi*std_Y**5*std_X)
            kernel[i, j] = (x_+y_)/exponent
    return kernel/np.sum(kernel)

def get_category(x):
    labels = np.load(STATIC_FOLDER + '/' + 'Labels_her.npy')
    return labels[x]

def predict_value(ip, model):
    model.eval()
    with torch.no_grad():
        output = model(ip)
        _, predicted = torch.max(output.data, 1)
        return get_category(predicted)



def recognize_image(image_read):
    net = Model()
    net.load_state_dict(torch.load(STATIC_FOLDER + '/' +'her_model', map_location='cpu'))
    image = cv2.resize(image_read, (400, 224))
    threshold = filters.threshold_local(image, block_size=195, offset=30)
    img = (image > threshold).astype(np.uint8)*255.
    kernel = define_kernel(5, 11, 7)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_REPLICATE).astype(np.uint8)
    _, img_threshold = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_threshold = 255-img_threshold
    _, components, _=cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    power_rect =[]
    for c in components:
        if cv2.contourArea(c) < 100:
            continue
        current_box = cv2.boundingRect(c)
        x, y, w, h = current_box
        seg_image = img[y:y+h, x:x+w]
        res.append((current_box, seg_image))
    sort = sorted(res, key=lambda entry:entry[0][0])
    predicted = []
    coordinate_y=[]
    mx = -1
    ans=""
    for j, w in enumerate(sort):
        word_box, _ = w
        x, y, w, h = word_box
        temp_img = img[y:y+h, x:x+w]/255.
        temp_img = cv2.resize(temp_img, (35, 35))
        temp_img = cv2.copyMakeBorder(temp_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value= [1.])
        temp_img = torch.from_numpy(temp_img).float()
        temp_img = temp_img.view(-1, 45, 45, 1)
        coordinate_y.append(y)
        mx = max(y, mx)
        print("y="+str(y))
        print(predict_value(temp_img, net))
        ans+=str(predict_value(temp_img, net))
        predicted.append(predict_value(temp_img, net))
    cutoff = 0.6*(mx)
    print("cutoff="+str(cutoff))
    for y in coordinate_y : 
        if y<cutoff :
          power_rect.append(1)
            
        else:
          power_rect.append(0)
    #i=0
    """for q in predicted:
        print(q)
        if power_rect[i]==1 :
          print("power hai")
        else:
          print("base hai")
        i+=1"""
        
    final=[]
    for q in predicted:
       
      if q=="alpha":
        final.append("\ alpha")
      elif q=="beta":
        final.append("\beta")
      elif q==",":
        final.append("/")
      elif q=="8":
        final.append("=")
      elif q=="gamma":
        final.append("\gamma")
      elif q=="Delta":
        final.append("\delta")
      elif q=="exists":
        final.append("\exists")
      elif q=="lt":
        final.append("<")
      elif q=="gt":
        final.append(">")
      elif q=="sum":
        final.append("\sum_{}^{}")
      #elif q=="int":
        #final.append("\int_{}^{}")
      elif q=="lambda":
        final.append("\lambda")
      elif q=="forall":
        final.append("\ forall")
      elif q=="mu":
        final.append("\mu")
      elif q=="theta":
        final.append("\theta")
      elif q=="phi":
        final.append("\phi")
      elif q=="pi":
        final.append("\pi")
      elif q=="sigma":
        final.append("\sigma")
      else:
        final.append(q)
    ans1 = "\["
    i = 0
    for a in final:
        
        if power_rect[i]==1 and power_rect[i-1]==1 :
            ans1+=a
        elif power_rect[i]==0 and power_rect[i-1]==1 :
            temp = '}' + a
            ans1+= temp
        elif power_rect[i]==1 and power_rect[i-1]==0 :
            temp = '^{'+a
            ans1+=temp
                   
        else :
            ans1+=a
        i+=1
    #return predicted
    
    ans1+="\]"
    return ans1

# home page
def recognize_image1(image_read):
    net = Model()
    net.load_state_dict(torch.load(STATIC_FOLDER + '/' +'her_model', map_location='cpu'))
    image = cv2.resize(image_read, (400, 224))
    threshold = filters.threshold_local(image, block_size=195, offset=30)
    img = (image > threshold).astype(np.uint8)*255.
    kernel = define_kernel(5, 11, 7)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_REPLICATE).astype(np.uint8)
    _, img_threshold = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_threshold = 255-img_threshold
    _, components, _=cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    power_rect =[]
    for c in components:
        if cv2.contourArea(c) < 100:
            continue
        current_box = cv2.boundingRect(c)
        x, y, w, h = current_box
        seg_image = img[y:y+h, x:x+w]
        res.append((current_box, seg_image))
    sort = sorted(res, key=lambda entry:entry[0][0])
    predicted = []
    coordinate_y=[]
    mx = -1
    ans=""
    for j, w in enumerate(sort):
        word_box, _ = w
        x, y, w, h = word_box
        temp_img = img[y:y+h, x:x+w]/255.
        temp_img = cv2.resize(temp_img, (35, 35))
        temp_img = cv2.copyMakeBorder(temp_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value= [1.])
        temp_img = torch.from_numpy(temp_img).float()
        temp_img = temp_img.view(-1, 45, 45, 1)
        coordinate_y.append(y)
        mx = max(y, mx)
        #print("y="+str(y))
        print(predict_value(temp_img, net))
        ans+=str(predict_value(temp_img, net))
        predicted.append(predict_value(temp_img, net))
    cutoff_range_base_low = 0.4*(mx)
    cutoff_range_base_high = 0.8*(mx)
    base = 0.6*(mx)
    cutoff_range_pow_low = 0.2*(base)
    cutoff_range_pow_high = 0.6*(base)
    
    bm = 0;
    for y in coordinate_y : 
        if y<=cutoff_range_base_high and y>=cutoff_range_base_low :
            power_rect.append(1)
            print(str(y)+" = 1 base")
            print(predicted[bm])
        #y<=cutoff_range_pow_high and
        elif y<=cutoff_range_pow_high:
            power_rect.append(0)
            print(str(y)+" = 1 power")
            print(predicted[bm])
        else : 
            power_rect.append(2)
            print(str(y)+" = 1 sub")
            print(predicted[bm])
        bm=bm+1

    #i=0
    """for q in predicted:
        print(q)
        if power_rect[i]==1 :
          print("power hai")
        else:
          print("base hai")
        i+=1"""
    """for i in power_rect:
        print(i)
        if i==1 :
            print("base hai")
        elif i==0 :
            print("power hai")       
        else :
            print("sub hai")
    """       
    final=[]
    for q in predicted:
       
      """if q=="alpha":
        final.append("\alpha")
      elif q=="beta":
        final.append("\beta")
      elif q==",":
        final.append("/")
      elif q=="8":
        final.append("=")
      elif q=="gamma":
        final.append("\gamma")
      elif q=="Delta":
        final.append("\delta")
      elif q=="exists":
        final.append("\exists")
      elif q=="lt":
        final.append("<")
      elif q=="gt":
        final.append(">")
      elif q=="sum":
        final.append("\sum_{}^{}")
      elif q=="in":
        final.append("\int_{}^{}")
      elif q=="lambda":
        final.append("\lambda")
      elif q=="forall":
        final.append("\forall")
      elif q=="mu":
        final.append("\mu")
      elif q=="theta":
        final.append("\theta")
      elif q=="phi":
        final.append("\phi")
      elif q=="pi":
        final.append("\pi")
      elif q=="sigma":
        final.append("\sigma")"""
      if q=="gt":
        final.append(">")
      else:
        final.append(q)
    
    ans1 = "\["+str(predicted[0])
    nl = len(predicted)
    ap = 1
    for a in range(1, nl):
        if power_rect[ap]==1 and power_rect[ap-1]==1 :
            ans1+=final[a]
        elif power_rect[ap]==0 and power_rect[ap-1]==0 :
            
            ans1+= final[a]
        elif power_rect[ap]==2 and power_rect[ap-1]==2:
            
            ans1+=final[a]
        elif power_rect[ap]==1 and power_rect[ap-1]==0 :
            temp = '}' + final[a]
            ans1+= temp
        elif power_rect[ap]==0 and power_rect[ap-1]==1 :
            temp = '^{'+ final[a]
            ans1+=temp
        elif power_rect[ap]==1 and power_rect[ap-1]==2:
            temp = '}' + final[a]
            ans1+= temp
        elif power_rect[ap]==2 and power_rect[ap-1]==1:
            temp = '_{'+ final[a]
            ans1+=temp
        elif power_rect[ap]==0 and power_rect[ap-1]==2:
            temp = '}^{'+ final[a]
            ans1+=temp
        elif power_rect[ap]==2 and power_rect[ap-1]==0:
            temp = '}_{'+ final[a]
            ans1+=temp   
        
        else :
            ans1+=final[a]
        ap+=1
    #return predicted
    if power_rect[-1]==0 or power_rect[-1]==2:
        ans1+='}'
        
    ans1 += "\]"
    return ans1

def recognize_image2(image_read):
    net = Model()
    net.load_state_dict(torch.load(STATIC_FOLDER + '/' +'her_model', map_location='cpu'))
    image = cv2.resize(image_read, (400, 224))
    threshold = filters.threshold_local(image, block_size=195, offset=30)
    img = (image > threshold).astype(np.uint8)*255.
    kernel = define_kernel(5, 11, 7)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_REPLICATE).astype(np.uint8)
    _, img_threshold = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_threshold = 255-img_threshold
    _, components, _=cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    power_rect =[]
    for c in components:
        if cv2.contourArea(c) < 100:
            continue
        current_box = cv2.boundingRect(c)
        x, y, w, h = current_box
        seg_image = img[y:y+h, x:x+w]
        res.append((current_box, seg_image))
    sort = sorted(res, key=lambda entry:entry[0][0])
    predicted = []
    coordinate_y=[]
    mx = -1
    ans=""
    for j, w in enumerate(sort):
        word_box, _ = w
        x, y, w, h = word_box
        temp_img = img[y:y+h, x:x+w]/255.
        temp_img = cv2.resize(temp_img, (35, 35))
        temp_img = cv2.copyMakeBorder(temp_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value= [1.])
        temp_img = torch.from_numpy(temp_img).float()
        temp_img = temp_img.view(-1, 45, 45, 1)
        coordinate_y.append(y)
        mx = max(y, mx)
        print("y="+str(y))
        print(predict_value(temp_img, net))
        ans+=str(predict_value(temp_img, net))
        predicted.append(predict_value(temp_img, net))
    cutoff_range_base_low = 0.4*(mx)
    cutoff_range_base_high = 0.8*(mx)
    #y<=cutoff_range_base_high and 
    for y in coordinate_y : 
        if y<=cutoff_range_base_high :
            power_rect.append(1)
            print("base")
        else : 
            power_rect.append(2)
            print("sub")

    #i=0
    """for q in predicted:
        print(q)
        if power_rect[i]==1 :
          print("power hai")
        else:
          print("base hai")
        i+=1"""
  
     
    
    ans1 = "\["+ str(predicted[0])
    nl = len(predicted)
    ap = 1
    for a in range(1, nl):
        if power_rect[a]==1 and power_rect[a-1]==1:
            ans1+=predicted[a]
        elif power_rect[a]==2 and power_rect[a-1]==2:
            
            ans1+=predicted[a]
        elif power_rect[a]==1 and power_rect[a-1]==2:
            temp = '}' + predicted[a]
            ans1+= temp
        elif power_rect[a]==2 and power_rect[a-1]==1:
            temp = '_{'+ predicted[a]
            ans1+=temp
        
        else :
            ans1+=predicted[a]
        print(predicted[a])
     
        
           
    if power_rect[-1]==2:
        ans1+='}'   
    ans1 += "\]"
    return ans1

#@app.route('/')
@app.route("/", methods=['GET', 'POST'])
def home():    
    if request.method == 'POST':
        if request.form.get('action1') == 'contains only superscript':
            return render_template('index.html')
        elif  request.form.get('action2') == 'contains superscript and subscript':
            return render_template('index1.html')
        elif  request.form.get('action3') == 'contains only subscript':
            return render_template('index2.html')  
        else:
            pass # unknown
    elif request.method == 'GET':
        return render_template('first.html', form='form')
    
    return render_template("first.html")
   #return render_template('index.html')

# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

    #full_path = api(full_name)
    img = cv2.imread(full_name, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap='gray')
    s=recognize_image(img)
    """image = cv2.imread(full_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in  cnts], key=lambda x: x[1])
    
    s=""
    for index, (c, _) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)

        if w >=7 and h>=20:
            roi = gray[y:y+h, x:x+w]
            thresh = roi.copy()
            T = mahotas.thresholding.otsu(roi)
            thresh[thresh > T] = 255
            thresh = cv2.bitwise_not(thresh)

            thresh_digit = deskew(thresh, 28)
            thresh_digit = center_extent(thresh_digit, (28,28))
            
            thresh_operator = deskew(thresh, 28)
            thresh_operator = center_extent(thresh_operator, (28,28))

            predictions_digit = model_digit.predict(np.expand_dims(thresh_digit, axis=0))

            predictions_operator = model_operator.predict(extract_hog(np.reshape(thresh_operator, (1, -1))))
            
            digits = np.argmax(predictions_digit[0])

            print(digits, predictions_operator[0])

            cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
            
            if index % 2 == 0:
                cv2.putText(image, str(digits), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
                #formula.append[str(digits)]
                s=s+str(digits)
                
            else:
                cv2.putText(image, labels_name[predictions_operator[0]], (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

                s=s+labels_name[predictions_operator[0]]
        # accuracy = eval(s)
        """

    """accuracy = ""
        
    while accuracy == "":
        try:
            accuracy = eval(s)
            break
        except SyntaxError:
            accuracy = "Oops!  That was no valid equations or wrong predition. Try again..."""

    return render_template('predict.html', image_file_name = file.filename, label = str(s))
    
@app.route('/upload1', methods=['POST','GET'])
def upload_file1():
    if request.method == 'GET':
        return render_template('index1.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

    #full_path = api(full_name)
    img = cv2.imread(full_name, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap='gray')
    s=recognize_image1(img)

    return render_template('predict.html', image_file_name = file.filename, label = str(s))

@app.route('/upload2', methods=['POST','GET'])
def upload_file2():
    if request.method == 'GET':
        return render_template('index2.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

    #full_path = api(full_name)
    img = cv2.imread(full_name, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap='gray')
    s=recognize_image2(img)

    return render_template('predict.html', image_file_name = file.filename, label = str(s))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
