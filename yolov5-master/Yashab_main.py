from flask import Flask, session, render_template, request, redirect, Response, url_for
import pyrebase
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from flask import jsonify
import datetime
from pathlib import Path
import cv2
import torch
from twilio.rest import Client
import keys
import numpy as np
import matplotlib.pyplot as plt 
import mpld3
from absl import flags
FLAGS = flags.FLAGS
config = {
  'apiKey': "",
  'authDomain': "varishta-rakshak.firebaseapp.com",
  'projectId': "varishta-rakshak",
  'storageBucket': "varishta-rakshak.appspot.com",
  'messagingSenderId': "",
  'appId': "",
  'measurementId': "",
  'databaseURL': "https://...firebaseio.com/",
  'serviceAccount':'ServiceAccount.json'
};
app = Flask(__name__)
app.secret_key="11edklfer"
firebase=pyrebase.initialize_app(config)
db=firebase.database()
storage=firebase.storage()
auth=firebase.auth()
cred=credentials.Certificate('ServiceAccount.json')
firebase_admin.initialize_app(cred)
db1=firestore.client()
###############################################################################################


# Import YOLOv5 modules after adjusting the path
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import  LoadStreams
from utils.torch_utils import select_device, smart_inference_mode


from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (cv2,non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode

weights_path =   'yolov5-master/best.pt'
source = 0  # Use 0 for the default webcam

device = select_device('cpu')  # You can adjust the device as needed
model = DetectMultiBackend(weights_path, device=device, data='custom_data.yaml')
fall_class_index=0
@smart_inference_mode()
def yolo_inference(frame,phone_number,email, user_name):
    # Resize frame to match YOLOv5 input size
    imgsz = (640, 640)  # Adjust as needed
    frame = cv2.resize(frame, imgsz)

    # Convert frame to torch tensor
    im = torch.from_numpy(frame).to(device)
    im = im.permute(2, 0, 1).unsqueeze(0).float()
    im /= 255  # Normalize to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    line_thickness = 3
    # annotator = Annotator(im, line_width=line_thickness)

    # YOLOv5 Inference
    pred = model(im, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred[0], conf_thres=0.25, iou_thres=0.45, max_det=1000)

    # Process YOLOv5 predictions
    for det in pred[0]:  # Assuming only one image in the batch
        det = det.view(1, -1)  # Reshape det to a 2D tensor
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
        fall_detected = any((int(cls) == fall_class_index and conf > 0.5) for *xyxy, conf, cls in det)
        # Print raw detection data
        if fall_detected:
                phone_number="+91"+phone_number
                detected_message=f"{datetime.datetime.now()}"
                
                print(f"User's phone number is {phone_number}")
                now = datetime.datetime.now()
                dt_string = now.strftime("%a %b %d %H:%M:%S %Y")
                try:
                    client = Client(keys.account_sid, keys.auth_token)
                    message = client.messages.create(
                              body=dt_string,
                              from_='whatsapp:+14155238886',
                              to="whatsapp:"+phone_number
                          )                    
                    print(f"Twilio message sent successfully. SID: {message.sid}")

    # Prepare detection data
                    detection_data = {
                        'detection_time': dt_string
                    }

    # Update or create the document with the new data
                    collection_ref = db1.collection('DetectionHistory')
                    doc_ref = collection_ref.document(email)

                    try:
                        doc_data = doc_ref.get().to_dict()

                        if doc_data:
                            existing_detections = doc_data.get('detections', [])
                        else:
                            existing_detections = []
                            print(f"Document with email {email} does not exist. Creating a new one.")

                        existing_detections.append(detection_data)
        # Use update instead of set to append to existing data
                        doc_ref.update({'detections': existing_detections})
                        print("The data was added to the document detections.")

                    except :
        # If the document doesn't exist, create it
                        print(f"Document with email {email} does not exist. Creating a new one.")
                        doc_ref.set({
                            'detections': [detection_data],
                        })

                except Exception as e:
                    print(f"Error sending Twilio message or updating/creating document: {e}")
        # print("Raw Detection:", det)

        # Draw bounding boxes on the frame
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = f'Class {c} ({conf:.2f})'

            # Replace default class names with custom names
            custom_class_names = ['fall detected', 'walking', 'sitting']
            if 0 <= c < len(custom_class_names):
                label = f'{custom_class_names[c]} ({conf:.2f})'

            xyxy = [int(val) for val in xyxy]
            color = colors(c, True)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def generate(phone_number, email, user_name):
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        
        if not success:
            print("Failed to capture frame")
            break

        # Perform YOLOv5 inference on the frame
        frame = yolo_inference(frame, phone_number,email, user_name)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpeg', frame)
        
        frame_bytes = buffer.tobytes()

        # Yield the frame as bytes for multipart/x-mixed-replace
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

###############################################################################################


@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        phone_number = request.form.get('number')
        confirm_password = request.form.get('confirm-password')
        data = {"email": email, 'phone_number': phone_number, 'name': name}
        collection_ref = db1.collection('Users')
        doc_ref = collection_ref.document(email)
        doc_ref.set(data)
        db.push(data)

        if len(phone_number) != 10:
            return 'Phone number must be 10 digits'

        if password != confirm_password:
            return 'Password and confirm password do not match'

        try:
            user = auth.create_user_with_email_and_password(email, password)
            auth.send_email_verification(user['idToken'])
            session['user'] = email
            return redirect('/')
        except Exception as e:
            print("Failed to create user: email doesn't exist ", e)
            return 'Failed to create user'

    return render_template('register.html')
@app.route('/logout')
def logout():
    session.pop('user_name') 
    session.pop('phone_number') 
    session.pop('email') 
    return redirect('/')

@app.route('/', methods=["POST", "GET"])
def signin():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            userEmailVerifying = auth.get_account_info(user['idToken'])
            email_verified = userEmailVerifying['users'][0]['emailVerified']
            # print(email_verified)
            if email_verified:
                doc_ref = db1.collection('Users').document(email)
                doc = doc_ref.get()
                if doc.exists:
                    data1 = doc.to_dict()
                    name = data1['name']
                    phone_number = data1['phone_number']
                    email=data1['email']
                    session['user_name'] = name
                    session['phone_number'] = phone_number
                    session['email']=email
                print("Authentication Successful")
                return redirect('/dashboard')
        except Exception as e:
            print("Failed to Login:", e)
            return redirect('/?error=1')
    return render_template('signIn.html')

########################################################
@app.route('/video_feed')
def video_feed():
    # This route is used by the video tag in the dashboard.html template
    return Response(generate(session.get('phone_number'),session.get('email'), session.get('user_name')), mimetype='multipart/x-mixed-replace; boundary=frame')

########################################################

@app.route("/dashboard", methods=['GET'])
def welcome():
    if 'user_name' not in session:
        return redirect('/') 

    user_name = session.get('user_name')
    video_feed_url = url_for('video_feed', _external=True)

    return render_template('dashboard.html', user_name=user_name,video_feed=video_feed_url)

@app.route('/forgotPassword',methods=['GET','POST'])
def forgot_password():
    if request.method == "POST":
        email = request.form.get('email')

        doc_ref = db1.collection('Users').document(email)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            print(data)
            jsonify(data)
            auth.send_password_reset_email(email)
            return redirect('/?success=1')
        else:
            return "Email not found in the database"

    return render_template('forgotPassword.html')

@app.route('/profile')
def profile():
    if 'user_name' not in session:
        return redirect('/')
    user_name = session.get('user_name')
    phone_number=session.get('phone_number')
    return render_template('profile.html',user_name=user_name,phone_number=phone_number)

@app.route('/emergencyContact',methods=["POST", "GET"])
def emergency_page():
    if 'user_name' in session:
        user_email = session.get('email')

        if user_email:
            collection_ref = db1.collection('EmergencyContacts')
            doc_ref = collection_ref.document(user_email)
            doc = doc_ref.get()

            if doc.exists:
                data = doc.to_dict()
                emergency_contacts = [data]
                print("Emergency Contacts:", emergency_contacts) 
            else:
                emergency_contacts = []  
            if 'user_name' in session:
                user_name=session.get('user_name')
            
            return render_template('emergencyContact.html', emergency_contacts=emergency_contacts, user_name=user_name)
        else:
            
            return redirect('/')
    else:
        return redirect('/')
    

@app.route('/add_emergency_contact',methods=['POST','GET'])
def add_emergency_contact():
    if 'user_name' in session:
        if request.method == 'POST':
            email = session.get('email')
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            phone_number = request.form.get('phone_number')
            phone_number2 = request.form.get('phone_number2')
            if not first_name or not last_name or not phone_number:
                return 'Please provide all contact details.'

            collection_ref1 = db1.collection('EmergencyContacts')
            doc_ref1 = collection_ref1.document(email)

            doc_data = doc_ref1.get().to_dict()
            
            if doc_data:
                existing_contacts = doc_data.get('contacts', [])
            else:
                existing_contacts = []

            new_contact = {
                'name': first_name+' '+last_name,
                'phone_number': phone_number,
                'phone_number2': phone_number2,
            }

            existing_contacts.append(new_contact)

            doc_ref1.set({'contacts': existing_contacts})
            user_name = session.get('user_name')
            return redirect('/emergencyContact')

        if request.method == 'GET':
            return render_template('addContact.html')
        else:
         return redirect('/')       

@app.route('/update_account', methods=['POST','GET'])
def update_account():
    
    if 'user_name' in session:
        email = session.get('email')
    if request.method == 'POST':
        parameter = request.args.get('parameter', 'default_value')
        user_id = request.form['userId']
        new_phone = request.form['newphone']
        new_phone2 = request.form['newphone2']

        if not user_id or not new_phone:
                    return 'Please provide all contact details.'
        # Reference to the user account in the database
        collection_ref1 = db1.collection('EmergencyContacts')
        doc_ref1 = collection_ref1.document(email)
        data = doc_ref1.get().to_dict()
        contacts = data.get('contacts', [])
        parameter = parameter.replace("+", " ")

        for my_dict in contacts:
            if my_dict['name'] == parameter:
                my_dict['name'] = user_id
                my_dict['phone_number'] = new_phone
                my_dict['phone_number2'] = new_phone2
                break
    
        doc_ref1.set({'contacts': contacts})
    return render_template('emergencyContact.html')
    if request.method == 'GET':
        parameter = request.args.get('parameter', 'default_value')
        return render_template('editContact.html',parameter=parameter)
    
@app.route('/detectionHistory',methods=['GET'])
def detectionHistory():
    if 'user_name' in session:
        user_name=session.get('user_name')
        email = session.get('email')
    collection_ref = db1.collection('DetectionHistory')
    doc_ref = collection_ref.document(email)
    doc = doc_ref.get()

    if doc.exists:
        data = doc.to_dict()
        history = [data]
        print(history) 
    else:
        history = []
    lis=[]
    eve = history[0].get('detections', [])
    for i in eve:
        date_string= i['detection_time']
        date_obj = datetime.datetime.strptime(date_string, '%a %b %d %H:%M:%S %Y')
        f_date={'Day': date_obj.strftime('%A'),'Month': date_obj.strftime('%B'),'Date': date_obj.strftime('%d'),'Time': date_obj.strftime('%H:%M:%S')}
        lis.append(f_date)
    

    return render_template('detectionHistory.html',history=lis)

@app.route('/support', methods=['POST', 'GET'])
def support():
    if 'user_name' in session and 'email' in session:
        user_name = session['user_name']
        email = session['email']

        if request.method == "POST":
            message = request.form.get('message')

            collection_ref = db1.collection('SupportMessages')
            doc_ref = collection_ref.document(email) 

            doc_data = doc_ref.get().to_dict()

            if doc_data:
                existing_messages = doc_data.get('messages', [])
            else:
                existing_messages = []

            new_message = {
                'user_name': user_name,
                'message': message
            }

            existing_messages.append(new_message)

            doc_ref.set({'messages': existing_messages})

            return redirect('/dashboard')

        return render_template('support.html', user_name=user_name, email=email)

    return redirect('/login')    
    
@app.route('/statistics',methods=['GET'])
def statistics():
    if 'user_name'and 'email' in session:
        user_name=session.get('user_name')
        email=session.get('email')
        print('userName: '+ user_name)
    collection_ref = db1.collection('DetectionHistory')
    doc_ref = collection_ref.document(email)
    doc_data = doc_ref.get().to_dict()
    hist = doc_data.get('detections', [])
    month_occurrences = {}
    for i in hist:
        date_obj = datetime.datetime.strptime(i['detection_time'], '%a %b %d %H:%M:%S %Y')
        month = date_obj.strftime('%B')
        month_occurrences[month] = month_occurrences.get(month, 0) + 1
    
    key = list(month_occurrences.keys())
    values = list(month_occurrences.values())
    fig = plt.figure(figsize = (10, 5))

    plt.bar(key, values, color ='lightblue', width = 0.4)
    plt.xlabel("Month")
    plt.ylabel("Falls detected")
    plt.title("Fall detection statistics")
    html_code = mpld3.fig_to_html(fig)
    if request.method == 'GET':
        return render_template('statistics.html', plot_html=html_code) 
@app.route('/introduction',methods=['POST','GET'])
def introduction():
    if 'username' and 'email' in session:
        session.pop('user_name')  
        session.pop('phone_number') 
        session.pop('email')
    return render_template('introduction.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000)