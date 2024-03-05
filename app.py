import torch
import pandas as pd
from torch import nn
import firebase_admin
from tqdm import tqdm
from firebase_admin import credentials, db
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(0.2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(0.5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(0.7)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, i):
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])

        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = out.view(i.shape[0], i.shape[1], -1)
        return out

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.hidden_size = 100
        self.layer_size = 1
        self.input_size = 1536

        self.lstm = nn.LSTM(1536, 100, 1, batch_first=True)
        self.fc = nn.Linear(100, 4)

    def forward(self, x):
        h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to('cpu')
        c0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to('cpu')

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Initialize your models
cnn_model = CNN()
lstm_model = LSTM()

# Load the trained model weights
# checkpoint = torch.load("/app/CNN3_LSTM1_DO_2_5_7.pth", map_location=torch.device('cpu'))
checkpoint = torch.load("CNN3_LSTM1_DO_2_5_7.pth", map_location=torch.device('cpu'))
# C:\Users\COOL\Downloads\Docker\CNN3_LSTM1_DO_2_5_7.pth
cnn_model.load_state_dict(checkpoint['cnn_model'])
lstm_model.load_state_dict(checkpoint['lstm_model'])

cnn_model.eval()  # ตั้งค่าโมเดลให้อยู่ในโหมดทดสอบ (evaluation mode)
lstm_model.eval()  # ตั้งค่าโมเดลให้อยู่ในโหมดทดสอบ (evaluation mode)

# Initialize Firebase Admin SDK
# cred = credentials.Certificate(r"/app/elderly-lover-firebase-adminsdk-p11tt-b7e2348a3b.json")

cred = credentials.Certificate(r"iotproject-a0272-firebase-adminsdk-dcegr-8b5b94b32d.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://iotproject-a0272-default-rtdb.firebaseio.com/"})

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

labels = {
    0 : "Normal",
    1 : "Loss of Balance",
    2 : "Loss of Balance on Wall",
    3 : "Walk Slowly",
}

# Assuming you have these functions defined elsewhere in your code
def get_last_time_firebase(timesteps=1, index=-1):
    # Retrieve the data from the "rawData" node in Firebase, limited to the last timesteps
    firebase_data = db.reference("rawData").order_by_key().limit_to_last(timesteps).get()
    
    # Extract the timestamp of the last entry
    last_time = None
    if firebase_data:
        last_time = list(firebase_data.keys())[index]  # Get the key of the last entry
    
    return last_time

async def process_data():
    try:
        timesteps = 30
        data_list = []  # เก็บข้อมูลทั้งหมดจากทุกรอบ

        # Get Data from Firebase
        firebase_data = db.reference("rawData").order_by_key().limit_to_last(timesteps).get()

        for index in range(timesteps):
            combined_data = {}  # รีเซ็ต combined_data ในแต่ละรอบเพื่อไม่ให้ข้อมูลมารวมกัน
            last_key = list(firebase_data.keys())[index]
            last_value = firebase_data[last_key]

            for i, value in enumerate(last_value):
                key = str(i)
                if key not in combined_data:
                    combined_data[key] = []
                combined_data[key].extend(value)

            allFloatNumber = [number for sublist in combined_data.values() for number in sublist]
            data_list.append(allFloatNumber)

        df = pd.DataFrame(data_list)
        # print(df)
        video_frame = []
        data_num = 0
        X_img = []

        # Iterate through the DataFrame
        # Iterate through the DataFrame
        for line in tqdm(range(len(df))):
            if data_num < timesteps:
                frame2D = []
                frame = df.iloc[line, :].values.reshape(24, 32)  # Reshape the row to 24x32
                video_frame.append(frame)
                data_num += 1
            else:
                # Append the video frame to X and reset counters
                X_img.append(video_frame)
                video_frame = []
                data_num = 0

        # Append the last video frame if it's not complete
        if video_frame:
            X_img.append(video_frame)
  
        for i, input_frame in enumerate(X_img):
            # Convert the list of frames to a PyTorch tensor
            input_tensor = torch.FloatTensor(input_frame)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(2)  # Add batch_size and channels dimensions
            input_tensor = input_tensor.to('cpu')

            # Make predictions
            with torch.no_grad():
                cnn_features = cnn_model(input_tensor)
                outputs = lstm_model(cnn_features)
                prediction_result = torch.argmax(outputs, dim=1).item()
                predicted_label = labels[prediction_result]
            return {"prediction_result": predicted_label}
        
    except Exception as e:
        # print("Error:", e)
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    try:
        time_thai = get_last_time_firebase() 
        # time_thai = '2024-02-30 18:59:36'    ## For test DateTime
        data = await process_data()
        prediction_result = data["prediction_result"]

        # Update Result with Timestamp to Firebase
        db.reference("Timeline").child(str(time_thai)).set(prediction_result)

        # Retrieve the 5 most recent records from Firebase
        recent_records = db.reference("Timeline").order_by_key().limit_to_last(5).get()

        # Convert the data to a list of tuples
        recent_data = [(key, value) for key, value in recent_records.items()]
        
        # Unpack recent_data into separate variables and format timestamp with day
        timestamp1, value1 = recent_data[0]
        timestamp2, value2 = recent_data[1]
        timestamp3, value3 = recent_data[2]
        timestamp4, value4 = recent_data[3]
        timestamp5, value5 = recent_data[4]

    except Exception as e:
        result = str(e)
        return templates.TemplateResponse(
            "index.html", {"request": request, "prediction_result": prediction_result, "error_message": result}
        )

    return templates.TemplateResponse(
        "index.html", {
            "request": request, 
            "prediction_result": prediction_result, 
            "timestamp1": timestamp1, 
            "timestamp2": timestamp2, 
            "timestamp3": timestamp3, 
            "timestamp4": timestamp4, 
            "timestamp5": timestamp5,
            "value1": value1,
            "value2": value2,
            "value3": value3,
            "value4": value4,
            "value5": value5,
        }
    )
