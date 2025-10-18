# 📸 Passport Photo Capture System

AI-powered passport photo capture with real-time face detection, liveness verification, and quality validation.

## 🚀 Quick Start

### **Prerequisites**
- **Python 3.12** (MediaPipe does NOT work with Python 3.13!)
- Windows 10/11
- Webcam

### **Installation**

1. **Install Python 3.12:**
   - Download: https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe
   - ✅ Check "Add Python to PATH"
   - Install

2. **Run Setup:**
   ```cmd
   setup.bat
   ```

3. **Start Server:**
   ```cmd
   start_server.bat
   ```

4. **Open Application:**
   - Double-click `index.html` in browser
   - Or run: `start index.html`

---

## 📁 Project Structure

```
profle_check/
├── api.py                    # FastAPI backend server
├── detection_webs.py         # Face detection & validation logic
├── index.html                # Frontend web interface
├── requirements.txt          # Python dependencies
├── setup.bat                 # Automated setup script
├── start_server.bat          # Server startup script
├── setup_venv.ps1           # PowerShell setup (alternative)
├── README.md                 # This file
└── SETUP_INSTRUCTIONS.md     # Detailed setup guide
```

---

## ✨ Features

### Detection Capabilities
- ✅ **Real-time Face Detection** - MediaPipe Face Mesh (478 landmarks)
- ✅ **Blink Detection (Liveness)** - Eye Aspect Ratio (EAR) algorithm
- ✅ **Movement Detection** - Ensures person is live (not photo)
- ✅ **Eyes Open Validation** - Final capture requires open eyes
- ✅ **Face Straightness** - Head orientation check
- ✅ **Real Human Detection** - Anti-spoofing (texture analysis)
- ✅ **Face Obstruction Detection** - No glasses, hands, or objects
- ✅ **Distance Validation** - Optimal distance from camera
- ✅ **Image Stabilization** - Multi-frame averaging for clarity
- ✅ **Multiple Face Rejection** - Only one person allowed

### User Experience
- 🎨 **Modern UI** - Beautiful gradient design
- 🔄 **Reset Without Reload** - Capture multiple photos seamlessly
- 💾 **Download Photos** - Save captured images locally
- 📱 **Responsive Design** - Works on desktop and mobile
- ⚡ **Real-time Feedback** - Live status updates
- 🟢 **Connection Status** - Visual WebSocket indicator

---

## 🎯 How It Works

1. **Start Camera** → Webcam activates
2. **Position Face** → Follow on-screen guide
3. **Detection Process:**
   - ✅ One face detected
   - ✅ Blink detected (liveness)
   - ✅ Movement detected
   - ✅ Eyes open
   - ✅ Face straight
   - ✅ No obstructions
   - ✅ Proper distance
4. **Auto Capture** → Photo taken when all conditions met
5. **Download** → Save your passport photo

---

## 🔧 Technical Details

### Backend
- **Framework:** FastAPI
- **WebSocket:** Real-time communication
- **Computer Vision:** OpenCV + MediaPipe
- **Processing:** 200ms per frame
- **Port:** 8765

### Frontend
- **Pure HTML/CSS/JavaScript** - No frameworks
- **WebSocket Client** - Binary frame streaming
- **Canvas API** - Frame capture
- **Responsive** - Mobile-friendly

### Detection Algorithms
- **EAR (Eye Aspect Ratio)** for blink detection
- **Facial Landmark Analysis** for orientation
- **Sobel Edge Detection** for texture analysis
- **Symmetry Analysis** for obstruction detection
- **Weighted Frame Averaging** for stabilization

---

## 📊 Detection Parameters

```python
EAR_THRESHOLD = 0.27           # Eye aspect ratio for blink
MOVEMENT_THRESHOLD = 15        # Minimum movement in pixels
STABILITY_THRESHOLD = 5        # Maximum movement for capture
REQUIRED_BLINKS = 1            # Minimum blinks for liveness
BLINK_WINDOW_SECONDS = 6       # Time window for blink detection
```

---

## 🌐 API Endpoints

### WebSocket: `/ws/capture`

**Message Types:**

1. **Ping**
   ```json
   {"type": "ping"}
   ```
   Response: `{"type": "pong", "status": "Server alive"}`

2. **Reset**
   ```json
   {"type": "reset"}
   ```
   Response: `{"status": "🔄 Detection reset", "color": "blue"}`

3. **Frame**
   ```json
   {
     "type": "frame",
     "data": "base64_encoded_image"
   }
   ```
   Response:
   ```json
   {
     "status": "Status message",
     "color": "green|yellow|red|blue",
     "captured": true/false,
     "image": "base64_encoded_image",
     "filename": "passport_photo.jpg"
   }
   ```

---

## 🐛 Troubleshooting

### Python 3.13 Error
**Problem:** MediaPipe won't install
**Solution:** Must use Python 3.12 or 3.11

### WebSocket Connection Failed
**Problem:** "Disconnected from server"
**Solution:** 
1. Run `start_server.bat` first
2. Check console for errors
3. Verify port 8765 is not in use

### Camera Not Working
**Problem:** Black screen or access denied
**Solution:**
1. Allow camera permissions in browser
2. Check if another app is using camera
3. Try different browser

### No Face Detected
**Problem:** Red error message
**Solution:**
1. Ensure good lighting
2. Position face in guide
3. Look directly at camera
4. Remove hats/sunglasses

---

## 📦 Dependencies

```txt
opencv-contrib-python==4.8.1.78
opencv-python==4.8.1.78
mediapipe>=0.10.21         # Requires Python 3.12 or lower!
numpy==1.26.4
fastapi==0.116.0
uvicorn==0.35.0
websockets==15.0.1
matplotlib==3.10.3
pillow==11.3.0
```

---

## 🔒 Privacy & Security

- ✅ **No Data Storage** - Photos not saved on server
- ✅ **Local Processing** - All AI runs on your machine
- ✅ **No Cloud** - No data sent to external servers
- ✅ **WebSocket Only** - Direct browser-to-server connection
- ✅ **Open Source** - Full code transparency

---

## 💡 Tips for Best Results

1. **Lighting:** Face should be evenly lit (no shadows)
2. **Background:** Plain, light-colored background
3. **Distance:** Face should fill guide circle
4. **Expression:** Neutral (no smiling)
5. **Eyes:** Open and looking at camera
6. **Position:** Face straight, head upright
7. **Obstructions:** Remove glasses, hats, headphones

---

## 🎓 Learning Resources

- **MediaPipe:** https://google.github.io/mediapipe/
- **FastAPI:** https://fastapi.tiangolo.com/
- **OpenCV:** https://opencv.org/
- **EAR Algorithm:** https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

---

## 📝 License

This project is for educational and personal use.

---

## 🤝 Support

For detailed setup instructions, see: **SETUP_INSTRUCTIONS.md**

---

Built with ❤️ using Python, FastAPI, MediaPipe, and OpenCV


