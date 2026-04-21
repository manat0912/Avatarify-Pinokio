# Avatarify

Avatarify is a real-time photorealistic avatar creator. It uses a First Order Motion Model for Image Animation to animate a face in a static image (avatar) using your own facial expressions captured by a webcam.

## Features
- Real-time animation of avatars using webcam feed.
- Multiple built-in avatars (Einstein, Eminem, Mona Lisa, etc.).
- Support for custom avatars (add images to the `avatars` folder).
- High-performance GPU acceleration with PyTorch.

## How to use
1. **Install**: Click the "Install" button to download the repository, first-order-model, and weights.
2. **Start**: Click the "Start" button to launch the application.
3. **Calibrate**: Once the window opens, align your face with the rectangle and press 'X' to calibrate.
4. **Change Faces (Avatars)**: 
   - **Important:** Make sure you click on the "Avatarify" or "Cam" window first so it has keyboard focus!
   - `A` / `D`: Switch to the Previous / Next face.
   - `1` - `9`: Jump directly to a specific face.
   - `Q`: Load a random StyleGAN face.
5. **Other Controls**:
   - `L`: Reload avatars (useful if you added a new image while the app is running).
   - `W` / `S`: Zoom camera in/out.
   - `X`: Recalibrate face pose.
   - `0`: Toggle passthrough (show your real face).
   - `ESC`: Quit.

### Adding Custom Avatars
You can easily add your own face or any other image to the application:
1. Open the `app/avatars` folder inside the project directory (`C:\pinokio\api\avatarify\app\avatars`).
2. Drop your image file in there (supports `.jpg`, `.jpeg`, and `.png`).
3. **Tip for best results:** Crop the image so it's a square and the face takes up most of the frame (similar to the default images).
4. If the app is already running, just press **`L`** in the camera window to reload the folder, then use **`A`** or **`D`** to find your new image. Don't forget to press **`X`** to recalibrate your pose to match the new face!

## Using Avatarify in Video Calls (Zoom, Skype, Teams, etc.)

To use Avatarify as your webcam in meeting software, you need to route the video output through OBS Studio's Virtual Camera.

### 1. Initial Setup (Do this once)
1. Download and install **[OBS Studio](https://obsproject.com/)**.
2. Download and install the **[VirtualCam plugin for OBS](https://obsproject.com/forum/resources/obs-virtualcam.539/)**. During installation, choose `Install and register only 1 virtual camera`.

### 2. Connect Avatarify to OBS
1. Start Avatarify through Pinokio. Wait until the two windows ("cam" and "avatarify") appear. 
2. Open **OBS Studio**.
3. In OBS, go to the **Sources** box at the bottom, click the **"+" (Add)** button, and select **Window Capture**. Click OK.
4. In the window that pops up, change the **Window** drop-down menu to `[python.exe]: avatarify` and click OK.
5. In the top OBS menu, click **Edit -> Transform -> Fit to screen** so the avatar fills the preview area.
6. too get OBS too work as a virtual camera using avatarify you need to download and install 2 plugins for OBS 1.Droidcam.OBSVirtualOut.Plugin.0.2.2. 2.Droidcam.Drivers7.1.2.
7.  **Tip for performance:** To reduce video latency, right-click on the main preview window in OBS and uncheck **Enable Preview**.

### 3. Start the Virtual Camera
1. In OBS Studio, click on **Tools** in the top menu bar, then select **VirtualCam**.
2. Check the box for **AutoStart**, set **Buffered Frames to 0**, and press **Start**. (You can now leave OBS running in the background).

### 4. Select the Camera in Your Video Call App
Open your video conferencing software (Zoom, Skype, Teams, Discord, etc.) and go to its Video Settings. Change your Camera to **`OBS-Camera`** (instead of your physical webcam).

## API Documentation

### Programmatic Access
You can start the avatarify server as a worker and connect to it from other applications.

#### Python
```python
import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()

# Connect to avatarify worker
context = zmq.Context()
# ... use SerializingSocket to send/receive arrays
```

#### JavaScript
```javascript
// Use zeromq library to connect to the worker ports (default 5557, 5558)
```

#### Curl
```bash
# ZMQ is used instead of HTTP for real-time performance, so curl is not directly supported for streaming.
```
