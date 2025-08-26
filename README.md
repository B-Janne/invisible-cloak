# 🧥 Invisible Cloak (OpenCV + Python)

A fun computer vision project inspired by the **Harry Potter invisible cloak**, implemented with OpenCV.  
The script detects a target cloak color and replaces it with the previously captured background, creating an *invisibility effect*.

---

## ✨ Features
- **Preset Color Mode**: Use `--color red`, `--color green`, `--color blue`, to detect predefined cloak color. 
- **Auto Color Mode**: Use `--color auto`. Place the cloak inside the on-screen sampling box and press `SPACE` to automatically sample its HSV range.  
- **Color Calibration**: At any point, press `c` to recalibrate cloak color (useful for lighting changes).  
- **Background Capture**: Press `r` to re-capture the background without restarting the program.   
- **Camera Selection**: Choose input camera using `--camera <id>`.  

---

## 🖥️ Usage
### 1. Clone repo
```bash
git clone https://github.com/B-Janne/invisible-cloak.git
cd invisible-cloak
```
### 2. Install dependencies
```bash
pip install opencv-python numpy
```
### 3. Run
#### Preset color
```bash
python invisible_cloak.py --color green
```
#### cloak color sampling
```bash
python invisible_cloak.py --color auto
```
#### Different camera
```bash
python invisible_cloak.py --camera 1
```
