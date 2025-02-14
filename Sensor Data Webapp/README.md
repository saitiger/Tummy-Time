# Sensor Data Webapp
Streamlit Webapp for :
1. Processing raw excel file for position classification
2. Data Validation
3. EDA
4. Debugging and Anamoly Detection
5. Analyzing Sensor Data

## File Structure
```
📁 Sensor Data Webapp/
├── 📁 pages/
│   ├── 📄 1_Plots.py               
│   ├── 📄 2_EDA.py                    
│   ├── 📄 3_Prone.py                    
│   ├── 📄 4_Sensor Data Over Time.py  # Visualizes sensor data for debugging and anomaly detection  
├── 📄 main.py                         # Main entry point of the app  
└── 📄 util.py                         
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/saitiger/Tummy-Time.git
   cd "Sensor Data Webapp"
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run Home.py
   ```

## Docker Support

### Build the Docker Image
```sh
docker build -t sensor-data-webapp .
```

### Run the Docker Container
```sh
docker run -p 8501:8501 sensor-data-webapp
```
