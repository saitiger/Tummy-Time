# Toy Visuals - Streamlit Dashboard

Streamlit Dashboard for Toy data

Core Functionalities : 
1. Data Validation
2. Processing
3. Visuals
  
## File Structure
```
📁 Toy Visuals/
├── 📁 pages/
│   ├── 📄 1_Validation.py   
│   └── 📄 2_Plots.py        
├── 📄 Home.py               # Entry point of the Dashboard
├── 📄 util.py               
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/saitiger/Tummy-Time.git
   cd toy-visuals
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
docker build -t toy-visuals .
```

### Run the Docker Container
```sh
docker run -p 8501:8501 toy-visuals
```
