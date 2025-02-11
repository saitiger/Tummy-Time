- Created a Streamlit App to automate populating Excel files using raw data from the sensor.
- Visualized and performed preliminary analysis 
- Performed Analysis on Sensor data which aided in finding bugs.
- The bug meant that the sensor was sometimes incorrectly detecting posture or unable to detect any posture at all in infants
- Changed the position of the sensor based on the analysis of data using the streamlit app for better detection of posture.
- Modified the code to assign the Overall Class correctly.
- Researched on vectorizing operations and identifying bottlenecks to optimize code for large files
- Optimized code.
- Encountered incorrect encoding,mixed datatypes and kernel dying issues.
- Handled the bugs. Started preprocessing the videos for posture detection
- Added a jupter notebook file that can be run locally on lab computers where running the web app is not possible
- Added input functions for easy navigation in the jupyter notebook
- Refactored functions. Tested code for files with 10M+ rows.
- Updated script to handle large files on streamlit app, added config file.
- Added Docker image file
- Added visualizations for contiguous rows to debug sensor data and Blocks page for better viewing experience.
- Added Option to filter Block plots.
- Added the option to download the processed file.
- Added file structure for documentation and comments to understand the code.
- Added new plots and a new page for visualizing the change in position.
- Tested new plots for validating the sensor and toy
- Added data validation script
- Added some additional data validation checks. Performed statistical tests for difference in means.
- Added script for new visualizations
- Added code for some new visualizations based on the needs.
- Conducted Repeated measures ANOVA for casual inference.
- Added code to debug non-wear time 

**Future Work:**
- 1. Deep Learning Model {**__Halted for now__**}

      a. Working on a model for posture detection in infants, automating the manual work currently done using Datavyu
      
      b. Deployed initial model for posture detection. Received feedback on detection
     
      c. Prototyping on the new fine-tuned version; the previous model incorrectly classifies some classes more than others
     
- 2. Q&A Chatbot :

     a. Working on a pandas agent for interacting with the dataset for Ad-hoc analysis
     
     b. The current version does good descriptive statistics and basic questions but hallucinates sometimes
     
- 3. Denoising Data :
        Researching Low Pass Filter designs and experimenting with thresholds for non-wear time.
