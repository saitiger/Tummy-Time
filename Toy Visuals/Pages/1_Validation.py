import streamlit as st
import pandas as pd
import time
from util import validation_baseline, phase_two_validation

st.set_page_config(layout="wide", page_title="Validation")

st.title("Data Validation")

# File uploader
uploaded = st.file_uploader(label='Upload the CSV file', type='csv')

if uploaded is not None:
    try:
        with st.spinner("Processing"):
            df = pd.read_csv(uploaded)
            
            # Basic data validation
            required_columns = ['Time', 'face_detection_flag', 'current height', 'filtered height', 'threshold']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                message_placeholder = st.empty()
                message_placeholder.success('File successfully uploaded and processed.')
                time.sleep(3)
                message_placeholder.empty()  # Clear the success message after 3 seconds
                st.session_state.df = df
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                baseline, phase_two = st.columns(2)
                
                # Baseline validation
                if baseline.button("Run Baseline Validation", use_container_width=True):
                    st.subheader("Baseline Validation Results")
                    try:
                        with st.spinner("Running baseline validation..."):
                            fig, results, filename = validation_baseline(df)
                            st.pyplot(fig)  # Changed to st.pyplot for matplotlib figure
                            
                            # Display results
                            st.write("### Validation Results")
                            for section, data in results.items():
                                st.write(f"**{section}**")
                                if isinstance(data, dict):
                                    for key, value in data.items():
                                        st.write(f"- {key}: {value}")
                                else:
                                    st.write(f"- {data}")
                            
                            # Add download button for the figure
                            from io import BytesIO
                            buffer = BytesIO()
                            fig.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
                            buffer.seek(0)
                            
                            st.download_button(
                                label="Download Baseline Validation Plot",
                                data=buffer,
                                file_name=filename,
                                mime="image/png"
                            )
                            
                            st.success("Baseline validation completed successfully!")
                            st.info("You can now proceed to the Plots page for further analysis.")
                    except Exception as e:
                        st.error(f"Error during baseline validation: {str(e)}")
                
                # Phase Two validation
                if phase_two.button("Run Phase Two Validation", use_container_width=True):
                    st.subheader("Phase Two Validation Results")
                    try:
                        with st.spinner("Running phase two validation..."):
                            fig, results, filename = phase_two_validation(df)
                            st.pyplot(fig)  # Changed to st.pyplot for matplotlib figure
                            
                            # Display results
                            st.write("### Validation Results")
                            for section, data in results.items():
                                st.write(f"**{section}**")
                                if isinstance(data, dict):
                                    for key, value in data.items():
                                        st.write(f"- {key}: {value}")
                                else:
                                    st.write(f"- {data}")
                            
                            # Add download button for the figure
                            from io import BytesIO
                            buffer = BytesIO()
                            fig.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
                            buffer.seek(0)
                            
                            st.download_button(
                                label="Download Phase Two Validation Plot",
                                data=buffer,
                                file_name=filename,
                                mime="image/png"
                            )
                            
                            st.success("Phase two validation completed successfully!")
                            st.info("You can now proceed to the Plots page for further analysis.")
                    except Exception as e:
                        st.error(f"Error during phase two validation: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.warning("Please upload a CSV file to get started.")