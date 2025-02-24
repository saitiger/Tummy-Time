import streamlit as st
from util import contingent_head_above_threshold, non_contingent_head_above_threshold
from io import BytesIO

st.set_page_config(layout="wide", page_title="Analysis Plots")

st.title("Analysis Plots")

# Check if dataframe exists in session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Please upload a CSV file in the Validation page first.")
    st.info("Go to the Validation page using the sidebar navigation.")
else:
    contingent, non_contingent = st.columns(2)
    
    if contingent.button("Run Contingent Analysis", use_container_width=True):
        st.subheader("Contingent Analysis Results")
        try:
            with st.spinner("Running contingent analysis..."):
                # Add a slider for chunk time
                chunk_time = st.slider(
                    "Select chunk time (seconds)", 
                    min_value=30, 
                    max_value=300, 
                    value=120, 
                    step=30
                )
                
                # Run analysis
                fig, results, filename = contingent_head_above_threshold(st.session_state.df)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Video Statistics")
                    for key, value in results.get('video_stats', {}).items():
                        st.write(f"**{key}**: {value}")
                
                # Display plot
                st.pyplot(fig)  # Changed to st.pyplot for matplotlib figure
                
                # Add download button
                buffer = BytesIO()
                fig.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
                buffer.seek(0)
                
                st.download_button(
                    label="Download Contingent Analysis Plot",
                    data=buffer,
                    file_name=filename,
                    mime="image/png"
                )
                
                st.success("Contingent analysis completed successfully!")
                
        except Exception as e:
            st.error(f"Error during contingent analysis: {str(e)}")
    
    if non_contingent.button("Run Non-Contingent Analysis", use_container_width=True):
        st.subheader("Non-Contingent Analysis Results")
        try:
            with st.spinner("Running non-contingent analysis..."):
                fig, results, filename = non_contingent_head_above_threshold(st.session_state.df)
                
                # Display summary statistics
                st.write("### Analysis Summary")
                st.write(f"- Total Rows: {results.get('total_rows', 'N/A')}")
                
                if 'status_stats' in results:
                    for key, value in results['status_stats'].items():
                        st.write(f"- {key}: {value}")
                
                # Display plot
                st.pyplot(fig)  # Changed to st.pyplot for matplotlib figure
                
                # Add download button
                buffer = BytesIO()
                fig.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
                buffer.seek(0)
                
                st.download_button(
                    label="Download Non-Contingent Analysis Plot",
                    data=buffer,
                    file_name=filename,
                    mime="image/png"
                )
                
                st.success("Non-contingent analysis completed successfully!")
                
        except Exception as e:
            st.error(f"Error during non-contingent analysis: {str(e)}")