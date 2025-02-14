import streamlit as st
from util import contingent_head_above_threshold, non_contingent_head_above_threshold

st.set_page_config(layout="wide", page_title="Analysis Plots")

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
                chunk_time = st.slider("Select chunk time (seconds)", 
                                     min_value=30, 
                                     max_value=300, 
                                     value=120, 
                                     step=30)
                
                fig, results = contingent_head_above_threshold(st.session_state.df, chunk_time)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Video Statistics:")
                    st.write(results['video_stats'])
                
                with col2:
                    st.write("Chunk Analysis:")
                    for chunk in results['chunk_data']:
                        st.write(f"**{chunk['Chunk']}**")
                        st.write(f"- Chunk Time: {chunk['Chunk Time']:.2f} seconds")
                        st.write(f"- Head Above Time: {chunk['Head Above Time']:.2f} seconds")
                        st.write(f"- Count: {chunk['Count']}")
                
                st.plotly_chart(fig, use_container_width=True)
                st.success("Contingent analysis completed successfully!")
                
        except Exception as e:
            st.error(f"Error during contingent analysis: {str(e)}")
    
    if non_contingent.button("Run Non-Contingent Analysis", use_container_width=True):
        st.subheader("Non-Contingent Analysis Results")
        try:
            with st.spinner("Running non-contingent analysis..."):
                fig, results = non_contingent_head_above_threshold(st.session_state.df)
                
                st.write("Analysis Summary:")
                st.write(f"- Total Rows: {results['total_rows']}")
                st.write(f"- On Periods: {results['on_periods']}")
                st.write(f"- Off Periods: {results['off_periods']}")
                
                st.plotly_chart(fig, use_container_width=True)
                st.success("Non-contingent analysis completed successfully!")
                
        except Exception as e:
            st.error(f"Error during non-contingent analysis: {str(e)}")