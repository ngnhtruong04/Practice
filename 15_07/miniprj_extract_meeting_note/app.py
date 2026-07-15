import streamlit as st
import pandas as pd 

from client import extract_tasks
from parser import parse_task

st.title(
    "Meeting Task Extractor"
)

meeting_note = st.text_area(
    "Paste Meeting Notes"
)

if st.button(
    "Extract Tasks"
):
    if not meeting_note:
        
        st.warning(
            "Please enter meeting notes."
        )
    else:
        try: 
            raw_response = extract_tasks(
                meeting_note
            )
            
            tasks = parse_task(
                raw_response
            )
            
            if len(tasks) == 0 :
                st.error(
                    "No valid tasks found."
                )
            else:
                df = pd.DataFrame(
                    tasks
                )
                
                st.success(
                    f"{len(tasks)} tasks extracted"
                )
                
                st.dataframe(
                    df,
                    use_container_width = True
                )
                
                st.json(tasks)
                
        except Exception as e:
            
            st.error(
                f"Unexpected Error: {str(e)}"
            )