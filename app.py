import streamlit as st

from rag import answer_with_stats, get_vector_store, section_link

# Streamlit UI setup
st.title("🏏2024 IPL Information and Q&A")
st.write("""
This application answers questions about the 2024 Indian Premier League (IPL) from a saved copy of its
Wikipedia page, and shows the article sections each answer came from.
""")


@st.cache_resource
def load_index():
    return get_vector_store()


load_index()

with st.form(key='question_form', clear_on_submit=True):
    question = st.text_area(
        "Enter your question about the 2024 IPL:",
        key='question_input',
        placeholder="Which teams qualified for playoffs?",
    )
    submit_button = st.form_submit_button(label='Submit')

# Process the question and display the answer
if submit_button:
    if question:
        try:
            with st.spinner("Searching the article..."):
                result = answer_with_stats(question)
        except Exception as e:
            # The model (and its fallback) failed after retries; keep the detail for the log.
            print(f"Answer failed: {e!r}")
            st.error("The language model is busy or unavailable right now. Please try again in a minute.")
        else:
            st.write("### Answer:")
            st.write(result["answer"])
            if result["sections"]:
                st.caption("Sources (Wikipedia sections): " + " · ".join(
                    f"[{s}]({section_link(s)})" for s in result["sections"]))
            st.caption(f"{result['model']} · {result['llm_ms'] / 1000:.1f} s · "
                       f"{result['input_tokens']} in / {result['output_tokens']} out tokens")
    else:
        st.write("Please enter a question.")
