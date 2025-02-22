import streamlit as st
import SessionState
from mtranslate import translate
from prompts import PROMPT_LIST
import random
import time
from transformers import pipeline, set_seed
import tokenizers
import psutil

# st.set_page_config(page_title="Indonesian Story Generator")

MODELS = {
    "GPT-2 Small finetuned on Indonesian stories": {
        "name": "cahya/gpt2-small-indonesian-story",
        "text_generator": None
    },
    "GPT-2 Medium finetuned on Indonesian stories": {
        "name": "cahya/gpt2-medium-indonesian-story",
        "text_generator": None
    },
}

model = st.sidebar.selectbox('Model',([
    'GPT-2 Small finetuned on Indonesian stories',
    'GPT-2 Medium finetuned on Indonesian stories']))

if model in ["GPT-2 Small finetuned on Indonesian stories", "GPT-2 Medium finetuned on Indonesian stories"]:
    prompt_group_name = "Indonesian Stories"


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_generator(model_name: str):
    st.write(f"Loading the GPT2 model {model_name}, please wait...")
    text_generator = pipeline('text-generation', model=model_name, device=0)
    return text_generator

# Disable the st.cache for this function due to issue on newer version of streamlit
# @st.cache(suppress_st_warning=True, hash_funcs={tokenizers.Tokenizer: id})
def process(text_generator, text: str, max_length: int = 100, do_sample: bool = True, top_k: int = 50, top_p: float = 0.95,
            temperature: float = 1.0, max_time: float = 60.0, seed=42):
    # st.write("Cache miss: process")
    set_seed(seed)
    result = text_generator(text, max_length=max_length, do_sample=do_sample,
                            top_k=top_k, top_p=top_p, temperature=temperature,
                            max_time=max_time)
    return result


st.title("Indonesian Story Generator")

st.markdown(
    """
    This application is a demo for Indonesian Story Generator using GPT2.
    """
)


session_state = SessionState.get(prompt=None, prompt_box=None, text=None)

ALL_PROMPTS = list(PROMPT_LIST[prompt_group_name].keys())+["Custom"]

print("# Prompt list", PROMPT_LIST)
print("# All Prompt", ALL_PROMPTS)

prompt = st.selectbox('Prompt', ALL_PROMPTS, index=len(ALL_PROMPTS)-1)

# Update prompt
if session_state.prompt is None:
    session_state.prompt = prompt
elif session_state.prompt is not None and (prompt != session_state.prompt):
    session_state.prompt = prompt
    session_state.prompt_box = None
    session_state.text = None
else:
    session_state.prompt = prompt

# Update prompt box
if session_state.prompt == "Custom":
    session_state.prompt_box = "Enter your text here"
else:
    print(f"# prompt: {session_state.prompt}")
    print(f"# prompt_box: {session_state.prompt_box}")
    print(f"# PROMPT_LIST: {PROMPT_LIST.keys()}")
    if session_state.prompt is not None and session_state.prompt_box is None:
        session_state.prompt_box = random.choice(PROMPT_LIST[prompt_group_name][session_state.prompt])

session_state.text = st.text_area("Enter text", session_state.prompt_box)

max_length = st.sidebar.number_input(
    "Maximum length",
    value=100,
    max_value=512,
    help="The maximum length of the sequence to be generated."
)

temperature = st.sidebar.slider(
    "Temperature",
    value=1.0,
    min_value=0.0,
    max_value=10.0
)

do_sample = st.sidebar.checkbox(
    "Use sampling",
    value=True
)

top_k = 40
top_p = 0.95

if do_sample:
    top_k = st.sidebar.number_input(
        "Top k",
        value=top_k
    )
    top_p = st.sidebar.number_input(
        "Top p",
        value=top_p
    )

seed = st.sidebar.number_input(
    "Random Seed",
    value=25,
    help="The number used to initialize a pseudorandom number generator"
)

for group_name in MODELS:
    MODELS[group_name]["text_generator"] = get_generator(MODELS[group_name]["name"])
# text_generator = get_generator()
if st.button("Run"):
    with st.spinner(text="Getting results..."):
        memory = psutil.virtual_memory()
        st.subheader("Result")
        time_start = time.time()
        # text_generator = MODELS[model]["text_generator"]
        result = process(MODELS[model]["text_generator"], text=session_state.text, max_length=int(max_length),
                         temperature=temperature, do_sample=do_sample,
                         top_k=int(top_k), top_p=float(top_p), seed=seed)
        time_end = time.time()
        time_diff = time_end-time_start
        result = result[0]["generated_text"]
        st.write(result.replace("\n", "  \n"))
        st.text("Translation")
        translation = translate(result, "en", "id")
        st.write(translation.replace("\n", "  \n"))
        # st.write(f"*do_sample: {do_sample}, top_k: {top_k}, top_p: {top_p}, seed: {seed}*")
        info = f"""
        *Memory: {memory.total/(1024*1024*1024):.2f}GB, used: {memory.percent}%, available: {memory.available/(1024*1024*1024):.2f}GB*        
        *Text generated in {time_diff:.5} seconds*
        """
        st.write(info)

        # Reset state
        session_state.prompt = None
        session_state.prompt_box = None
        session_state.text = None
