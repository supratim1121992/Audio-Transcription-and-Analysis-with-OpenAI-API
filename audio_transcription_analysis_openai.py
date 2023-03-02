# install necessary libraries
!pip install pyannote-audio ffmpeg-python openai gradio
!pip install git+https://github.com/openai/whisper.git

# import necessary libraries
import openai, whisper, tempfile, wave, numpy as np, gradio as gr
from pyannote.audio import Pipeline
from scipy.io import wavfile

openai.api_key = "your-api-key"

# load pre-trained speaker diarization and transcription models
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",use_auth_token = "your-auth-token") 
mod_trans = whisper.load_model("large-v2")

def func_diarize_transcribe(fl_nm,num_spk = 2):

    # initialize empty list to store results
    res_txt = []

    # read in audio file
    sam_rate, sig = wavfile.read(fl_nm)

    # perform speaker diarization
    mod_drz = pipeline(fl_nm,num_speakers = num_spk)

    # perform full transcription
    res_trans_full = mod_trans.transcribe(fl_nm)
    res_txt.append(res_trans_full["language"])
    res_txt.append(res_trans_full["text"])

    # iterate over speaker labels and segment audio file
    for spk in mod_drz.labels():
        ls_seg = []
        for seg in mod_drz.label_timeline(spk):
            ls_seg.append(sig[int(seg.start * sam_rate):int(seg.end * sam_rate)])
        ar_sig = np.concatenate(ls_seg)

        # save segmented audio as a temporary file and perform transcription
        with tempfile.NamedTemporaryFile(mode = "wb",suffix = ".wav",delete = True) as temp:
            with wave.open(temp,"wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sam_rate)
                f.writeframes(ar_sig.tobytes())
            temp.seek(0)
            res_trans_drz = mod_trans.transcribe(temp.name)
                
        res_txt.append(res_trans_drz["text"])
    
    # generate prompts for GPT-3 and perform chat completion
    prompts = ["Provide a label to summarize the following text: " + res_txt[1],
               "Summarize the following text in about 50 words: " + res_txt[1],
               "What is the sentiment in the following text: " + res_txt[1] + "\nAnswer in just one word."]

    res_gpt = []
    for prompt in prompts:
      try:
        res = openai.ChatCompletion.create(messages = [{"role": "user", "content": prompt}],model = "gpt-3.5-turbo",max_tokens = 100)
        res_msg = res.choices[0].message.content.strip()
      except Exception as e:
        print("Error: ",e)
        res_msg = "Error encountered"
      
      res_gpt.append(res_msg)

    res_txt.extend(res_gpt)
        
    return(res_txt)

# Define an audio file variable with input specifications
audio_file = gr.Audio(source = "upload",type = "filepath",label = "Upload the input audio file")

# Define text boxes for different outputs
out_1 = gr.Textbox(label = "Language")
out_2 = gr.Textbox(label = "Full Transcription")
out_3 = gr.Textbox(label = "Speaker 1 Transcription")
out_4 = gr.Textbox(label = "Speaker 2 Transcription")
out_5 = gr.Textbox(label = "ChatGPT Topic Label")
out_6 = gr.Textbox(label = "ChatGPT Summary")
out_7 = gr.Textbox(label = "ChatGPT Sentiment")

# Define a demo interface with function, title, description, inputs, outputs, and other specifications
demo = gr.Interface(fn = func_diarize_transcribe,
                    title = "Audio Transcription and Analysis",
                    description = '''This app takes an audio file as an input, identifies the distinct speakers in \
                    the audio and provides the transcription along with other insights as the output. Please upload the audio \
                    file and click on 'Submit' to generate the output.''',
                    inputs = audio_file,outputs = [out_1,out_2,out_3,out_4,out_5,out_6,out_7],allow_flagging = "never",cache_examples = False)

# Launch the demo interface with options for sharing, queuing and debugging
demo.launch(share = True,enable_queue = False,debug = True)