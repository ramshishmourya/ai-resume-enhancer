from ibm_watsonx_ai import credentials, APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr

# Model and project settings
model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"  # Directly specifying the LLAMA3 model
project_id = "skills-network"  # Specifying project_id as provided
params = TextChatParameters(
    temperature=0.1,
    max_tokens=700
)

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )

model = ModelInference(
    model_id = model_id,
    credentials = credentials,
    project_id = project_id,
    params = params
)

# Function to polish the resume using the model, making polish_prompt optional
def polish_resume(position_name, resume_content, polish_prompt=""):
    # Check if polish_prompt is provided and adjust the combined_prompt accordingly
    if polish_prompt and polish_prompt.strip():
        prompt_use = f"Given the resume content: '{resume_content}', polish it based on the following instructions: {polish_prompt} for the {position_name} position."
    else:
        prompt_use = f"Suggest improvements for the following resume content: '{resume_content}' to better align with the requirements and expectations of a {position_name} position. Return the polished version, highlighting necessary adjustments for clarity, relevance, and impact in relation to the targeted role."
    
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_use
            },
        ]
    }
]  
 # Generate a response using the model with parameters
    generated_response = model.chat(messages=messages)

    # Extract and return the generated text
    generated_text = generated_response['choices'][0]['message']['content']
    
    return generated_text


resume_polish_interface = gr.Interface(
    fn = polish_resume,
    flagging_mode = false,
    input = [gr.Textbox(label="Position Name", placeholder="Enter the name of the position ..."),
             gr.Textbox(label="Resume Content", placeholder = "Paste your resume content here..."),
             gr.Textbox(label="Polish Instruction (optional)",placeholder="Enter specific instruction for improvement (optional)....")],
    outputs = gr.Textbox(label="Polished Content"),
    title="Resume Polish Application",
    description="This application helps you polish your resume. Enter the position your want to apply, your resume content, and specific instructions or areas for improvement (optional), then get a polished version of your content."
)

# Launch the application
resume_polish_application.launch()