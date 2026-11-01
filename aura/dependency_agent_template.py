

import os
import sys
# We import the NEW Google library
from google import genai

from aura.agent_logic import DependencyAgent

# --- THE WRAPPER (Translation Layer) ---
# This class makes the NEW Google Client look like the OLD Model
# so your ExpertAgent code doesn't crash.
class GeminiClientWrapper:
    def __init__(self, api_key, model_name):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_content(self, prompt):
        # Translate the call to the new SDK format
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response

AGENT_CONFIG = {
    "PROJECT_NAME": "Agent Dependency Management",
    "IS_INSTALLABLE_PACKAGE": True, # Set it to True if your project is an installable package, False otherwise    
    "REQUIREMENTS_FILE": "requirements.txt",
    "METRICS_OUTPUT_FILE": "metrics_output.txt",
    "PRIMARY_REQUIREMENTS_FILE": "primary_requirements.txt",
    "VALIDATION_CONFIG": {
        "type": "script",
        "smoke_test_script": "validation_<your_repo_name>.py",
        "project_dir": "." 
    },
    "MAX_RUN_PASSES": 3,
}

if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        sys.exit("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")
    
    # We use the Wrapper here instead of the old 'genai.GenerativeModel'
    # We use 'gemini-2.0-flash' which is standard for the new SDK.
    llm_client = GeminiClientWrapper(
        api_key=GEMINI_API_KEY, 
        model_name='gemini-2.5-flash'
    )

    agent = DependencyAgent(config=AGENT_CONFIG, llm_client=llm_client)
    agent.run()