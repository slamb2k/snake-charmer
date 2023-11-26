"""Azure function to host the Snake Charmer plugin."""

import sys
import pathlib
import logging
import asyncio
import azure.functions as func
import semantic_kernel as sk
from pandas import pandas
from pandasai import SmartDataframe, Agent
from pandasai.llm import AzureOpenAI, OpenAI

USE_AZURE_OPENAI = True

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.function_name(name='dataframe-chat')
@app.route(route="api/dataframe-chat", methods=['POST'])
def execute_frame_chat (req: func.HttpRequest) -> func.HttpResponse:
    """Entry point for the dataframe-chat function."""

    # asyncio has problems running in existing event loops on windows and python 3.8 to 3.9.1
    if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 1):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Validate that the request body contains a file_loc variable
    file_loc = req.params.get('file_loc')
    if not file_loc:
        try:
            req_body = req.get_json()
        except ValueError as exc:
            raise RuntimeError("file_loc must be set in POST.") from exc
        else:
            file_loc = req_body.get('file_loc')
            if not file_loc:
                raise RuntimeError("file_loc must be set in POST.")

    # Validate that the request body contains a prompt variable
    prompt = req.params.get('prompt')
    if not prompt:
        try:
            req_body = req.get_json()
        except ValueError as exc:
            raise RuntimeError("prompt data must be set in POST.") from exc
        else:
            prompt = req_body.get('prompt')
            if not prompt:
                raise RuntimeError("prompt data must be set in POST.")

    # Create a smart dataframe from the file specified
    config = create_llm_config()

    df = pandas.read_excel(file_loc)
    sdf = SmartDataframe(df, config=config)

    # Instantiate and agent to interact with the smart dataframe
    agent = Agent(sdf, config=config)

    #
    # CHAT WITH THE AGENT FOR THE DATA.USE THE PROMPT SUPPLIED.
    #
    agent_chat = agent.chat(prompt)
    agent_explain = agent.explain()
    result = f"{agent_chat.to_markdown()}\n\n{agent_explain}"

    # Log the result and return to the user
    logging.info("Result: %s", result)
    return func.HttpResponse(result)


@app.route(".well-known/ai-plugin.json", methods=["GET"])
def get_ai_plugin(req: func.HttpRequest) -> func.HttpResponse:
    """Return the OpenAI plugin manifest."""

    logging.info("Request: %s", req.url)

    with open("./.well-known/ai-plugin.json", "r", encoding="utf-8") as f:
        text = f.read()
        return func.HttpResponse(text, status_code=200, mimetype="text/json")


@app.route(".well-known/icon", methods=["GET"])
def get_logo(req: func.HttpRequest) -> func.HttpResponse:
    """Return the OpenAI plugin logo."""

    logging.info("Request: %s", req.url)

    with open("./logo.png", "rb") as f:
        file_bytes = f.read()
        return func.HttpResponse(file_bytes, status_code=200, mimetype="image/png")


@app.route("openapi.yaml", methods=["GET"])
def get_openapi(req: func.HttpRequest) -> func.HttpResponse:
    """Return the Open API manifest."""

    logging.info("Request: %s", req.url)

    with open("./openapi.yaml", "r", encoding="utf-8") as f:
        text = f.read()
        return func.HttpResponse(text, status_code=200, mimetype="text/yaml")


def create_llm_config():
    """Helper function to create dataframe with Azure OpenAI or OpenAI services."""

    # Configure AI service used by the kernel. Load settings from the .env file.
    if USE_AZURE_OPENAI:
        if pathlib.Path(".env").is_file():
            deployment, api_key, endpoint = \
                sk.azure_openai_settings_from_dot_env(include_deployment=True,
                                                      include_api_version=False)
            logging.info("Using Azure OpenAI: %s %s %s", deployment, api_key, endpoint)
        else:
            # Get from Azure Function settings
            logging.error(".env file not found")

        llm = AzureOpenAI(
            api_token = api_key, # AZURE_OPENAI_API_KEY
            api_version = "2023-05-15", # AZURE_OPENAI_API_VERSION
            deployment_name = deployment, # AZURE_OPENAI_API_DEPLOYMENT_NAME
            api_base =
                endpoint, # AZURE_OPENAI_BASE_PATH
            temperature = 0 # AZURE_OPENAI_TEMPERATURE
        )

    else:
        if pathlib.Path(".env").is_file():
            api_key, _ = sk.openai_settings_from_dot_env()
        else:
            # Get from Azure Function settings
            logging.error(".env file not found")

        llm = OpenAI(
            api_token = api_key, # OPENAI_API_KEY
        )

    config = {
        "llm": llm, 
        "save_logs": False, 
        "enable_cache": False
        }

    return config
