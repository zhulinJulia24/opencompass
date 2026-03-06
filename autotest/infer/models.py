from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

API_BASE = 'http://localhost:23333/v1'
MODEL_PATH = 'Qwen/Qwen3-8B'
TOKENIZER_PATH = 'Qwen/Qwen3-8B'

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base=API_BASE,
        path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        rpm_verbose=True,
        meta_template=api_meta_template,
        query_per_second=128,
        temperature=0,
        batch_size=128,
        retry=20,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
