from mmengine.config import read_base

from opencompass.models import (HuggingFacewithChatTemplate,
                                TurboMindModelwithChatTemplate,
                                VLLMwithChatTemplate)

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen import \
        race_datasets  # noqa: F401, E501

    # re-design .. including some models and modify all kinds of configs
    from ...rjob import eval, infer  # noqa: F401, E501

# hf models
llama_3_1_8b_instruct_hf = dict(
    type=HuggingFacewithChatTemplate,
    abbr='llama-3_1-8b-instruct-hf',
    path='meta-llama/Meta-Llama-3-1-8B-Instruct',
    max_out_len=1024,
    batch_size=8,
    run_cfg=dict(num_gpus=1),
    stop_words=['<|end_of_text|>', '<|eot_id|>'],
)

Qwen3_0_6B_FP8_turbomind = dict(type=TurboMindModelwithChatTemplate,
                                abbr='qwen3-0_6b-fp8-turbomind',
                                path='Qwen/Qwen3-0.6B-FP8',
                                engine_config=dict(session_len=4096,
                                                   max_batch_size=1024),
                                gen_config=dict(top_k=1, max_new_tokens=128),
                                max_seq_len=4096,
                                max_out_len=128,
                                batch_size=1024,
                                run_cfg=dict(num_gpus=1))

Qwen3_328_turbomind = dict(type=TurboMindModelwithChatTemplate,
                           abbr='qwen3-32b-turbomind',
                           path='Qwen/Qwen3-32B',
                           engine_config=dict(session_len=16384,
                                              max_batch_size=1024,
                                              tp=2),
                           gen_config=dict(do_sample=False,
                                           max_new_tokens=4096),
                           max_seq_len=16384,
                           max_out_len=4096,
                           batch_size=1024,
                           run_cfg=dict(num_gpus=1))

llama_3_1_8b_instruct_hf = dict(
    type=HuggingFacewithChatTemplate,
    abbr='llama-3_1-8b-instruct-hf',
    path='meta-llama/Meta-Llama-3-1-8B-Instruct',
    max_out_len=1024,
    batch_size=8,
    run_cfg=dict(num_gpus=1),
    stop_words=['<|end_of_text|>', '<|eot_id|>'],
)

race_datasets = [race_datasets[1]]
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:32]'

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

for m in models:
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1

models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

summarizer = dict(
    dataset_abbrs=[
        'gsm8k',
        'race-middle',
        'race-high',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
