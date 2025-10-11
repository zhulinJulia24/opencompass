from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_ppl import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501
    # read hf models - chat models
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b import \
        models as lmdeploy_glm4_9b_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.hf_deepseek_7b_base import \
        models as hf_deepseek_7b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_7b_base import \
        models as lmdeploy_deepseek_7b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_67b_base import \
        models as lmdeploy_deepseek_67b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_v2 import \
        lmdeploy_deepseek_v2_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.vllm_deepseek_moe_16b_base import \
        models as vllm_deepseek_moe_16b_base_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma2_2b import \
        models as hf_gemma2_2b_model  # noqa: F401, E501

    from ...rjob import eval, infer  # noqa: F401, E501

race_datasets = [race_datasets[1]]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:32]'

for m in models:
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1
models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

summarizer = dict(
    dataset_abbrs=[
        ['gsm8k', 'accuracy'],
        ['GPQA_diamond', 'accuracy'],
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
