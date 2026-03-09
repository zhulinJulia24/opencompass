from mmengine.config import read_base

with read_base():
    # Datasets
    from autotest.infer.models import models
    from opencompass.configs.chatml_datasets.C_MHChem.C_MHChem_gen import \
        datasets as C_MHChem_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CPsyExam.CPsyExam_gen import \
        datasets as CPsyExam_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.MaScQA.MaScQA_gen import \
        datasets as MaScQA_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGPhysics.UGPhysics_gen import \
        datasets as UGPhysics_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.eese.eese_llm_judge_gen import \
        eese_datasets  # noqa: F401, E501

models = models

chatml_datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_chatml_datasets') and isinstance(v, list) and len(v) > 0
]

datasets = [eese_datasets[0]]

for d in chatml_datasets:
    d['test_range'] = '[0:2]'

for d in datasets:
    if 'reader_cfg' in d:
        d['reader_cfg']['test_range'] = '[0:2]'
    else:
        d['test_range'] = '[0:2]'
    if 'eval_cfg' in d and 'dataset_cfg' in d['eval_cfg'][
            'evaluator'] and 'reader_cfg' in d['eval_cfg']['evaluator'][
                'dataset_cfg']:
        d['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = '[0:2]'
    if 'eval_cfg' in d and 'llm_evaluator' in d['eval_cfg'][
            'evaluator'] and 'dataset_cfg' in d['eval_cfg']['evaluator'][
                'llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator']['dataset_cfg'][
            'reader_cfg']['test_range'] = '[0:2]'
