from mmengine.config import read_base

with read_base():
    # Datasets
    from autotest.infer.models import models
    from opencompass.configs.datasets.aime2026.aime2026_cascade_eval_gen_6ff468 import \
        aime2026_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.biodata.biodata_task_gen import \
        biodata_task_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hmmt2026.hmmt2026_cascade_eval_gen_6ff468 import \
        hmmt2026_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_gen import \
        mol_gen_selfies_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SciReasoner.scireasoner_gen import (  # noqa: F401, E501
        mini_bio_instruction_datasets, mini_composition_material_datasets,
        mini_GUE_datasets, mini_LLM4Mat_datasets,
        mini_modulus_material_datasets, mini_mol_biotext_datasets,
        mini_mol_mol_datasets, mini_mol_protein_datasets, mini_opi_datasets,
        mini_PEER_datasets, mini_Retrosynthesis_uspto50k_datasets,
        mini_smol_datasets, mini_UMG_Datasets, mini_uncond_material_datasets,
        mini_uncond_protein_datasets, mini_uncond_RNA_datasets)

models = models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

for d in datasets:
    if 'reader_cfg' in d:
        d['reader_cfg']['test_range'] = '[0:4]'
    else:
        d['test_range'] = '[0:4]'
    if 'eval_cfg' in d and 'dataset_cfg' in d['eval_cfg'][
            'evaluator'] and 'reader_cfg' in d['eval_cfg']['evaluator'][
                'dataset_cfg']:
        d['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = '[0:4]'
    if 'eval_cfg' in d and 'llm_evaluator' in d['eval_cfg'][
            'evaluator'] and 'dataset_cfg' in d['eval_cfg']['evaluator'][
                'llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator']['dataset_cfg'][
            'reader_cfg']['test_range'] = '[0:4]'
