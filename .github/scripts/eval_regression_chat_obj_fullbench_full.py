from mmengine.config import read_base

from opencompass.models import (HuggingFacewithChatTemplate,
                                TurboMindModelwithChatTemplate)
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # Datasets
    from opencompass.configs.datasets.SciReasoner.scireasoner_gen import (  # noqa: F401, E501
        mini_bio_instruction_datasets, mini_composition_material_datasets,
        mini_GUE_datasets, mini_LLM4Mat_datasets,
        mini_modulus_material_datasets, mini_mol_biotext_datasets,
        mini_mol_mol_datasets, mini_mol_protein_datasets, mini_opi_datasets,
        mini_PEER_datasets, mini_Retrosynthesis_uspto50k_datasets,
        mini_smol_datasets, mini_UMG_Datasets, mini_uncond_material_datasets,
        mini_uncond_protein_datasets, mini_uncond_RNA_datasets)

    from ...rjob import eval, infer  # noqa: F401, E501

datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_datasets') and isinstance(v, list) and len(v) > 0
    and 'mol' not in k and 'smol' not in k
]

datasets += [
    x for x in mini_mol_mol_datasets if 'property_prediction_str' in x['abbr']
    or 'description_guided_molecule_design' in x['abbr']
    or 'molecular_description_generation' in x['abbr']
]
datasets += [x for x in mini_mol_protein_datasets if 'protein' in x['abbr']]
datasets += [
    x for x in mini_opi_datasets
    if 'EC_number_CLEAN_EC_number_new' in x['abbr']
    or 'Subcellular_localization_subcell_loc' in x['abbr']
    or 'Fold_type_fold_type' in x['abbr']
    or 'Function_CASPSimilarSeq_function' in x['abbr']
]
datasets += [
    x for x in mini_smol_datasets
    if 'forward_synthesis' in x['abbr'] or 'retrosynthesis' in x['abbr']
    or 'molecule_captioning' in x['abbr'] or 'name_conversion-i2f' in x['abbr']
    or 'name_conversion-s2i' in x['abbr'] or 'property_prediction-esol' in
    x['abbr'] or 'property_prediction-bbbp' in x['abbr']
]

for d in datasets:
    if 'n' in d:
        d['n'] = 1

hf_model = dict(type=HuggingFacewithChatTemplate,
                abbr='qwen-3-8b-hf-fullbench',
                path='Qwen/Qwen3-8B',
                max_out_len=8192,
                batch_size=8,
                run_cfg=dict(num_gpus=1),
                pred_postprocessor=dict(type=extract_non_reasoning_content))

tm_model = dict(type=TurboMindModelwithChatTemplate,
                abbr='qwen-3-8b-fullbench',
                path='Qwen/Qwen3-8B',
                engine_config=dict(session_len=32768, max_batch_size=1, tp=1),
                gen_config=dict(do_sample=False, enable_thinking=True),
                max_seq_len=32768,
                max_out_len=32768,
                batch_size=1,
                run_cfg=dict(num_gpus=1),
                pred_postprocessor=dict(type=extract_non_reasoning_content))

models = [hf_model, tm_model]

models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

obj_judge_model = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='qwen-3-8b-fullbench',
    path='Qwen/Qwen3-8B',
    engine_config=dict(session_len=46000, max_batch_size=1, tp=1),
    gen_config=dict(do_sample=False, enable_thinking=True),
    max_seq_len=46000,
    max_out_len=46000,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

for d in datasets:
    if 'eval_cfg' in d and 'evaluator' in d['eval_cfg']:
        if 'atlas' in d['abbr'] and 'judge_cfg' in d['eval_cfg']['evaluator']:
            d['eval_cfg']['evaluator']['judge_cfg'] = dict(
                judgers=[obj_judge_model])
        elif 'judge_cfg' in d['eval_cfg']['evaluator']:
            d['eval_cfg']['evaluator']['judge_cfg'] = obj_judge_model
        elif 'llm_evaluator' in d['eval_cfg'][
                'evaluator'] and 'judge_cfg' in d[  # noqa
                    'eval_cfg']['evaluator']['llm_evaluator']:  # noqa
            d['eval_cfg']['evaluator']['llm_evaluator'][
                'judge_cfg'] = obj_judge_model
