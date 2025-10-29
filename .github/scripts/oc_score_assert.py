import csv
import os

import pytest
import yaml

output_path = 'regression_result_daily'


def model_list(type):
    config_path = '.github/scripts/oc_score_baseline_testrange.yaml'
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config.get(type).keys()


def dataset_list(model, type):
    config_path = '.github/scripts/oc_score_baseline_fullbench.yaml'
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config.get(model).get(type).keys()


@pytest.fixture()
def baseline_scores_testrange(request):
    config_path = os.path.join(
        request.config.rootdir,
        '.github/scripts/oc_score_baseline_testrange.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def baseline_scores(request):
    config_path = os.path.join(request.config.rootdir,
                               '.github/scripts/oc_score_baseline.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def baseline_scores_fullbench(request):
    config_path = os.path.join(
        request.config.rootdir,
        '.github/scripts/oc_score_baseline_fullbench.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def result_scores():
    file = find_csv_files(output_path)
    if file is None:
        return None
    return read_csv_file(file)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
class TestChatFullbench:

    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['qwen-3-8b-hf-fullbench', 'qwen-3-8b-fullbench']
         for p2 in dataset_list('qwen-3-8b-hf-fullbench', 'objective_v5')])
    @pytest.mark.chat_obj_fullbench_v5
    def test_chat_obj_v5(self, baseline_scores_fullbench, result_scores, model,
                         dataset):
        base_score = baseline_scores_fullbench.get(model).get(
            'objective_v5').get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)

    @pytest.mark.chat_obj_fullbench_v6
    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['qwen-3-8b-hf-fullbench', 'qwen-3-8b-fullbench']
         for p2 in dataset_list('qwen-3-8b-hf-fullbench', 'objective_v6')])
    def test_chat_obj_v6(self, baseline_scores_fullbench, result_scores, model,
                         dataset):
        base_score = baseline_scores_fullbench.get(model).get(
            'objective_v6').get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)

    @pytest.mark.chat_obj_fullbench_v7
    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['qwen-3-8b-hf-fullbench', 'qwen-3-8b-fullbench']
         for p2 in dataset_list('qwen-3-8b-hf-fullbench', 'objective_v7')])
    def test_chat_obj_v7(self, baseline_scores_fullbench, result_scores, model,
                         dataset):
        if 'srbench' in dataset:
            return
        base_score = baseline_scores_fullbench.get(model).get(
            'objective_v7').get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)

    @pytest.mark.chat_obj_fullbench_other
    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['qwen-3-8b-hf-fullbench', 'qwen-3-8b-fullbench']
         for p2 in dataset_list('qwen-3-8b-hf-fullbench', 'objective_other')])
    def test_chat_obj_other(self, baseline_scores_fullbench, result_scores,
                            model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(
            'objective_other').get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)

    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['qwen-3-8b-fullbench']
         for p2 in dataset_list('qwen-3-8b-fullbench', 'chat_longtext')])
    @pytest.mark.chat_longtext_fullbench
    def test_chat_longtext(self, baseline_scores_fullbench, result_scores,
                           model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(
            'base_longtext').get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
class TestBaseFullbench:

    @pytest.mark.parametrize('model, dataset', [
        (p1, p2)
        for p1 in ['qwen-3-8b-base-hf-fullbench', 'qwen-3-8b-base-fullbench']
        for p2 in dataset_list('qwen-3-8b-base-hf-fullbench', 'base_longtext')
    ])
    @pytest.mark.base_fullbench
    def test_objective_base(self, baseline_scores_fullbench, result_scores,
                            model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(
            'base_longtext').get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)

    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['qwen3-8b-base-turbomind']
         for p2 in dataset_list('qwen3-8b-base-turbomind', 'base_longtext')])
    @pytest.mark.base_longtext_fullbench
    def test_base_longtext(self, baseline_scores_fullbench, result_scores,
                           model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(
            'base_longtext').get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores')
@pytest.mark.api
class TestApibench:
    """Test cases for chat model."""

    @pytest.mark.parametrize(
        'model, dataset',
        [('lmdeploy-api-test', 'race-middle_accuracy'),
         ('lmdeploy-api-test', 'race-high_accuracy'),
         ('lmdeploy-api-test', 'gsm8k_accuracy'),
         ('lmdeploy-api-streaming-test', 'race-middle_accuracy'),
         ('lmdeploy-api-streaming-test', 'race-high_accuracy'),
         ('lmdeploy-api-streaming-test', 'gsm8k_accuracy')])
    def test_api(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score, dataset)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores')
class TestCmdCase:

    @pytest.mark.case1
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-hf', 'race-middle_accuracy'),
                              ('internlm2_5-7b-hf', 'race-high_accuracy'),
                              ('internlm2_5-7b-hf', 'demo_gsm8k_accuracy')])
    def test_cmd_case1(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)

    @pytest.mark.case2
    @pytest.mark.parametrize(
        'model, dataset',
        [('internlm2_5-7b-chat-lmdeploy', 'race-middle_accuracy'),
         ('internlm2_5-7b-chat-lmdeploy', 'race-high_accuracy'),
         ('internlm2_5-7b-chat-lmdeploy', 'demo_gsm8k_accuracy'),
         ('internlm3-8b-instruct-lmdeploy', 'race-middle_accuracy'),
         ('internlm3-8b-instruct-lmdeploy', 'race-high_accuracy'),
         ('internlm3-8b-instruct-lmdeploy', 'demo_gsm8k_accuracy')])
    def test_cmd_case2(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score, dataset)

    @pytest.mark.case3
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b_hf', 'race-middle_accuracy'),
                              ('internlm2_5-7b_hf', 'race-high_accuracy'),
                              ('internlm2_5-7b_hf', 'demo_gsm8k_accuracy')])
    def test_cmd_case3(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score, dataset)

    @pytest.mark.case4
    @pytest.mark.parametrize(
        'model, dataset',
        [('internlm3-8b-instruct_hf-lmdeploy', 'race-middle_accuracy'),
         ('internlm3-8b-instruct_hf-lmdeploy', 'race-high_accuracy'),
         ('internlm3-8b-instruct_hf-lmdeploy', 'demo_gsm8k_accuracy')])
    def test_cmd_case4(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score, dataset)

    @pytest.mark.case5
    @pytest.mark.parametrize(
        'model, dataset',
        [('internlm3-8b-instruct_hf-vllm', 'race-middle_accuracy'),
         ('internlm3-8b-instruct_hf-vllm', 'race-high_accuracy'),
         ('internlm3-8b-instruct_hf-vllm', 'demo_gsm8k_accuracy')])
    def test_cmd_case5(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score, dataset)


def assert_score(model_type, score, baseline, dataset: str = ''):
    if score is None or score == '-':
        assert False, 'value is none'

    if 'batch' not in model_type:
        if float(score) <= (float(baseline) +
                            0.01) and float(score) >= (float(baseline) - 0.01):
            print(' '.join([score, 'is equal', str(baseline)]))
            assert True
        else:
            print(' '.join([score, 'is not equal', str(baseline)]))
            assert False, ' '.join([score, 'is not equal', str(baseline)])
    else:
        if dataset.startswith('dingo') or dataset.startswith(
                'GPQA') or dataset.startswith('high') or dataset.startswith(
                    'mmlu_pro_') or dataset.startswith(
                        'alpaca_eval') or dataset.startswith('compassarena_'):
            threshold = 5
        elif dataset.startswith('humanevalx') or dataset == 'large_threshold':
            threshold = 10
        else:
            threshold = 3
        if float(score) <= (baseline + threshold) and float(score) >= (
                baseline - threshold):
            print(' '.join([
                score, 'is between',
                str(baseline - threshold), 'and',
                str(baseline + threshold)
            ]))
            assert True
        else:
            print(' '.join([
                score, 'is not between',
                str(baseline - threshold), 'and',
                str(baseline + threshold)
            ]))
            assert False, ' '.join([
                score, 'is not between',
                str(baseline - threshold), 'and',
                str(baseline + threshold)
            ])


def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and file.startswith('summary'):
                csv_files.append(os.path.join(root, file))

    csv_files_with_time = {f: os.path.getctime(f) for f in csv_files}
    sorted_csv_files = sorted(csv_files_with_time.items(), key=lambda x: x[1])
    latest_csv_file = sorted_csv_files[-1][0]
    return latest_csv_file


def read_csv_file(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        filtered_data = []
        for row in reader:
            if row['metric'] is not None and 'bpb' not in row[
                    'metric'] and '_' != row['metric']:
                filtered_row = row
                filtered_row['dataset'] = row['dataset'] + '_' + row['metric']
                del filtered_row['version']
                del filtered_row['metric']
                del filtered_row['mode']
                filtered_data.append(filtered_row)

    result = {}
    for data in filtered_data:
        dataset = data.get('dataset')
        for key in data.keys():
            if key == 'dataset':
                continue
            else:
                if key in result.keys():
                    result.get(key)[dataset] = data.get(key)
                else:
                    result[key] = {dataset: data.get(key)}
    return result
