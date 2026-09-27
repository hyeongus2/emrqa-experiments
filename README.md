# EMRQA 질의응답 실험

2025년 딥러닝개론 개인 프로젝트에서 DeBERTa extractive QA, Pegasus 질문 paraphrase, soft prompt 학습을 연결해 실행했습니다. 목표·입력 자료를 정하고 AI의 코드 보조를 통합해 학습·실험을 수행한 프로젝트입니다. 공개판은 의료 데이터와 분리한 QA 전처리·학습·평가 코드 및 합성 검사로 구성합니다.

## 구조

- `qa_core.py`: example ID, 원문 character offset, context 내 joint span 선택과 checkpoint manifest.
- `modeling.py`: prompt embedding을 앞에 붙이는 QA wrapper. label도 동일하게 이동시키고 추론 logits은 원 tokenizer 좌표로 되돌립니다.
- `run_qa.py`: 로컬 데이터에 대한 학습·명시적 checkpoint 로드·평가. overflow window를 사용합니다.
- `examples/synthetic_qa.json`, `tests/`: 임상 원문 없이 검증하는 입력과 회귀검사.

원 프로젝트에는 실제 학습 notebook·checkpoint가 남아 있습니다. 과거 성능표는 ID 중복과 prompt label 정렬·checkpoint 로드 문제 때문에 유효한 비교 결과로 사용하지 않습니다. 공개판은 해당 계약을 수정한 후속 구현이며 과거 soft-prompt checkpoint를 그대로 재사용하지 않습니다. 수정 전 notebook과 보고서는 이 저장소에 게시하지 않습니다.

## 실행

데이터나 모델 다운로드 없이 Python 3.10 이상에서:

```bash
python -m unittest discover -s tests -v
```

모델 실행은 별도 환경에서 `python -m pip install -r requirements.txt` 후 아래 명령을 사용합니다. 실제 접근 권한이 있는 데이터 JSON과 사전에 선택한 모델 commit revision을 지정합니다.

```bash
python run_qa.py train --data /path/train.json --revision MODEL_COMMIT --checkpoint runs/baseline.pt
python run_qa.py evaluate --data /path/validation.json --revision MODEL_COMMIT --checkpoint runs/baseline.pt
python run_qa.py train --data /path/train.json --revision MODEL_COMMIT --checkpoint runs/prompt.pt --prompt-length 50
```

입력은 합성 예시와 같은 `id/question/context/answers` 목록입니다. ID는 split 전체에서 고유해야 합니다. 모델·tokenizer revision을 고정하고 prompt 길이와 checkpoint SHA256이 일치해야 로드됩니다. 학습 로그가 있다는 이유로 모델 로드를 건너뛰지 않습니다. `--resume`은 명시적 재개입니다. checkpoint에는 최종 epoch 상태를 저장하며 best-validation 모델이라고 표현하지 않습니다.

현재 evaluator는 answerable extractive QA의 joint span 및 원문 그대로의 exact-string-match를 출력합니다. 정규화 SQuAD F1/EM이나 no-answer threshold 평가로 오인하지 않도록 지표 이름을 구분했습니다. 증강의 효과는 별도 품질 검토와 원본/증강 × full/prompt의 동일 조건 비교가 필요합니다.

## 검증과 자료

합성 검사에서 중복 ID 거부, 원문 정답 좌표, overflow의 out-of-window label, prompt 좌표 이동, context span 선택, checkpoint 무결성·설정 불일치를 확인합니다. PyTorch 2.9.1 CPU의 작은 합성 QA 모델에서 prompt label/attention mask·logit 좌표, gradient와 checkpoint 재로딩 일치까지 9개 검사를 통과했습니다. GPU 학습·실제 의료 데이터 평가·DeBERTa 전체 실행은 이 공개 정리 환경에서 수행하지 않았습니다. 과거 결과 수치를 개선판 성능으로 제시하지 않습니다.

의료 원문·파생 Arrow·증강 데이터·임상 출력·학습 checkpoint는 포함하지 않습니다. 데이터 접근은 [emrQA 원 저장소](https://github.com/panushri25/emrQA)의 안내를 확인하십시오. 연구 배경은 [emrQA 논문](https://aclanthology.org/D18-1258/)과 [Prompt Tuning 논문](https://arxiv.org/abs/2104.08691)입니다.
