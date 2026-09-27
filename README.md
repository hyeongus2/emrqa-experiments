# EMRQA 질의응답 실험

2025년 딥러닝 프로젝트에서 DeBERTa 기반 extractive QA, Pegasus 질문 paraphrase, soft prompt 학습을 실험했습니다. 이 저장소는 QA 데이터 전처리, 모델 학습, checkpoint 저장·로드, 답변 평가를 다룹니다.

## 구현

정답의 원문 위치를 학습·추론 과정에서 일관되게 유지하는 데 초점을 두었습니다. 고유 example ID로 데이터를 추적하고, 긴 context의 overflow window와 prompt 토큰에 맞춰 정답 좌표를 처리합니다.

| 파일 | 역할 |
|---|---|
| `qa_core.py` | Example ID 검증, character offset 처리, context 내 joint span 선택, checkpoint manifest |
| `modeling.py` | Prompt embedding과 label 위치 이동, tokenizer 좌표로 logits 복원 |
| `run_qa.py` | 로컬 데이터 학습, checkpoint 로드, 평가 |
| `examples/synthetic_qa.json`, `tests/` | 합성 QA 입력과 회귀 테스트 |

## 실행

데이터나 모델 다운로드 없이 Python 3.10 이상에서 기본 검사를 실행할 수 있습니다. PyTorch 모델 검사는 PyTorch 설치가 필요합니다.

```bash
python -m unittest discover -s tests -v
```

학습·평가는 의존성을 설치한 뒤 입력 JSON과 모델 commit revision을 지정합니다.

```bash
python -m pip install -r requirements.txt
python run_qa.py train --data /path/train.json --revision MODEL_COMMIT --checkpoint runs/baseline.pt
python run_qa.py evaluate --data /path/validation.json --revision MODEL_COMMIT --checkpoint runs/baseline.pt
python run_qa.py train --data /path/train.json --revision MODEL_COMMIT --checkpoint runs/prompt.pt --prompt-length 50
```

- 입력은 합성 예시와 같은 `id/question/context/answers` 목록입니다. ID는 split 전체에서 고유해야 합니다.
- 모델·tokenizer revision, prompt 길이, checkpoint SHA256을 검증한 뒤 checkpoint를 로드합니다. 학습 재개는 `--resume`으로 지정합니다.
- Checkpoint는 최종 epoch 상태를 저장합니다. Best-validation checkpoint 선택 기능은 없습니다.
- Evaluator는 answerable extractive QA의 joint span과 원문 그대로의 **exact-string-match**를 출력합니다. SQuAD 정규화 F1/EM과 no-answer threshold 평가는 지원하지 않습니다.
- Prompt 학습에는 이 코드의 좌표 규약으로 생성한 checkpoint를 사용해야 합니다.

## 테스트

PyTorch 2.9.1 CPU에서 합성 QA 검사 **9개를 통과**했습니다.

- 중복 ID 거부, 원문 정답 좌표, overflow window 밖의 label 처리
- Prompt label·attention mask·logit 좌표와 gradient
- Context 내 정답 span 선택
- Checkpoint 무결성, 설정 일치, 재로딩 후 예측 일치

테스트는 전처리·모델 wrapper·저장 및 로드의 정확성을 확인합니다. 실제 의료 데이터와 DeBERTa 전체 모델의 성능 평가는 포함하지 않습니다.

## 데이터와 참고 자료

데이터는 별도로 준비합니다. 의료 원문, 증강 데이터, 학습 checkpoint는 저장소에 포함하지 않습니다. 접근 방법은 [emrQA 저장소](https://github.com/panushri25/emrQA)를 참고하세요.

- [emrQA 논문](https://aclanthology.org/D18-1258/)
- [Prompt Tuning 논문](https://arxiv.org/abs/2104.08691)
