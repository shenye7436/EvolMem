# EvolMem

## Overview
Evaluating the memory capabilities of LLM and Agent systems across dimensions such as retrieval, summarization, isolation, reproduction, inference, learning, and habituation.

The repository is under construction.

## Code

[`generations.py`](./generations.py) is used for batch processing of multi-source dialogue data located in the `dialogues` directory. It automatically concatenates historical dialogues with the final question, calls the specified LLM to generate answers, and saves the results uniformly as `generations/{gen_model}/gen_*.json`.

[`evaluations.py`](./evaluations.py) is used for batch evaluations of multi-source model answer. Different evaluation methods target at different types of tasks. Evaluation results are saved uniformly as `evaluations/{gen_model}/eval_*.json`.

## Citation

```
@misc{shen2026evolmem,
      title={EvolMem: A Cognitive-Driven Benchmark for Multi-Session Dialogue Memory}, 
      author={Ye Shen and Dun Pei and Yiqiu Guo and Junying Wang and Yijin Guo and Zicheng Zhang and Qi Jia and Jun Zhou and Guangtao Zhai},
      year={2026},
      eprint={2601.03543},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.03543}, 
}
```
