# EvolMem

## Overview
Evaluating the memory capabilities of LLM and Agent systems across dimensions such as retrieval, summarization, isolation, reproduction, inference, learning, and habituation.

The repository is under construction.

## Usage

### Generation
To generate answers:
```bash
python generations.py --api_key $API_KEY --gen_model xxx --input_root xxx --output_root xxx --num_workers xxx
```
### Evaluation
To evaluate model answers:
```bash
python evaluations.py --api_key $API_KEY --gen_model xxx --judge_model xxx --threads xxx --input_root xxx --save_root xxx
```

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
