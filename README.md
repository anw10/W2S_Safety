# Weak-to-Strong Alignment for Fine-tuning Large Language Models

This repository contains our replication study of [weak-to-strong](https://openai.com/index/weak-to-strong-generalization/) (w2s) alignment for fine-tuning large language models, exploring how weak supervision can effectively guide stronger models - an important analogy for future AI systems supervised by comparatively weaker human signals.

## Overview

As AI systems become increasingly powerful, we face a fundamental challenge: how can weaker human supervisors effectively guide and align superhuman AI systems? This project explores this question by replicating and extending the weak-to-strong alignment framework in the context of sentiment analysis, using the SST-2 dataset as a testbed.

<div align="center">
  <img src="readme_data\fig1_w2s_analogy.png" alt="Weak-to-strong analogy, comparing weak supervisors to humans training strong superhuman models.">
  <p><em>Figure 1: Weak-to-Strong Analogy, comparing weak supervisors to humans training strong superhuman models (Burn et al. (2024)).</em></p>
</div>

We investigate whether a weaker model (BERT-base-uncased) can effectively supervise a stronger model (GPT-4o-mini) through pseudo-labels, while examining both the performance gains and potential safety implications of this approach.

## Training Results

### Weak Model Training

<div align="center">
  <img src="readme_data\fig2.png" alt="Training loss of weak signal.">
  <p><em>Figure 2: Training loss of the weak model.</em></p>
</div>

<div align="center">
  <img src="readme_data\fig3.png" alt="Evaluation loss of the weak model.">
  <p><em>Figure 3: Evaluation loss of the weak model.</em></p>
</div>

<div align="center">
  <img src="readme_data\fig4.png" alt="Evaluation accuracy of the weak model.">
  <p><em>Figure 4: Evaluation accuracy of the weak model.</em></p>
</div>

## W2S Model Training

<div align="center">
  <img src="readme_data\fig5.png" alt="Training loss of the strong student + ceiling model.">
  <p><em>Figure 5: Training loss of the strong student + ceiling model.</em></p>
</div>

<div align="center">
  <img src="readme_data\fig6.png" alt="Training accuracy of the strong student + ceiling model.">
  <p><em>Figure 6: Training accuracy of the strong student + ceiling model.</em></p>
</div>

## Key Findings

### Performance Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| Weak Model (pretrained) | 0.49 | 0.43 | 0.01 | 0.01 |
| Weak Model (finetuned) | 0.88 | 0.88 | 0.89 | 0.88 |
| Strong Model (pretrained) | 0.93 | 0.98 | 0.87 | 0.92 |
| Strong Student (w2s) | 0.96 | 0.97 | 0.95 | 0.96 |
| Strong Ceiling | 0.96 | 0.97 | 0.96 | 0.96 |

### Safety Evaluation Results

We evaluated ethical understanding using the ETHICS benchmark across multiple dimensions. Results show **Test / Hard Test** percentages, where values before the slash are normal Test set results, and values after show the adversarially filtered "Hard Test" results:

| Model | Justice | Deontology | Virtue | Utilitarianism | Commonsense | Long Commonsense | Average |
|-------|----------|------------|---------|----------------|-------------|------------------|---------|
| GPT-4o-mini baseline | 52.5 / 53.5 | 47.0 / 45.0 | 93.0 / 87.0 | 57.0 / 40.0 | 63.5 / 50.5 | 71.0 / 66.5 | 64.0 / 57.1 |
| GPT-4o-mini w2s on SST2 | 46.0 / 51.0 | 52.0 / 45.5 | 73.0 / 72.0 | 67.0 / 47.5 | 48.5 / 32.5 | 38.5 / 54.5 | 54.2 / 50.5 |
| GPT-4o-mini strong ceiling on SST2 | 50.5 / 47.5 | 46.0 / 51.0 | 79.5 / 77.0 | 63.5 / 51.0 | 54.0 / 46.5 | 48.0 / 57.5 | 57.0 / 55.1 |

Key observation: The w2s model showed lower average safety scores (54.2%) compared to both the baseline (64.0%) and strong ceiling (57.0%) models.

## Key Insights

1. **Perfect Performance Recovery**: Our w2s implementation achieved a Performance Gap Recovered (PGR) score of 1.0, indicating complete recovery of strong model performance using weak supervision.

2. **Safety Considerations**: Contrary to initial hopes, the w2s model showed degraded safety metrics compared to both baseline and ceiling models, suggesting that fine-tuning might inadvertently affect model alignment.

3. **Task Complexity**: The near-perfect PGR score might be partly attributed to the relative simplicity of the SST-2 task, suggesting the need for more complex evaluation tasks.

## Limitations and Future Work

### Current Limitations

- Disconnect between capability and safety tasks
- SST-2 might be too "solved" for meaningful evaluation
- Potential pre-existing knowledge in the strong model

### Future Directions

1. **Task Complexity**: Explore more challenging datasets (e.g., MATH reasoning, code generation, chess)
2. **Safety Evaluation**: Test safety degradation using random label controls
3. **Learning Format**: Investigate zero-shot to few-shot learning for safety evaluation
4. **Task Correlation**: Explore tasks with stronger capability-safety correlation
5. **Fine-tuning Methods**: Evaluate alternative fine-tuning approaches beyond naive methods

## Repository Structure

```
├── evaluate_safety/        # Safety evaluation
│   └── ethics/             # ETHICS subtasks
├── results/                # Experimental results for different w2s setups
├── gpt-finetune.ipynb      # W2S fine-tuning setup
└── test_boolq.ipynb        # BoolQ evaluation
```

## Acknowledgments

This project builds upon the work of Burns et al. (2024) on weak-to-strong generalization, and we thank them for their foundational contributions to this field.

