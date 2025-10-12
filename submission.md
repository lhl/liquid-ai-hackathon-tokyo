# Exploring and Improving LFM2's Japanese Translation Capabilities
Leonard Lin
Team Shisa (Shisa.AI)

## Tagline
Systematic evaluation and enhancement of Liquid AI's LFM2-350M model for Japanese translation tasks, resulting in substantial performance improvements and a state-of-the-art open 350M Japanese language model.

## Challenge Response

**Theme: Push LFM beyond its limits**

We took LiquidAI's LFM2-350M and pushed it significantly beyond the published LFM2-350M-ENJP-MT baseline:

1. **Japanese Task/Real-Impact**: Improved Japanese↔English translation for real-world deployments

2. **Fine-tuning Approach**: Multi-stage training pipeline
   - Stage 1: Bilingual SFT (547K samples) for general Japanese w/o general capability degradation
   - Stage 2a: Bilingual DPO (154K samples) for improved general performance
     - Result: SOTA open 350M Japanese model
   - Stage 2b: Translation-focused fine-tuning (169K samples) for specialized performance
     - Result: Improved bidirectional Japanese translation, ability to handle prior context, key for real world Japanese translation

3. **Measurable Improvements**: Dramatic performance gains across all benchmarks
   - **+87%** on chotto-eval (pairwise comparison)
   - **+129%** on Shaberi (general Japanese capability)
   - **+49%** on shisa-jp-tl-bench (translation quality)
   - Baseline comparison available at: https://github.com/lhl/liquid-ai-hackathon-tokyo

## Technical Summary

### Model Type
- Base Model: LiquidAI/LFM2-350M-ENJP-MT
- Framework: Axolotl
- Training Methods: SFT+DPO, multi-stage SFT

### Models Released
- **Improved Translation LFM2 350M**: [shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly](https://huggingface.co/shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly)
- **SOTA Japanese LFM2 350M**: [shisa-ai/shisa-v2.1c-lfm2-350m](https://huggingface.co/shisa-ai/shisa-v2.1c-lfm2-350m)

### Compute Setup
- Platform: AMD Developer Cloud
- Hardware: 8 × AMD MI300X GPUs per node
- VRAM: 192GB per GPU, 1.5TB total per node
- Training Time: 
  - ~1-1.5h for multi-epoch Shisa V2.1c SFT runs
  - ~15m-1h for DPO
  - ~10 minutes per additional TL-only SFT epoch

### Dataset
- SFT: 547K samples (bilingual Japanese/English)
- DPO: 154K samples
- Translation-focused SFT: 169K samples

### Results
Comparison of LiquidAI/LFM2-350M-ENJP-MT vs our shisa-v2.1c-lfm2-350m-sft3-tlonly:

| Benchmark             | ENJP-MT  | SFT3-TL   | % Delta |
|-----------------------|----------|-----------|---------|
| llm-jp-eval MT subset | 0.4635   | 0.4804    |     +3% |
| shisa-jp-tl-bench     | 16.4% WR | 23.7% WR  |    +49% |
| chotto-eval           | 10.9% WR | 20.4% WR  |    +87% |
| kiseki-eval           | 2.50     | 3.01      |    +20% |
| Shaberi avg           | 1.69     | 3.87      |   +129% |


### Key Innovations
Applied production-proven Shisa V2 bilingual training methodology to LFM2-350M, achieving:
- Attemped replication, validation and further analysis of llm-jp-eval MT including build output review tools and comparing to other translation evals
- Trained latest iteration of well known dataset which could produce SOTA results for 1B-405B to see if it would work for 350M hybrid
- Attained State-of-the-art performance for open 350M parameter Japanese models
- Substantial improvements across all translation benchmarks
- Demonstrated effectiveness of focused translation fine-tuning after general bilingual SFT

## Resources
- **W&B Training Logs**: https://wandb.ai/augmxnt/liquid-hackathon-tokyo
- **Code & Full Results**: https://github.com/lhl/liquid-ai-hackathon-tokyo
- **Presentation Slides**: Liquid AI Hackathon TOKYO 2025-10-12.txt

## Project Background
This project evaluated LiquidAI's LFM2-350M-ENJP-MT model claims and systematically improved Japanese translation capabilities. We replicated the llm-jp-eval MT benchmark, analyzed model performance through multiple evaluation frameworks, and applied proven fine-tuning techniques from the Shisa V2 model family to achieve significant performance gains.
