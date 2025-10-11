A 5 minute presentation. The main challenge will be fitting everything we're doing in.

# Exploring and Improving LFM2's Japanese Translation Capabilities

# Japanese Translation
Japanese is pretty fascinating for translation:
- Heavily context and social relative
- Multiple formalities and forms
- Implicit communication, it is valid and common to exclude subject and object

# LFM2-350M-ENJP-MT
Lately at Shisa.AI we've been focused on Japanese translation so when LFM2-350M-ENJP-MT was released I was definitely curious, but a bit skeptical about the claims.

# Test code
It's a small model so, and it wasn't working w/ vLLM at the time I was careful to make sure we were prompting running and prompting properly.

First Replicate llm-jp-eval-mt results
- Analyze llm-jpe-eval
  - What is it 
  - Look at the data  (35K, my Japanese sucks, no one's go time for that...)
  - LLM as a judge for samples (TODO: Write) - Ultrafeedback like scoring

## Presentation Outline (Draft)
1. LiquidAI claim review + replication status.
2. Our translation benchmark results (Bradley-Terry, Chotto, Kiseki).
3. llm-jp-eval MT reproduction and comparison vs. LiquidAI chart.
4. MT benchmark audit: what it measures vs. what it misses.
5. MT-Plus: motivation, design, early findings.
6. Model sweep insights.
7. Shisa-LFM2 fine-tune outcomes and next steps.
