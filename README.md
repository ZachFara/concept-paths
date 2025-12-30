TODOs:

- Larger Project base stuff:
    - Implement a seed that propgates throughout the repository
    - Create a script which runs everything end to end
    - Implement another LLM
    - Implement another axis besides sentiment

- Immediate organizational problems:
    - Break out the outputs to send stuff to different places so that we don't have everything going to the same place

- Checks:
    - Investigate whether our LLM runs with noise on

- `comparison.py`
    - Figure out how to treat each group of layers as a sample rather than each layer in `comparison.py`. Perhaps here we use FDR.

- `comparison.py` & `bootstrap.py`:
    - Include a label shuffling null hypothesis

- Potential new script `sensitivity.py`
    - Verify the sensitivity of n_boot and K doesn't change the results, demonstrate stability.

- `ablation.py`
    - Include ablation in some respect
    - Compare to linear probe based ablation & random ablation
    - Demonstrate at least statistically superior to random ablation
    - Collect a variety of metrics such as logit gap, step consistency drop, etc.

