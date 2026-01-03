TODOs:

- Make something that demonstrate that the directions between different concepts for the same model are different. Perhaps `directions.py`

- `comparison.py` & `bootstrap.py`:
    - Include a label shuffling null hypothesis

- Potential new script `sensitivity.py`
    - Verify the sensitivity of n_boot and K doesn't change the results, demonstrate stability.

- `ablation.py`
    - DONE! Include ablation in some respect
    - DONE! Compare to linear probe based ablation & random ablation
    - DONE! Demonstrate at least statistically superior to random ablation
    - Collect a variety of metrics such as 
        - DONE! logit gap
        - DONE! Accuracy drop
        - step consistency drop


THOUGHTS:
- For a second axis it would be good to include something that could demonstrate tracing safety related axis like rudeness vs. politeness
- Since sentiment will correlated with rudeness vs. politeness, let's include a third axis of formality

