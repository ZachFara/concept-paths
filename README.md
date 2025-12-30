
TODOs:
- Implement a seed that propgates throughout the repository
- Investigate whether our LLM runs with noise on
- Create a script which runs everything end to end
- Break out the outputs to send stuff to different places so that we don't have everything going to the same place
- Figure out how to treat each group of layers as a sample rather than each layer in `comparison.py`. Perhaps here we use FDR.
- Implement another LLM
- Implement another axis besides sentiment
- Verify the sensitivity of n_boot and K doesn't change the results, demonstrate stability.
- Include a label shuffling null hypothesis
- Include ablation in some respect
