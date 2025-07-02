# Taskflow

## Agents
- [x] Plan work
- [ ] Document code (classes and functions)
- [x] Evaluator should answer if a score (1..5) and it should not 'break the build'
- [ ] Review diffs
    - [x] Simple diff
    - [ ] Allow the user to specify requirements (ddd, ports-and-adapters...)
- [x] Write a commit message
- [ ] Agents
    - [ ] Coder
    - [ ] Architect
    - [ ] QA

## Tools
- [x] Get diff from a source control system (gitlab)
- [x] Get diff from a source control system (github)
- [ ] Approve merge request in gitlab
- [ ] Internet search
- [ ] Wikipedia
- [ ] CodeGraph (stractor)
    - [ ] Structure

## TODO
- [ ] Some tools need approval to be used, should be ease to configure
- [ ] Feedback
    - [ ] Store prompts and outputs
    - [ ] Store tools selection and outputs
    - [ ] Gather user feedback
- [ ] Allow the prompt to be passed in the terminal
- [ ] Improve the memory to allow reproducibility
    - [ ] Continue process in cases of failure
- [ ] Extract the prompts from the code
- [ ] Add costs/token usage
    - [ ] Store somewhere
- [ ] Deal with exceptions

