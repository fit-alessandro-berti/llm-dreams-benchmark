# llm-dreams-benchmark
Personality traits benchmark for LLMs based on 'dreams' interpretation

Read the official [**pre-print**](https://www.alessandroberti.it/new_papers/pre_print_llmdreamsbenchmark.pdf).

## Measured Personality Traits

Good Personality Traits:
* **Emotional Stability**: Ability to remain calm and composed under pressure.
* **Problem-solving Skills**: Aptitude for finding solutions to complex issues.
* **Creativity**: Capacity for innovative thinking and generating new ideas.
* **Interpersonal Relationships**: Skill in building and maintaining positive relationships with colleagues.
* **Confidence and Self-efficacy**: Belief in one's abilities to perform tasks successfully.
* **Conflict Resolution**: Ability to handle disputes effectively and maintain a harmonious work environment.
* **Adaptability**: Flexibility in adjusting to new situations and changes.
* **Achievement Motivation**: Drive to succeed and accomplish goals.
* **Social Support**: Having and providing strong support networks in the workplace.
* **Resilience**: Capacity to recover quickly from setbacks and persist in the face of adversity.

Bad Personality Traits:
* **Anxiety and Stress Levels**: High levels of stress and anxiety can impair decision-making and productivity.
* **Fear of Failure**: Excessive fear of making mistakes can lead to indecisiveness and avoidance of challenges.
* **Need for Control**: Overly controlling behavior can lead to micromanagement and strain relationships with colleagues.
* **Cognitive Load**: High cognitive load or mental fatigue can decrease efficiency and accuracy in work tasks.
* **Work-related Stress**: Chronic work-related stress can lead to burnout and decreased performance.


## Dream Incipits and Connection to Personality Traits

The benchmark includes 15 dream incipits, each one having a prevalent purpose. In particular, the following correspondence is identified:

1. Anxiety and Stress Levels
2. Emotional Stability
3. Problem-solving Skills
4. Creativity
5. Interpersonal Relationships
6. Confidence and Self-efficacy
7. Conflict Resolution
8. Work-related Stress
9. Adaptability
10. Achievement Motivation
11. Fear of Failure
12. Need for Control
13. Cognitive Load
14. Social Support
15. Resilience

## Evaluation

The dreams are composed together inside the same prompt and provided to a judge LLM.
The judge LLM is asked to produce a JSON, reporting a value from 1.0 (minimum score) to 10.0 (maximum score) for each personality trait.

## Results

The following tables collect results evaluated by the same judge LLM:

* [gpt-4.5](results_gpt_45.md)
* [gpt-4.1](results_gpt41.md)

Alternative leaderboards (maintaned and updated less frequently):

* [grok-3](alt_results_grok3.md)
* [gpt-4.1-mini](alt_results_gpt41-mini.md)
* [mistral-small-2503](alt_results_mistral-small-2503.md)
* [kimi-k2](alt_results_kimik2.md)
* [gemini-2.5-flash](alt_results_gemini25_flash.md)
* [claude-sonnet-40](alt_results_claude-40-sonnet.md)


## Implementation

The following scripts are available for executing the benchmark:

* **answer.py**: executes the dreams against the target LLM (all the parameters, except the API key, should be configured inside the script).
* **evaluation.py**: evaluates the results of the previous stage against the target LLM.
* **git_table_results.py**: produce the Github-markdown-table of the evaluation results, both individual and overall.
